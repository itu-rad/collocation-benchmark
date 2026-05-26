"""Bandwidth-contention analysis for the VQA A vs B experiment.

Compares VQA pipelines that run the *same* workload (CLIP-Large vision
encode → FAISS over 10 k COCO captions → Qwen 3.5-9B answer) on
*different* accelerator mappings and *different* scheduling regimes.

Cells of the experiment (each is a separate run, saved with `--label`):

    A_pipelined  CLIP on MPS,        LLM on MPS, serialize_queries=False
    A_serial     CLIP on MPS,        LLM on MPS, serialize_queries=True
    B_pipelined  CLIP on ANE/CoreML, LLM on MPS, serialize_queries=False
    B_serial     CLIP on ANE/CoreML, LLM on MPS, serialize_queries=True

Interpretation:
- A_pipelined vs A_serial  → contention cost on A (all stages on MPS)
- B_pipelined vs B_serial  → contention cost on B (heterogeneous)
- B_pipelined vs A_pipelined → heterogeneity benefit *under* contention
- B_serial    vs A_serial    → heterogeneity benefit *without* contention

If unified memory bandwidth is the bottleneck, the heterogeneity benefit
should collapse under contention (A_pipelined ≈ B_pipelined) even though
B_serial may still beat A_serial.

Usage:
    python evaluation/scripts/bandwidth_analysis.py                       # default 2-way
    python evaluation/scripts/bandwidth_analysis.py vqa_a_pipe vqa_b_pipe # specific pair
    python evaluation/scripts/bandwidth_analysis.py --cells               # 4-way 2x2
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


RESULTS_DIR = Path("evaluation/results")
REPORT_PATH = RESULTS_DIR / "bandwidth_report.md"

# Stage → device for each pipeline label (used for per-accelerator busy time).
# Matches both the original A/B labels and the 2×2 cell labels below.
_BASE_DEVICE_MAP = {
    "VQA dataloader": "CPU",
    "FAISS retriever": "CPU",
    "VQA formatter": "CPU",
    "End stage": "CPU",
    "LLM generator": "MPS",
}
DEVICE_MAP = {
    "vqa_mps_monolith":         {**_BASE_DEVICE_MAP, "CLIP vision encoder": "MPS"},
    "vqa_heterogeneous_split":  {**_BASE_DEVICE_MAP, "CLIP vision encoder (CoreML)": "ANE"},
    # 2x2 cell labels
    "vqa_a_pipe":   {**_BASE_DEVICE_MAP, "CLIP vision encoder": "MPS"},
    "vqa_a_serial": {**_BASE_DEVICE_MAP, "CLIP vision encoder": "MPS"},
    "vqa_b_pipe":   {**_BASE_DEVICE_MAP, "CLIP vision encoder (CoreML)": "ANE"},
    "vqa_b_serial": {**_BASE_DEVICE_MAP, "CLIP vision encoder (CoreML)": "ANE"},
}

# Rosetta RAG on a single unified-memory GPU (DGX Spark / GB10): every LLM
# stage runs on cuda; dataloader / retriever / formatters / routers / capture
# are CPU-side. T1 = 9B monolith, T2 = 3x 4B distributed. Stage names match
# pipeline_configs/rosetta_topology_{1,2}_cuda.yml verbatim.
_ROSETTA_T1_DEVICE_MAP = {
    "Question dataloader": "CPU",
    "Document retriever": "CPU",
    "Monolith formatter": "CPU",
    "Monolith LLM": "cuda",
    "Monolith router": "CPU",
    "End stage": "CPU",
    "Query rewrite formatter": "CPU",
    "Query rewrite LLM": "cuda",
}
_ROSETTA_T2_DEVICE_MAP = {
    "Question dataloader": "CPU",
    "Document retriever": "CPU",
    "Retrieval grader formatter": "CPU",
    "Grader LLM": "cuda",
    "Grade router": "CPU",
    "Answer generator formatter": "CPU",
    "Generator LLM": "cuda",
    "Hallucination grader formatter": "CPU",
    "Hallucination LLM": "cuda",
    "Hallucination router": "CPU",
    "Query rewrite formatter": "CPU",
    "Query rewrite LLM": "cuda",
    "End stage": "CPU",
}
DEVICE_MAP.update({
    "rosetta_t1_pipe":   _ROSETTA_T1_DEVICE_MAP,
    "rosetta_t1_serial": _ROSETTA_T1_DEVICE_MAP,
    "rosetta_t2_pipe":   _ROSETTA_T2_DEVICE_MAP,
    "rosetta_t2_serial": _ROSETTA_T2_DEVICE_MAP,
    # HotpotQA multi-hop variant — identical stage→device layout.
    "rosetta_hotpot_t1_pipe":   _ROSETTA_T1_DEVICE_MAP,
    "rosetta_hotpot_t1_serial": _ROSETTA_T1_DEVICE_MAP,
    "rosetta_hotpot_t2_pipe":   _ROSETTA_T2_DEVICE_MAP,
    "rosetta_hotpot_t2_serial": _ROSETTA_T2_DEVICE_MAP,
})

DEFAULT_PAIR = ("vqa_mps_monolith", "vqa_heterogeneous_split")
DEFAULT_CELLS = ("vqa_a_pipe", "vqa_a_serial", "vqa_b_pipe", "vqa_b_serial")


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PipelineTrace:
    name: str
    pipeline_label: str  # human-readable from CSV (e.g. "VQA MPS Monolith")
    prepare: Interval | None = None
    stage_runs: dict[str, list[Interval]] = field(default_factory=lambda: defaultdict(list))
    stage_prepares: dict[str, Interval] = field(default_factory=dict)
    query_runs: dict[tuple, Interval] = field(default_factory=dict)


def _parse_csv(csv_path: Path) -> PipelineTrace:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    trace = PipelineTrace(name=csv_path.stem, pipeline_label="")
    open_runs: dict[tuple, list[float]] = defaultdict(list)
    open_pipeline_runs: dict[tuple, float] = {}

    for raw in csv_path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 5:
            continue
        ts = float(parts[0])
        pipeline_name = parts[1]
        stage = parts[2]
        phase = parts[3]
        state = parts[4]
        if not trace.pipeline_label:
            trace.pipeline_label = pipeline_name

        if stage == "pipeline" and phase == "prepare":
            if state == "start":
                trace.prepare = Interval(start=ts, end=ts)
            elif state == "end" and trace.prepare is not None:
                trace.prepare.end = ts
        elif stage.startswith("pipeline - "):
            # "pipeline - val, run, start/end, qid, ts, epoch, batch"
            if len(parts) >= 9:
                key = (parts[5], parts[7], parts[8])
                if state == "start":
                    open_pipeline_runs[key] = ts
                elif state == "end":
                    start = open_pipeline_runs.pop(key, None)
                    if start is not None:
                        trace.query_runs[key] = Interval(start=start, end=ts)
        elif phase == "prepare":
            if state == "start":
                open_runs[(stage, "prepare")].append(ts)
            elif state == "end" and open_runs[(stage, "prepare")]:
                start = open_runs[(stage, "prepare")].pop()
                trace.stage_prepares[stage] = Interval(start=start, end=ts)
        elif phase == "run":
            if state == "start":
                open_runs[(stage, "run")].append(ts)
            elif state == "end" and open_runs[(stage, "run")]:
                start = open_runs[(stage, "run")].pop()
                trace.stage_runs[stage].append(Interval(start=start, end=ts))
    return trace


def _query_stats(trace: PipelineTrace) -> dict:
    durs = [i.duration for i in trace.query_runs.values()]
    if not durs:
        return {"count": 0}
    durs_sorted = sorted(durs)
    p_idx = max(0, int(0.95 * len(durs)) - 1)
    return {
        "count": len(durs),
        "mean_s": statistics.mean(durs),
        "median_s": statistics.median(durs),
        "min_s": min(durs),
        "max_s": max(durs),
        "total_s": sum(durs),
        "p95_s": durs_sorted[p_idx],
    }


def _wall_time(trace: PipelineTrace) -> float:
    if not trace.query_runs:
        return 0.0
    starts = [i.start for i in trace.query_runs.values()]
    ends = [i.end for i in trace.query_runs.values()]
    return max(ends) - min(starts)


def _throughput(trace: PipelineTrace) -> float:
    wall = _wall_time(trace)
    return (len(trace.query_runs) / wall) if wall > 0 else 0.0


def _concurrency_profile(trace: PipelineTrace) -> dict:
    events: list[tuple[float, int]] = []
    for intervals in trace.stage_runs.values():
        for iv in intervals:
            events.append((iv.start, +1))
            events.append((iv.end, -1))
    if not events:
        return {"max_concurrent": 0, "time_at_level": {}, "mean_level": 0.0}
    events.sort()
    level = 0
    last_t = events[0][0]
    time_at: dict[int, float] = defaultdict(float)
    for t, delta in events:
        time_at[level] += t - last_t
        level += delta
        last_t = t
    total = sum(time_at.values())
    mean_level = sum(l * t for l, t in time_at.items()) / total if total > 0 else 0.0
    return {
        "max_concurrent": max(time_at.keys()) if time_at else 0,
        "time_at_level": dict(time_at),
        "mean_level": mean_level,
    }


def _device_busy(trace: PipelineTrace, pipeline_key: str) -> dict[str, float]:
    """Merged per-device busy time (union of intervals on each device)."""
    devmap = DEVICE_MAP.get(pipeline_key, {})
    per_dev_intervals: dict[str, list[Interval]] = defaultdict(list)
    for stage, intervals in trace.stage_runs.items():
        dev = devmap.get(stage, "unknown")
        per_dev_intervals[dev].extend(intervals)
    busy: dict[str, float] = {}
    for dev, intervals in per_dev_intervals.items():
        intervals = sorted(intervals, key=lambda i: i.start)
        merged = 0.0
        cur_start = cur_end = None
        for iv in intervals:
            if cur_start is None:
                cur_start, cur_end = iv.start, iv.end
            elif iv.start <= cur_end:
                cur_end = max(cur_end, iv.end)
            else:
                merged += cur_end - cur_start
                cur_start, cur_end = iv.start, iv.end
        if cur_start is not None:
            merged += cur_end - cur_start
        busy[dev] = merged
    return busy


def _render_cell_summary(label: str, trace: PipelineTrace, pipeline_key: str) -> list[str]:
    q = _query_stats(trace)
    if q.get("count", 0) == 0:
        return [f"### {label}", "", "_no query data_", ""]
    wall = _wall_time(trace)
    tput = _throughput(trace)
    conc = _concurrency_profile(trace)
    dev = _device_busy(trace, pipeline_key)
    out = [f"### {label} ({trace.pipeline_label})", ""]
    out.append(
        f"- queries: **{q['count']}** | wall: **{wall:.2f} s** | "
        f"throughput: **{tput:.3f} q/s**"
    )
    out.append(
        f"- per-query latency: mean **{q['mean_s']:.3f} s**, "
        f"median {q['median_s']:.3f} s, p95 {q['p95_s']:.3f} s, "
        f"max {q['max_s']:.3f} s"
    )
    out.append(
        f"- stage concurrency: max {conc['max_concurrent']}, "
        f"mean **{conc['mean_level']:.2f}** stages running in parallel"
    )
    if dev:
        dev_strs = ", ".join(
            f"{d} **{t:.1f} s** ({t/wall*100:.0f}%)" for d, t in sorted(dev.items())
        )
        out.append(f"- device busy: {dev_strs}")
    out.append("")
    return out


def _render_pair_comparison(label_a: str, trace_a: PipelineTrace,
                            label_b: str, trace_b: PipelineTrace,
                            interpretation: str) -> list[str]:
    qa = _query_stats(trace_a)
    qb = _query_stats(trace_b)
    if not qa.get("mean_s") or not qb.get("mean_s"):
        return [f"### {label_a} vs {label_b}", "", "_insufficient data_", ""]
    wall_a = _wall_time(trace_a)
    wall_b = _wall_time(trace_b)
    tput_a = _throughput(trace_a)
    tput_b = _throughput(trace_b)

    delta_lat = (qb["mean_s"] - qa["mean_s"]) / qa["mean_s"]
    delta_wall = (wall_b - wall_a) / wall_a if wall_a > 0 else 0
    delta_tput = (tput_b - tput_a) / tput_a if tput_a > 0 else 0

    out = [f"### {label_a}  →  {label_b}", "", f"_{interpretation}_", ""]
    out.append("| | mean latency | wall | throughput |")
    out.append("|---|---|---|---|")
    out.append(f"| {label_a} | {qa['mean_s']:.3f} s | {wall_a:.2f} s | {tput_a:.3f} q/s |")
    out.append(f"| {label_b} | {qb['mean_s']:.3f} s | {wall_b:.2f} s | {tput_b:.3f} q/s |")
    out.append(
        f"| Δ (B vs A) | **{delta_lat*100:+.1f}%** | "
        f"**{delta_wall*100:+.1f}%** | **{delta_tput*100:+.1f}%** |"
    )
    out.append("")
    return out


def _render_2x2(cells: dict[str, PipelineTrace],
                pipeline_key_for: dict[str, str] | None = None) -> str:
    """Render full 2x2 report from 4 cell traces.

    cells keys: "A_pipelined", "A_serial", "B_pipelined", "B_serial"
    pipeline_key_for maps those cell labels → the DEVICE_MAP key (i.e. the
    actual run label/stem). Defaults to the VQA labels for backward
    compatibility; callers pass the real stems so non-VQA experiments
    (e.g. Rosetta) resolve their own DEVICE_MAP entries.
    """
    if pipeline_key_for is None:
        pipeline_key_for = {
            "A_pipelined": "vqa_a_pipe",
            "A_serial":    "vqa_a_serial",
            "B_pipelined": "vqa_b_pipe",
            "B_serial":    "vqa_b_serial",
        }
    out: list[str] = ["# Bandwidth-contention 2×2 experiment", ""]
    out.append(
        "Four cells across two orthogonal axes:\n\n"
        "- **Mapping**: A = all stages on MPS · B = CLIP on ANE/CoreML, LLM on MPS\n"
        "- **Schedule**: pipelined = multiple queries in flight · "
        "serial = `serialize_queries: True` (one query end-to-end at a time)\n"
    )

    out.append("## Per-cell summary")
    out.append("")
    for label in ("A_pipelined", "A_serial", "B_pipelined", "B_serial"):
        if label in cells:
            out.extend(_render_cell_summary(label, cells[label], pipeline_key_for[label]))

    out.append("## Pairwise comparisons")
    out.append("")
    pairs = [
        ("A_serial", "A_pipelined",
         "Same hardware, pipelining ON vs OFF — contention cost on the MPS-monolith mapping. "
         "If pipelining helps (B faster), parallelism payoff > contention cost. "
         "If it hurts/no-change, MPS is fully saturated already."),
        ("B_serial", "B_pipelined",
         "Same hardware, pipelining ON vs OFF — contention cost on the heterogeneous mapping. "
         "CLIP on ANE and LLM on MPS *could* run in parallel; the question is whether shared "
         "unified memory lets them."),
        ("A_pipelined", "B_pipelined",
         "Pipelined throughput: monolith vs heterogeneous. The bandwidth thesis predicts "
         "B ≈ A here (heterogeneity advantage collapses under shared-memory contention)."),
        ("A_serial", "B_serial",
         "Serial latency: monolith vs heterogeneous. Without contention, B should win by "
         "the CLIP-on-ANE-vs-MPS speed difference."),
    ]
    for la, lb, interp in pairs:
        if la in cells and lb in cells:
            out.extend(_render_pair_comparison(la, cells[la], lb, cells[lb], interp))

    # Synthesis verdict.
    out.append("## Verdict")
    out.append("")
    needed = {"A_pipelined", "A_serial", "B_pipelined", "B_serial"}
    if needed.issubset(cells):
        qa_p = _query_stats(cells["A_pipelined"])
        qa_s = _query_stats(cells["A_serial"])
        qb_p = _query_stats(cells["B_pipelined"])
        qb_s = _query_stats(cells["B_serial"])
        if all(s.get("mean_s") for s in (qa_p, qa_s, qb_p, qb_s)):
            # Heterogeneity advantage with and without contention.
            adv_serial = (qa_s["mean_s"] - qb_s["mean_s"]) / qa_s["mean_s"] * 100
            adv_pipe = (qa_p["mean_s"] - qb_p["mean_s"]) / qa_p["mean_s"] * 100
            collapse = adv_serial - adv_pipe
            lines = [
                f"- Heterogeneity advantage **without** contention "
                f"(serial): B is {adv_serial:+.1f}% faster than A.",
                f"- Heterogeneity advantage **under** contention "
                f"(pipelined): B is {adv_pipe:+.1f}% faster than A.",
                f"- **Advantage collapse**: {collapse:+.1f} pp lost when contention turned on.",
                "",
            ]
            if collapse > 5 and adv_pipe < adv_serial * 0.5:
                lines.append(
                    "**Verdict: bandwidth-bound under load.** Heterogeneity gives a real "
                    "single-query speedup but most of that advantage disappears once multiple "
                    "queries contend for shared unified memory — consistent with the thesis "
                    "that bandwidth is the binding constraint."
                )
            elif adv_pipe > 0.7 * adv_serial:
                lines.append(
                    "**Verdict: compute-bound.** Heterogeneity advantage persists under "
                    "contention, suggesting MPS compute (not bandwidth) was the bottleneck. "
                    "Bandwidth contention is not yet the dominant cost at this workload size."
                )
            else:
                lines.append(
                    "**Verdict: mixed.** Some erosion of heterogeneity advantage under "
                    "contention, but not enough to confidently call it bandwidth-bound. "
                    "Try larger queries-in-flight or a heavier per-stage workload."
                )
            out.extend(lines)
    else:
        missing = sorted(needed - set(cells))
        out.append(f"_Missing cells: {missing}_")
    out.append("")
    return "\n".join(out)


def _render_simple_pair(trace_a: PipelineTrace, trace_b: PipelineTrace,
                        key_a: str, key_b: str) -> str:
    """Original 2-way report retained for backward compatibility."""
    qa = _query_stats(trace_a)
    qb = _query_stats(trace_b)
    out = [
        f"# Bandwidth-contention report: {trace_a.pipeline_label} vs {trace_b.pipeline_label}",
        "",
    ]
    out.extend(_render_cell_summary("A", trace_a, key_a))
    out.extend(_render_cell_summary("B", trace_b, key_b))
    if qa.get("mean_s") and qb.get("mean_s"):
        delta = (qb["mean_s"] - qa["mean_s"]) / qa["mean_s"] * 100
        out.append(f"## Delta")
        out.append("")
        out.append(f"B vs A mean latency: **{delta:+.1f}%**")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipelines", nargs="*",
                        help="0, 2, or 4 pipeline label stems.")
    parser.add_argument("--cells", action="store_true",
                        help=f"Run the full 4-cell 2x2 report on default labels "
                             f"({', '.join(DEFAULT_CELLS)}).")
    parser.add_argument("--out", default=None,
                        help=f"Output report path (default: {REPORT_PATH}). "
                             f"Use a distinct path for non-VQA experiments so "
                             f"the default report is not clobbered.")
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else REPORT_PATH

    if args.cells or len(args.pipelines) == 4:
        labels = args.pipelines if len(args.pipelines) == 4 else list(DEFAULT_CELLS)
        cell_keys = ("A_pipelined", "A_serial", "B_pipelined", "B_serial")
        cells: dict[str, PipelineTrace] = {}
        pipeline_key_for: dict[str, str] = {}
        for cell, stem in zip(cell_keys, labels):
            pipeline_key_for[cell] = stem
            path = RESULTS_DIR / f"{stem}.csv"
            try:
                cells[cell] = _parse_csv(path)
            except FileNotFoundError:
                print(f"  WARN: missing {path}")
        report = _render_2x2(cells, pipeline_key_for)
    else:
        keys = args.pipelines if len(args.pipelines) == 2 else list(DEFAULT_PAIR)
        trace_a = _parse_csv(RESULTS_DIR / f"{keys[0]}.csv")
        trace_b = _parse_csv(RESULTS_DIR / f"{keys[1]}.csv")
        report = _render_simple_pair(trace_a, trace_b, keys[0], keys[1])

    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
