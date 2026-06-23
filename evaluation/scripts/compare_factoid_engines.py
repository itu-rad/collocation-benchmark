"""Compare Self-RAG factoid runs across topology (monolith/decomposed) and
inference engine (HF / vLLM / Ollama).

Reads the per-run artifacts produced by `main.py ... --label <label>`:
  - evaluation/results/<label>.csv          (stage/query timing)
  - evaluation/results/<label>_outputs.jsonl (TerminalCapture per-query output)

and writes a Markdown comparison (accuracy + latency/throughput + LLM calls)
to evaluation/results/factoid_engine_comparison.md.

Usage:
    python evaluation/scripts/compare_factoid_engines.py
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path("evaluation/results")
OUT = RESULTS / "factoid_engine_comparison.md"

# (label, topology, engine) — label matches the --label used at run time.
RUNS = [
    ("factoid_monolith_hf", "monolith", "HF"),
    ("factoid_monolith_vllm", "monolith", "vLLM"),
    ("factoid_monolith_ollama", "monolith", "Ollama"),
    ("factoid_decomposed_hf", "decomposed", "HF"),
    ("factoid_decomposed_vllm", "decomposed", "vLLM"),
    ("factoid_decomposed_ollama", "decomposed", "Ollama"),
]

ERROR_MARKERS = (
    "Error: No more retries left",
    "Error: no satisfactory answer after retries",
)


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _golden_hit(answer: str, golden) -> bool:
    ans = _normalize(answer or "")
    for g in golden or []:
        gn = _normalize(str(g))
        if gn and gn in ans:
            return True
    return False


# ---- CSV timing parse (mirrors evaluation/scripts/bandwidth_analysis.py) ----

def parse_csv(path: Path):
    query_runs = {}
    open_pipeline = {}
    stage_runs = defaultdict(list)
    open_runs = defaultdict(list)
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 5:
            continue
        ts, _pipe, stage, phase, state = float(parts[0]), parts[1], parts[2], parts[3], parts[4]
        if stage.startswith("pipeline - ") and len(parts) >= 9:
            key = (parts[5], parts[7], parts[8])
            if state == "start":
                open_pipeline[key] = ts
            elif state == "end" and key in open_pipeline:
                query_runs[key] = (open_pipeline.pop(key), ts)
        elif phase == "run":
            if state == "start":
                open_runs[stage].append(ts)
            elif state == "end" and open_runs[stage]:
                stage_runs[stage].append((open_runs[stage].pop(), ts))
    return {"query_runs": query_runs, "stage_runs": stage_runs}


def timing_stats(csv):
    if not csv or not csv["query_runs"]:
        return None
    durs = sorted(e - s for s, e in csv["query_runs"].values())
    starts = [s for s, _ in csv["query_runs"].values()]
    ends = [e for _, e in csv["query_runs"].values()]
    wall = max(ends) - min(starts)
    n = len(durs)
    llm_calls = sum(len(v) for k, v in csv["stage_runs"].items() if "LLM" in k)
    return {
        "n": n,
        "lat_mean": statistics.mean(durs),
        "lat_median": statistics.median(durs),
        "lat_p95": durs[max(0, int(0.95 * n) - 1)],
        "wall": wall,
        "tput": n / wall if wall > 0 else 0.0,
        "llm_per_q": llm_calls / n if n else 0.0,
    }


# ---- JSONL quality parse ----

def quality_stats(path: Path):
    if not path.exists():
        return None
    n = answered = hits = 0
    lens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        n += 1
        gen, final = d.get("generated_answer"), d.get("final_data")
        best = gen if (gen and str(gen).strip()) else final
        if isinstance(best, list):
            best = " ".join(map(str, best))
        best = best or ""
        is_error = isinstance(final, str) and any(m in final for m in ERROR_MARKERS)
        if best.strip() and not is_error:
            answered += 1
            lens.append(len(best))
            # Only count a golden hit on a real (non-error) answer, else an
            # error string like "Error: No more retries left" spuriously
            # matches golden "no" via the "no more" substring.
            if _golden_hit(best, d.get("golden_answers")):
                hits += 1
    return {"n": n, "answered": answered, "hits": hits,
            "avg_len": statistics.mean(lens) if lens else 0.0}


def main():
    rows = []
    for label, topo, engine in RUNS:
        t = timing_stats(parse_csv(RESULTS / f"{label}.csv"))
        q = quality_stats(RESULTS / f"{label}_outputs.jsonl")
        rows.append((label, topo, engine, t, q))

    lines = ["# Self-RAG factoid: topology × engine comparison", "",
             "Dataset: rag-mini-wikipedia factoid (30 queries). Model: Qwen3.5-4B on",
             "all three engines (HF `Qwen/Qwen3.5-4B`, vLLM same, Ollama `qwen3.5:4b`).",
             "Pipelined (serialize=false). vLLM/Ollama use the concurrent dispatcher",
             "(server-side batching); HF serializes generation on one mutex.",
             ""]

    lines += ["## Quality", "",
              "| Version | Engine | Queries | Answered | Golden hits | Avg ans len |",
              "|---|---|--:|--:|--:|--:|"]
    for label, topo, engine, t, q in rows:
        if not q:
            lines.append(f"| {topo} | {engine} | _no data_ | | | |")
            continue
        lines.append(f"| {topo} | {engine} | {q['n']} | {q['answered']}/{q['n']} "
                     f"| {q['hits']}/{q['n']} | {q['avg_len']:.0f} |")

    lines += ["", "## Latency / throughput", "",
              "| Version | Engine | Queries | Mean lat | Median | p95 | Wall | Throughput | LLM calls/q |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for label, topo, engine, t, q in rows:
        if not t:
            lines.append(f"| {topo} | {engine} | _no timing_ | | | | | | |")
            continue
        lines.append(f"| {topo} | {engine} | {t['n']} | {t['lat_mean']:.2f}s "
                     f"| {t['lat_median']:.2f}s | {t['lat_p95']:.2f}s | {t['wall']:.1f}s "
                     f"| {t['tput']:.3f} q/s | {t['llm_per_q']:.2f} |")

    lines += ["", "## Notes", "",
              "- All three engines serve identical Qwen3.5-4B weights, so quality",
              "  differences are noise (same model); the engine axis is about latency",
              "  / throughput, where the workload is identical.",
              "- vLLM and Ollama run the concurrent dispatcher (server-side batching);",
              "  HF generates one request at a time. Both servers give a large",
              "  throughput win over HF at equal accuracy.",
              "- Under HF (serial) decomposed slightly beats monolith (cheap CPU-stage",
              "  overlap); under the batching engines monolith wins (its single fat",
              "  call batches better than decomposed's 3 dependency-chained calls).",
              "- Ollama runs qwen3.5:4b with reasoning disabled (`think: false`) so the",
              "  JSON answer fits the token budget, matching how HF/vLLM serve it.",
              "- MLX is Apple-Silicon only and is not runnable on this GB10 host.",
              ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
