#!/usr/bin/env python3
"""Validation gate for collected cells — run repeatedly before full-R collection.

Checks, per collected run and rolled up per cell, that (a) everything is
tracked correctly, (b) every statistic the paper needs is computable from the
artifacts, and (c) the methodology's assumptions hold:

  1. TRACE INTEGRITY   parses; run start/end pairs complete; query_ids unique;
                       per-stage perf monotone; completed == expected
                       (loadgen max_queries / n_quality); no NaN/negative
                       latencies.
  2. SIDECARS          _arrivals.csv present for every open-loop run, 0 blocked
                       puts, realized rate within ±5% of intended (reuses
                       evaluation/pilots/verify_knobs.verify_traces);
                       _outputs.jsonl for quality cells: one record per query,
                       non-empty answers.
  3. INSTRUMENTATION   first_token sub-phase pairs 1:1 with the generator
                       stage's run pairs, ordered run_start <= ft_start <
                       ft_end <= run_end; n_generated_tokens in (0, max_tokens].
                       Traces without first_token rows are flagged
                       pre-instrumentation (pre-90b8726) when LLM stages exist.
  4. STATISTICS        per-cell p50/p95 (with the >=500-pooled p95 gate
                       accounting at planned R), per-run throughput, serial
                       capacity, EM/F1 via evaluation/scripts/score_quality,
                       hierarchical CIs where R >= 2.
  5. WARM-UP           pilot_lib.detect_warmup on each timing run's latency
                       series after dropping the knob's k; non-flat -> flag.
  6. RADT/MLFLOW       per run: matches the trace's wall window to an MLflow
                       run (sqlite store or a pre-dumped summary TSV), checks
                       listener metric samples > 0 and span export non-empty
                       (== 0 for _notrace cells).
  7. CROSS-RUN         per-run median dispersion (>20% spread flagged — the
                       E2-contamination class); normalized-answer agreement
                       across runs of the same cell (greedy determinism).

Every WARN/FAIL carries a class: "tracking" (bug in collection/tracking),
"stats" (a paper statistic is blocked), or "assumption" (methodology
assumption violated). Output: human matrix on stdout + machine JSON.

    python evaluation/collect/validate_pass.py --device mlx
    python evaluation/collect/validate_pass.py --device cuda \
        --results-dir /path/to/cuda_bf16_archive --archived-bf16 \
        --mlflow-summary /path/to/mlflow_summary.tsv --out /path/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "pilots"))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "contention"))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "scripts"))
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402
import pilot_lib as pl  # noqa: E402
import staged_lib as sl  # noqa: E402
import verify_knobs as vk  # noqa: E402
import score_quality as sq  # noqa: E402
from run_collection import build_cells, Cell, DEVNAME  # noqa: E402

NS = 1e9
P95_GATE = 500          # >=500 pooled queries for a reportable p95
DISPERSION_TOL = 0.20   # per-run median spread flag threshold
IN_PROGRESS_S = 20 * 60  # log newer than this + no CSV => in-progress, not FAIL

SEV = {"PASS": 0, "INFO": 0, "WARN": 1, "FAIL": 2}


def finding(status, check, msg, klass=None):
    return {"status": status, "check": check, "class": klass, "msg": msg}


# ---------------------------------------------------------------------------
# Cell/config resolution
# ---------------------------------------------------------------------------

def load_cell_meta(cell: Cell, device: str) -> dict:
    """Static expectations for a cell from its (knob-locked) committed config."""
    doc = yaml.safe_load((REPO_ROOT / cell.config).read_text(encoding="utf-8"))
    pipes = {}
    for pipe in doc.get("pipelines", []):
        lg = dict(pipe.get("loadgen") or {})
        if cell.quality_n:
            lg = {"component": "loadgen.OfflineLoadScheduler",
                  "max_queries": cell.quality_n, "config": {"rate": 0}}
        elif cell.loadgen_override:
            lg = dict(cell.loadgen_override)
        rate = float((lg.get("config") or {}).get("rate") or 0)
        comp = str(lg.get("component", ""))
        open_loop = rate > 0 and "Poisson" in comp
        max_tokens = {}
        for st in pipe.get("stages", []):
            cfg = st.get("config") or {}
            gk = ((cfg.get("model") or {}).get("gen_kwargs")
                  or cfg.get("gen_kwargs") or {})
            mt = gk.get("max_tokens") or gk.get("max_new_tokens")
            if mt:
                max_tokens[st["name"]] = int(mt)
        pipes[pipe["name"]] = {
            "expected_queries": int(lg.get("max_queries") or 0) or None,
            "rate": rate, "open_loop": open_loop,
            "queue_depth": lg.get("queue_depth"),
            "max_tokens": max_tokens,
            "has_llm": bool(max_tokens),
        }
    return {"cell": cell, "pipes": pipes,
            "quality": bool(cell.quality_n),
            "notrace": bool((cell.env or {}).get("CHOREO_DISABLE_TRACING")),
            "device": device}


# ---------------------------------------------------------------------------
# Per-run checks
# ---------------------------------------------------------------------------

def wall_window(csv_path: Path):
    """(first, last) wall timestamps of a trace CSV (column 0)."""
    t0 = t1 = None
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            head = line.split(",", 1)[0].strip()
            try:
                w = float(head)
            except ValueError:
                continue
            if t0 is None:
                t0 = w
            t1 = w
    return t0, t1


def check_trace_integrity(csv_path: Path, meta: dict) -> tuple[list, dict]:
    out = []
    try:
        traces = sl.parse_trace_files([csv_path])
    except Exception as e:  # noqa: BLE001
        return [finding("FAIL", "trace", f"unparseable: {e}", "tracking")], {}
    if not traces or not any(pt.queries for pt in traces.values()):
        return [finding("FAIL", "trace", "no pipeline run rows", "tracking")], {}
    for name, pt in traces.items():
        exp = None
        pm = meta["pipes"].get(name)
        # single-pipeline cells: fall back to the only configured pipeline
        if pm is None and len(meta["pipes"]) == 1:
            pm = next(iter(meta["pipes"].values()))
        if pm:
            exp = pm["expected_queries"]
        n, comp = len(pt.queries), len(pt.completed)
        if comp < n:
            out.append(finding("FAIL", "trace",
                               f"{name}: {n - comp} query run pairs incomplete "
                               f"(start without end)", "tracking"))
        qids = [q.query_id for q in pt.queries if q.query_id]
        if len(set(qids)) != len(qids):
            dup = len(qids) - len(set(qids))
            out.append(finding("FAIL", "trace",
                               f"{name}: {dup} duplicate query_ids "
                               f"(query_id-provenance class)", "tracking"))
        if exp is not None and comp != exp:
            out.append(finding("FAIL", "trace",
                               f"{name}: completed {comp} != expected {exp}",
                               "tracking"))
        bad = [q for q in pt.completed
               if not (q.latency_s > 0) or math.isnan(q.latency_s)]
        if bad:
            out.append(finding("FAIL", "trace",
                               f"{name}: {len(bad)} non-positive/NaN latencies",
                               "tracking"))
        for stage, unpaired in pt.stage_unpaired.items():
            if unpaired:
                out.append(finding("FAIL", "trace",
                                   f"{name}/{stage}: {unpaired} unpaired stage "
                                   f"events", "tracking"))
        for stage, pairs in pt.stage_execs.items():
            if any(e < s for s, e, _, _ in pairs):
                out.append(finding("FAIL", "trace",
                                   f"{name}/{stage}: non-monotone perf pair",
                                   "tracking"))
    if not out:
        out.append(finding("PASS", "trace", "integrity ok"))
    return out, traces


def check_sidecars(csv_path: Path, meta: dict, traces: dict) -> list:
    out = []
    open_loop = any(p["open_loop"] for p in meta["pipes"].values())
    lam = max((p["rate"] for p in meta["pipes"].values()), default=0) or None
    arr_path = csv_path.with_name(csv_path.stem + "_arrivals.csv")
    if open_loop:
        if not arr_path.exists():
            out.append(finding("FAIL", "sidecar",
                               "open-loop run missing _arrivals.csv", "tracking"))
        else:
            # reuse the knob-rule checker (R-QDEPTH / R-LAMBDA / 3-sigma nominal)
            ok, notes = vk.verify_traces(str(csv_path), lam, None)
            for n in notes:
                out.append(finding("FAIL" if not ok else "WARN", "sidecar",
                                   n, "assumption"))
            arr = pl.parse_arrivals(arr_path)
            exp = next((p["expected_queries"] for p in meta["pipes"].values()
                        if p["open_loop"]), None)
            if exp and arr.n != exp:
                out.append(finding("FAIL", "sidecar",
                                   f"arrivals rows {arr.n} != {exp}", "tracking"))
            if arr.blocked_puts:
                out.append(finding("FAIL", "sidecar",
                                   f"{arr.blocked_puts} blocked puts "
                                   f"(max {arr.max_block_s*1e3:.1f} ms)",
                                   "assumption"))
    elif arr_path.exists():
        out.append(finding("INFO", "sidecar",
                           "arrivals sidecar present on closed-loop run"))
    outputs = csv_path.with_name(csv_path.stem + "_outputs.jsonl")
    needs_outputs = meta["quality"]
    has_llm = any(p["has_llm"] for p in meta["pipes"].values())
    if outputs.exists():
        recs = sq.load_run(outputs)
        exp = next((p["expected_queries"] for p in meta["pipes"].values()), None)
        answered = sum(1 for r in recs if r.answer)
        questions = [r.question for r in recs if r.question]
        if needs_outputs and exp and len(recs) != exp:
            out.append(finding("FAIL", "sidecar",
                               f"outputs.jsonl {len(recs)} records != {exp}",
                               "stats"))
        if len(set(questions)) != len(questions):
            out.append(finding("WARN", "sidecar",
                               "duplicate questions in outputs.jsonl", "tracking"))
        if recs and answered == 0:
            out.append(finding("FAIL", "sidecar",
                               "ALL answers empty — capture bug, not model "
                               "behavior", "tracking"))
        elif recs and answered < len(recs):
            out.append(finding("WARN", "sidecar",
                               f"{len(recs)-answered}/{len(recs)} unanswered/"
                               f"empty answers (retry-exhaustion class; "
                               f"answered rate is itself a reported metric)",
                               "stats"))
    elif needs_outputs or (has_llm and meta["quality"]):
        out.append(finding("FAIL", "sidecar",
                           "quality cell missing _outputs.jsonl", "stats"))
    if not out:
        out.append(finding("PASS", "sidecar", "sidecars ok"))
    return out


def check_instrumentation(meta: dict, traces: dict, archived: bool) -> list:
    out = []
    any_ft = any(s.endswith("::first_token")
                 for pt in traces.values() for s in pt.stage_execs)
    has_llm = (any(p["has_llm"] for p in meta["pipes"].values()) or any_ft
               or any(pt.stage_token_counts for pt in traces.values()))
    if not has_llm:
        return [finding("PASS", "instr", "n/a (no generator stages)")]
    if not any_ft:
        msg = ("no first_token rows (pre-90b8726 trace) — TTFT/decode split "
               "not derivable from this run")
        if archived:
            return [finding("INFO", "instr", "ARCHIVED-bf16: " + msg, "stats")]
        # only the staged experiment's H2 verdict REQUIRES TTFT; elsewhere the
        # missing sub-phase is provenance info, not a statistic blocker
        sev = "WARN" if meta["cell"].phase == "staged" else "INFO"
        return [finding(sev, "instr", msg, "stats")]
    for name, pt in traces.items():
        pm = meta["pipes"].get(name) or (next(iter(meta["pipes"].values()))
                                         if len(meta["pipes"]) == 1 else {})
        for stage, ft_pairs in pt.stage_execs.items():
            if not stage.endswith("::first_token"):
                continue
            base = stage[:-len("::first_token")]
            run_pairs = pt.stage_execs.get(base, [])
            if len(ft_pairs) != len(run_pairs):
                out.append(finding("FAIL", "instr",
                                   f"{base}: {len(ft_pairs)} first_token pairs "
                                   f"vs {len(run_pairs)} run pairs", "tracking"))
                continue
            for k, ((fs, fe, _, _), (rs, re_, _, _)) in enumerate(
                    zip(ft_pairs, run_pairs)):
                if not (rs <= fs < fe <= re_):
                    out.append(finding("FAIL", "instr",
                                       f"{base}[{k}]: first_token pair outside "
                                       f"its run pair", "tracking"))
                    break
            toks = pt.stage_token_counts.get(base, [])
            mt = (pm.get("max_tokens") or {}).get(base)
            if len(toks) != len(run_pairs):
                out.append(finding("FAIL", "instr",
                                   f"{base}: {len(toks)} n_generated_tokens "
                                   f"rows vs {len(run_pairs)} run pairs",
                                   "tracking"))
            bad = [t for t in toks if t <= 0 or (mt and t > mt)]
            if bad:
                out.append(finding("FAIL", "instr",
                                   f"{base}: {len(bad)} token counts outside "
                                   f"(0, {mt or '?'}]", "tracking"))
    if not out:
        out.append(finding("PASS", "instr", "first_token/token rows consistent"))
    return out


def check_warmup(traces: dict, warmup_k: int) -> tuple[list, dict]:
    out, series = [], {}
    for name, pt in traces.items():
        x = [q.latency_s for q in pt.completed]
        if len(x) < 8:
            continue
        series[name] = x
        wu = pl.detect_warmup(x[warmup_k:], window=5, epsilon=0.10)
        if wu.converged and wu.k_star > 0:
            out.append(finding("WARN", "warmup",
                               f"{name}: still warming after knob k={warmup_k} "
                               f"(k*={wu.k_star} post-drop)", "assumption"))
        elif not wu.converged:
            out.append(finding("WARN", "warmup",
                               f"{name}: series not flat post-k "
                               f"({wu.note or 'no onset'})", "assumption"))
        if wu.outlier_idxs:
            out.append(finding("INFO", "warmup",
                               f"{name}: {len(wu.outlier_idxs)} first-call-class "
                               f"outliers at {wu.outlier_idxs[:5]}"))
    if not out:
        out.append(finding("PASS", "warmup", "flat after knob k"))
    return out, series


# ---------------------------------------------------------------------------
# MLflow / radt tracking
# ---------------------------------------------------------------------------

def load_mlflow_summary(db_path: Path | None, tsv_path: Path | None):
    """[(start_ms, run_uuid, status, n_metrics, n_spans)] sorted by start."""
    rows = []
    if tsv_path and tsv_path.exists():
        for line in tsv_path.read_text().splitlines():
            p = line.split("\t")
            if len(p) >= 5:
                rows.append((int(p[1]), p[0], p[2], int(p[3]), int(p[4])))
    elif db_path and db_path.exists():
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        spans = dict(con.execute(
            "select value, count(*) from trace_request_metadata "
            "where key='mlflow.sourceRun' group by value"))
        mets = dict(con.execute(
            "select run_uuid, count(*) from metrics group by run_uuid"))
        for uid, st, status in con.execute(
                "select run_uuid, start_time, status from runs"):
            rows.append((st, uid, status, mets.get(uid, 0), spans.get(uid, 0)))
        con.close()
    return sorted(rows)


def check_mlflow(csv_path: Path, meta: dict, ml_rows: list) -> list:
    if not ml_rows:
        return [finding("WARN", "radt",
                        "no MLflow store available to match against", "tracking")]
    t0, t1 = wall_window(csv_path)
    if t0 is None:
        return [finding("WARN", "radt", "no wall timestamps in trace", "tracking")]
    # the MLflow run starts (log_artifact) shortly before the first trace row;
    # take the latest run start <= t0(+5s), require it within 30 min.
    cand = [r for r in ml_rows if r[0] <= (t0 + 5) * 1000]
    if not cand or (t0 * 1000 - cand[-1][0]) > 30 * 60 * 1000:
        return [finding("WARN", "radt",
                        f"no MLflow run starts within 30 min before trace start "
                        f"(different store/host?)", "tracking")]
    start_ms, uid, status, n_metrics, n_spans = cand[-1]
    out = [finding("INFO", "radt",
                   f"matched run {uid[:12]} (start {time.strftime('%F %T', time.localtime(start_ms/1000))}, "
                   f"status={status}, listener_samples={n_metrics}, spans={n_spans})")]
    if n_metrics == 0:
        out.append(finding("FAIL", "radt",
                           "0 listener metric samples (macmon/top never "
                           "attached — run_collection bypasses "
                           "radt.schedule_external, so RADT_PRESENT/"
                           "RADT_LISTENER_* are never set on the -p 0 path)",
                           "tracking"))
    if meta["notrace"]:
        if n_spans > 0:
            out.append(finding("FAIL", "radt",
                               f"_notrace cell exported {n_spans} spans "
                               f"(CHOREO_DISABLE_TRACING ineffective)",
                               "tracking"))
    elif n_spans == 0:
        out.append(finding("FAIL", "radt",
                           "span export empty on a spans-on run", "tracking"))
    return out


# ---------------------------------------------------------------------------
# Cell-level statistics computability + cross-run consistency
# ---------------------------------------------------------------------------

def pctl(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] if s else float("nan")


def cell_statistics(cell_label: str, meta: dict, run_traces: dict,
                    warmup_k: int, planned_R: int) -> tuple[list, dict]:
    """run_traces: {run_id: {pipe_name: PipelineTrace}}."""
    out, stats = [], {}
    fg_vecs, throughputs, run_medians = [], [], {}
    for rid, traces in sorted(run_traces.items()):
        for name, pt in traces.items():
            lat = [q.latency_s for q in pt.completed][warmup_k:]
            if not lat:
                continue
            fg_vecs.append(lat)
            run_medians[f"{rid}:{name}" if len(traces) > 1 else rid] = pl.statistics.median(lat)
            t0, t1 = pt.span_wall()
            if t1 > t0:
                throughputs.append(len(pt.completed) / (t1 - t0))
    pooled = [v for vec in fg_vecs for v in vec]
    if not pooled:
        return [finding("FAIL", "stats", "no post-warm-up latencies pooled",
                        "stats")], stats
    stats["n_pooled"] = len(pooled)
    stats["p50_s"] = pctl(pooled, 0.50)
    stats["throughput_qps_mean"] = (sum(throughputs) / len(throughputs)
                                    if throughputs else float("nan"))
    stats["run_medians_s"] = {k: round(v, 4) for k, v in run_medians.items()}
    exp = next((p["expected_queries"] for p in meta["pipes"].values()
                if p["expected_queries"]), 0)
    planned_pooled = max(0, (exp - warmup_k)) * planned_R
    stats["planned_pooled_at_R"] = planned_pooled
    if len(pooled) >= P95_GATE:
        stats["p95_s"] = pctl(pooled, 0.95)
    else:
        stats["p95_s"] = None
        if not meta["quality"]:  # p95 is a timing-cell statistic
            sev = "INFO" if planned_pooled >= P95_GATE else "WARN"
            out.append(finding(sev, "stats",
                               f"p95 gate: pooled {len(pooled)} < {P95_GATE} "
                               f"now; at planned R={planned_R} pooled would be "
                               f"{planned_pooled} — "
                               + ("gate met once all runs land"
                                  if planned_pooled >= P95_GATE else
                                  "GATE UNREACHABLE at planned R: paper must "
                                  "raise max_queries/R or pre-register "
                                  "dropping p95 for this experiment"),
                               "stats"))
    # serial capacity from serial cells: 1 / median service time
    if "_serial" in cell_label:
        stats["serial_capacity_qps"] = 1.0 / stats["p50_s"] if stats["p50_s"] else None
    if len(fg_vecs) >= 2:
        lo, hi = sl.hier_boot_ci(fg_vecs, lambda a: float(sl.median(list(a))),
                                 n=2000)
        stats["p50_ci95_s"] = [round(lo, 4), round(hi, 4)]
        meds = list(run_medians.values())
        spread = (max(meds) - min(meds)) / pl.statistics.median(meds)
        stats["run_median_spread"] = round(spread, 3)
        if spread > DISPERSION_TOL:
            out.append(finding("WARN", "xrun",
                               f"per-run median spread {spread:.0%} > "
                               f"{DISPERSION_TOL:.0%} (E2-contamination class): "
                               f"{[round(m,3) for m in meds]}", "assumption"))
    else:
        out.append(finding("INFO", "stats",
                           "single run — CIs n/a (point estimates only)"))
    if not any(f["status"] in ("WARN", "FAIL") for f in out):
        out.append(finding("PASS", "stats",
                           f"p50/throughput{'/p95' if stats['p95_s'] else ''} "
                           f"computable (n={len(pooled)})"))
    return out, stats


def cell_quality(cell_label: str, out_paths: list[Path]) -> tuple[list, dict]:
    out, stats = [], {}
    try:
        s = sq.score_arm(cell_label, out_paths)
        stats = {"n_pooled": s["n_pooled"],
                 "em": round(s["em"]["pooled"], 4),
                 "em_wilson95": [round(v, 4) for v in s["em"]["wilson95"]],
                 "f1": round(s["f1"]["pooled"], 4),
                 "answered": round(s["answered"]["pooled"], 4)}
        out.append(finding("PASS", "stats",
                           f"EM={stats['em']:.3f} F1={stats['f1']:.3f} "
                           f"answered={stats['answered']:.3f} "
                           f"(N={stats['n_pooled']})"))
    except Exception as e:  # noqa: BLE001
        out.append(finding("FAIL", "stats", f"score_quality failed: {e}",
                           "stats"))
    return out, stats


def answer_agreement(out_paths: list[Path]) -> tuple[list, dict]:
    """Greedy-decoding determinism: identical normalized answers across runs."""
    if len(out_paths) < 2:
        return [], {}
    per_run = []
    for p in out_paths:
        per_run.append({r.question: sq.normalize_answer(r.answer)
                        for r in sq.load_run(p) if r.question and r.answer})
    shared = set.intersection(*(set(d) for d in per_run))
    if not shared:
        return [finding("WARN", "xrun", "no shared answered questions across "
                        "runs", "stats")], {}
    agree = sum(1 for q in shared
                if len({d[q] for d in per_run}) == 1)
    frac = agree / len(shared)
    st = {"answer_agreement": round(frac, 3), "shared_questions": len(shared)}
    if frac < 1.0:
        sev = "WARN" if frac >= 0.9 else "FAIL"
        return [finding(sev, "xrun",
                        f"greedy answers differ across runs on "
                        f"{len(shared)-agree}/{len(shared)} shared questions "
                        f"(agreement {frac:.0%})", "assumption")], st
    return [finding("PASS", "xrun",
                    f"answers identical across runs ({len(shared)} shared)")], st


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["mlx", "cuda"], required=True)
    ap.add_argument("--results-dir", default=None,
                    help="default: evaluation/collect/results/<device>")
    ap.add_argument("--mlflow-db", default=str(REPO_ROOT / "mlflow.db"))
    ap.add_argument("--mlflow-summary", default=None,
                    help="pre-dumped TSV (run_uuid, start_ms, status, "
                         "n_metrics, n_spans) — for remote stores")
    ap.add_argument("--archived-bf16", action="store_true",
                    help="ARCHIVED-bf16 mode: tracking/mechanics checks only; "
                         "numbers marked not-for-paper; missing first_token "
                         "instrumentation is expected")
    ap.add_argument("--planned-R", type=int, default=None,
                    help="override planned R for the p95-gate accounting "
                         "(default: each cell's own R from run_collection)")
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args()

    results_dir = Path(args.results_dir or (HERE / "results" / args.device))
    if not results_dir.is_dir():
        sys.exit(f"no results dir {results_dir}")
    tag = "archived_bf16" if args.archived_bf16 else args.device
    out_json = Path(args.out or (HERE / f"validate_report_{tag}.json"))

    knobs = pl.load_knobs()
    dn = DEVNAME[args.device]
    cells = {c.label + f"_{args.device}": load_cell_meta(c, args.device)
             for c in build_cells(args.device)
             if (REPO_ROOT / c.config).exists()}
    ml_rows = load_mlflow_summary(
        Path(args.mlflow_db) if args.mlflow_db else None,
        Path(args.mlflow_summary) if args.mlflow_summary else None)

    # discover runs
    run_re = re.compile(r"^(?P<cell>.+)_r(?P<r>\d+)$")
    runs_by_cell: dict[str, dict[int, Path]] = defaultdict(dict)
    logs_by_cell: dict[str, dict[int, Path]] = defaultdict(dict)
    for p in sorted(results_dir.iterdir()):
        if p.suffix == ".csv" and not p.name.endswith(("_arrivals.csv",
                                                       "_bandwidth.csv")):
            m = run_re.match(p.stem)
            if m:
                runs_by_cell[m["cell"]][int(m["r"])] = p
        elif p.suffix == ".log":
            m = run_re.match(p.stem)
            if m:
                logs_by_cell[m["cell"]][int(m["r"])] = p

    report = {"generated": time.strftime("%F %T"), "device": args.device,
              "results_dir": str(results_dir),
              "archived_bf16": args.archived_bf16,
              "git_commit": pl.repo_git_commit(), "cells": {}}
    now = time.time()

    for cell_label in sorted(set(runs_by_cell) | set(logs_by_cell)):
        meta = cells.get(cell_label)
        crep = {"runs": {}, "findings": [], "stats": {}}
        report["cells"][cell_label] = crep
        if meta is None:
            crep["findings"].append(finding(
                "WARN", "layout", "label not in run_collection cell table "
                "(stale/foreign artifact?)", "tracking"))
            continue
        cell: Cell = meta["cell"]
        exp_key = cell.phase.split("_")[0]
        warmup_k = int(pl.get_knob(knobs, exp_key, dn, "warmup_k", 1) or 1)
        planned_R = args.planned_R or cell.runs

        # failed / in-progress runs (log without curated csv)
        for r, logp in sorted(logs_by_cell.get(cell_label, {}).items()):
            if r in runs_by_cell.get(cell_label, {}):
                continue
            age = now - logp.stat().st_mtime
            if age < IN_PROGRESS_S:
                crep["runs"][f"r{r}"] = [finding("INFO", "layout",
                                                 "in progress (log active, no "
                                                 "curated CSV yet)")]
            else:
                sev = "WARN" if cell.tolerate_failure else "FAIL"
                orphan = REPO_ROOT / "evaluation" / "results" / f"{cell_label}_r{r}.csv"
                extra = (f"; orphan trace left in evaluation/results/ "
                         f"({orphan.stat().st_size} B)" if orphan.exists() else "")
                crep["runs"][f"r{r}"] = [finding(
                    sev, "layout", f"run failed (log present, no curated CSV; "
                    f"log idle {age/60:.0f} min){extra}", "tracking")]

        run_traces, out_paths = {}, []
        for r, csvp in sorted(runs_by_cell.get(cell_label, {}).items()):
            fnds, traces = check_trace_integrity(csvp, meta)
            if traces:
                fnds += check_sidecars(csvp, meta, traces)
                fnds += check_instrumentation(meta, traces, args.archived_bf16)
                wu_f, _ = check_warmup(traces, warmup_k)
                fnds += wu_f
                fnds += check_mlflow(csvp, meta, ml_rows)
                run_traces[f"r{r}"] = traces
            crep["runs"][f"r{r}"] = fnds
            op = csvp.with_name(csvp.stem + "_outputs.jsonl")
            if op.exists():
                out_paths.append(op)

        # cell-level statistics + cross-run
        if run_traces:
            sf, stats = cell_statistics(cell_label, meta, run_traces,
                                        warmup_k, planned_R)
            crep["findings"] += sf
            crep["stats"].update(stats)
            if meta["quality"] and out_paths:
                qf, qstats = cell_quality(cell_label, out_paths)
                crep["findings"] += qf
                crep["stats"]["quality"] = qstats
            af, astats = answer_agreement(out_paths)
            crep["findings"] += af
            crep["stats"].update(astats)
        n_present = len(runs_by_cell.get(cell_label, {}))
        if n_present < cell.runs:
            crep["findings"].append(finding(
                "INFO", "layout",
                f"{n_present}/{cell.runs} planned runs present"))
        if args.archived_bf16:
            crep["findings"].append(finding(
                "INFO", "layout", "ARCHIVED-bf16 — mechanics-only validation; "
                "numbers are NOT paper inputs"))

    # roll-up
    def worst(fnds):
        return max((SEV[f["status"]] for f in fnds), default=0)

    matrix = []
    for cell_label, crep in report["cells"].items():
        sev = max([worst(crep["findings"])]
                  + [worst(f) for f in crep["runs"].values()])
        verdict = {0: "PASS", 1: "WARN", 2: "FAIL"}[sev]
        if not crep["stats"] and all(
                f["check"] == "layout" and f["status"] == "INFO"
                for fl in crep["runs"].values() for f in fl):
            verdict = "PEND"  # nothing curated yet (in-progress/queued)
        crep["verdict"] = verdict
        reasons = [f"{f['check']}: {f['msg']}"
                   for fl in ([crep["findings"]] + list(crep["runs"].values()))
                   for f in fl if SEV[f["status"]] == sev and sev > 0]
        crep["top_reasons"] = reasons[:4]
        matrix.append((cell_label, verdict, reasons[:1]))

    out_json.write_text(json.dumps(report, indent=1, default=str),
                        encoding="utf-8")
    width = max((len(c) for c, _, _ in matrix), default=10)
    print(f"\n== validate_pass {tag} — {len(matrix)} cells "
          f"(report: {out_json}) ==")
    for cell_label, verdict, reason in matrix:
        print(f"  {cell_label:<{width}s}  {verdict:4s}  "
              f"{reason[0] if reason else ''}")
    n_fail = sum(1 for _, v, _ in matrix if v == "FAIL")
    print(f"\n  {sum(1 for _, v, _ in matrix if v == 'PASS')} PASS / "
          f"{sum(1 for _, v, _ in matrix if v == 'WARN')} WARN / "
          f"{n_fail} FAIL / "
          f"{sum(1 for _, v, _ in matrix if v == 'PEND')} PEND")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
