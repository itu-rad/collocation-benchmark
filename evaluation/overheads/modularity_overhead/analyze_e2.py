#!/usr/bin/env python3
"""E2 — the cost of decomposition. SELF-CONTAINED: parsing, statistics, tables,
LaTeX and figures all live in this one file (imports nothing local except
utils/span_reader.py for the span-derived breakdown).

Three things run per cell, on the identical workload (EfficientNetV2 fine-tune on
Imagenette, frozen backbone; one query = one batch = one training step):

    monolith        a hand-written PyTorch loop, no framework
    choreo          the same workload declared as a Choreo pipeline, tracing off
    choreo-traced   the same Choreo pipeline with radt proc tracing on

    cost of choreo   = choreo - monolith        -> the framework wrapper itself
    cost of tracing  = choreo-traced - choreo   -> what turning tracing on adds

Metric of record: TIME PER QUERY — start-to-start between consecutive queries,
i.e. 1/throughput. It covers the WHOLE cycle (data loading and preprocessing
included), and it is anchor-invariant in steady state: measured from pipeline
starts or from training starts it gives the same number (66118 vs 66242 us on a
b8 cell). That invariance is what makes it comparable across the three, whose
marker sets differ — the Choreo stages run with `disable_logs: true` so that no
synchronous write+flush sits inside the measured interval on one side only.

Co-headline: the QUERY LATENCY BREAKDOWN, taken from the spans of the
choreo-traced runs — per-stage latency (dataloader, training) plus the auxiliary
framework overheads (entry, handoff, exit) INSIDE the pipeline; the trailing
turnaround to the next query is loadgen admission, reported separately. It sums to the time per
query by construction; E1 verified the decomposition exact (residual 0.000 us
over 300 queries on a 2-stage pipeline). Negative intervals are a hard failure,
never something to take a median over.

All latencies come from the monotonic perf_counter_ns column (the trailing CSV
field), never wall-clock column 0.

Statistic of record: the PAIRED across-run difference. The three configurations
are interleaved run-by-run, their order rotated by repetition index, so runs pair
by id: d_i = median(choreo_i) - median(monolith_i). The CI resamples PAIRS with
replacement and re-resamples queries within each chosen run.

    python analyze_e2.py [--machines m2pro gb10] [--warmup 50] [--drop-runs 1]
                         [--fig-dir DIR] [--no-breakdown] [--latex MACHINE]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NS_PER_MS = 1e6
NS_PER_US = 1e3

# Steps dropped at the head of each run. Was 200, which was never needed:
# measured on the previous collection, step time is flat from step 0 on the GB10
# (38.8, 38.7, 38.7 ... ms across 900 steps) and flat on a warm M2 Pro run. What
# does exist is a whole-run effect, handled by DROP_RUNS below.
WARMUP = 50

# Markers. The monolith brackets its own step; Choreo runs with disable_logs
# on both stages, so it writes NO per-stage rows -- its timing comes from the
# pipeline-level row, which pipeline.py emits unconditionally. That asymmetry
# is deliberate: it makes the two sides write the same NUMBER of rows through
# the same synchronous handler, so the instrument is equal on both.
MONOLITH_STEP = "training_step"            # monolith per-step marker
MONOLITH_LOOP = "training_loop"            # session marker (see parse_monolith_steps)
PIPELINE_ROW = "pipeline - "               # Choreo per-query row, prefix match
PIPELINE_PREPARE = "pipeline"              # session marker (see _pipeline_events)
DATALOADER_STAGE = "Load Imagenette samples from TorchVision Dataset"
TRAIN_STAGE = "EfficientNet training"

# The three things run per cell, named for what they are.
MONOLITH, SUITE_TRACED = "monolith", "choreo-traced"
CONFIGS = (MONOLITH, SUITE_TRACED)
# The untraced `choreo` arm is gone. With the pipeline's per-query CSV rows
# gated off it has no instrument at all -- no spans (tracing off) and no rows --
# so it cannot be measured, and keeping the rows on for it alone would price
# tracing by comparing two different instruments. E1 prices tracing directly on
# a clean microbenchmark instead. See collect_e2.sh.
CHOREO = "choreo"                      # historical token, for reading old data
CONFIG_DESC = {
    MONOLITH:      "bare PyTorch loop, no framework",
    CHOREO:        "the framework, tracing off",
    SUITE_TRACED: "the framework, tracing on",
}

# The model sweep is a SIZE ladder within one architecture family (S -> M -> L),
# so step time is the only thing changing.
MODEL_DISPLAY = {"effv2s": "EfficientNetV2-S", "effv2m": "EfficientNetV2-M",
                 "effv2l": "EfficientNetV2-L"}
MODEL_ORDER = ["effv2s", "effv2m", "effv2l"]   # by step-time, re-sorted at runtime
BATCHES = [1, 2, 4, 8, 16, 32, 64]
ANCHOR_MODEL, ANCHOR_BATCH = "effv2s", 8

_FNAME_RE = re.compile(
    r"^mod_m(?P<model>[a-z0-9]+)_b(?P<batch>\d+)"
    r"_(?P<config>monolith|choreo-traced|choreo)"
    r"_(?P<machine>[a-z0-9]+)_r(?P<run>\d+)\.csv$"
)


def parse_filename(path):
    """Return dict(model, batch, config, machine, run, path) or None.

    No aliasing of the previous naming: that data was deleted rather than
    archived, so there is nothing to stay compatible with, and a regex that
    silently accepts two schemes is how two collection eras get mixed.
    """
    m = _FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return {"model": m["model"], "batch": int(m["batch"]),
            "config": m["config"], "machine": m["machine"],
            "run": int(m["run"]), "path": path}


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def _rows(path):
    """Yield (module, phase, event, perf_ns) for well-formed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                perf = int(parts[-1])
            except ValueError:
                continue
            yield parts[2], parts[3], parts[4], perf


def _pair_spans(evs):
    """Pair alternating (perf, start/end) events into (start_ns, end_ns) spans."""
    evs.sort()
    spans, i = [], 0
    while i < len(evs) - 1:
        if evs[i][1] == "start" and evs[i + 1][1] == "end":
            spans.append((evs[i][0], evs[i + 1][0]))
            i += 2
        else:
            i += 1                      # skip a stray/unpaired event
    return spans


def _pair(evs):
    """Pair alternating (perf, start/end) events into durations."""
    return [e - s for s, e in _pair_spans(evs)]


def parse_monolith_steps(path):
    """Per-step durations for the LAST session in the file.

    baseline_finetune historically opened its log in append mode, so re-runs
    could accumulate stale sessions (the append bug — it can flip the measured
    overhead negative). Each session emits one `training_loop, run, start`
    marker; reset at every marker so only the final session counts. Robust even
    against pre-fix contaminated files."""
    evs = []
    for (mod, phase, event, perf) in _rows(path):
        if mod == MONOLITH_LOOP and phase == "run" and event == "start":
            evs = []
        elif mod == MONOLITH_STEP and phase == "run" and event in ("start", "end"):
            evs.append((perf, event))
    return _pair(evs)


def _pipeline_events(path):
    """Choreo's per-query row events, for the LAST session in the file.

    The rows are emitted unconditionally by pipeline.py, so they survive
    `disable_logs: true` on the stages.

    Reset at every `pipeline, prepare, start`, exactly as parse_monolith_steps
    does for the monolith. main.py appends to an existing label's CSV, so a run
    that is killed part-way leaves a partial session and the next run with the
    same label concatenates onto it. That happened for real —
    mod_meffv2l_b8_choreo-traced_m2pro_r7 came out at 655 rows instead of 603
    and carried a 57-MINUTE interval between the two sessions. Its median
    survived only because the junk landed inside the dropped warm-up, which is
    luck rather than a guarantee: the same file at a different warm-up, or a
    kill later in the run, would corrupt the cell silently."""
    evs = []
    for (mod, phase, event, perf) in _rows(path):
        if mod == PIPELINE_PREPARE and phase == "prepare" and event == "start":
            evs = []
        elif (mod.startswith(PIPELINE_ROW) and phase == "run"
              and event in ("start", "end")):
            evs.append((perf, event))
    return evs


def _monolith_events(path):
    """The LAST session's step events (see parse_monolith_steps for why)."""
    evs = []
    for (mod, phase, event, perf) in _rows(path):
        if mod == MONOLITH_LOOP and phase == "run" and event == "start":
            evs = []
        elif mod == MONOLITH_STEP and phase == "run" and event in ("start", "end"):
            evs.append((perf, event))
    return evs


def _events(path, config):
    return _monolith_events(path) if config == MONOLITH else _pipeline_events(path)


def parse_time_per_query(path, config):
    """TIME PER QUERY: start-to-start between consecutive queries. The metric.

    One query is one batch here, so this is equally time per batch, and it is
    1/throughput. It covers the WHOLE cycle -- data loading and preprocessing
    included -- unlike the step's own duration, which excludes everything
    between steps and so is blind to where most of the framework's cost lands.

    It is also anchor-invariant: in steady state the same period is measured
    whether you cut at the monolith's step row or Choreo's pipeline row. That
    is what makes the two comparable while they emit different markers.

    Returned as a per-query series (n-1 values) so it drops straight into
    paired_overhead, which resamples within a run and across run pairs.
    """
    spans = _pair_spans(_events(path, config))
    starts = np.asarray([s for s, _ in spans], dtype=np.float64)
    return np.diff(starts) if starts.size > 1 else np.asarray([])


def parse_step_spans(path, config):
    """(start_ns, end_ns) per query -- the bracketed interval, for coverage."""
    return _pair_spans(_events(path, config))


# Repetitions discarded as SYSTEM warm-up (whole runs, not steps). The first
# repetition of a cell is measurably slower for its ENTIRE duration — e.g. the
# 2026-08-18 mps anchor: baseline r1 = 97.2 ms/step vs 89.2 for r2..r6, still
# 97.4 in its last 300 steps, so per-step warm-up dropping cannot remove it
# (cold page cache / power state on the collection's first process launch).
# Default 1, not opt-in: on the E2 smoke collection leaving run 1 in moved one
# breakdown term by 800 us. Collection therefore runs R+1 repetitions so that
# dropping the first still leaves R usable. The per-run medians printed under
# each table keep any further outlier visible rather than silently dropped.
DROP_RUNS = 1


def steps_by_run(metas, warmup=WARMUP, drop_runs=None):
    """{run_id: [step_ns, ...]} with the first `warmup` steps dropped per run,
    and the lowest `drop_runs` run ids discarded entirely as system warm-up."""
    drop = DROP_RUNS if drop_runs is None else drop_runs
    keep = sorted({m["run"] for m in metas})[drop:] if drop else None
    out = {}
    for m in metas:
        if keep is not None and m["run"] not in keep:
            continue
        fn = parse_monolith_steps
        d = fn(m["path"])[warmup:]
        if d:
            out[m["run"]] = d
    return out


def periods_by_run(metas, warmup=WARMUP, drop_runs=None, keep_runs=None):
    """{run_id: [period_ns, ...]} — same warm-up / run-dropping policy as
    steps_by_run, plus an explicit `keep_runs` so the regime filter's verdict
    (computed on step durations) is applied identically here."""
    drop = DROP_RUNS if drop_runs is None else drop_runs
    keep = sorted({m["run"] for m in metas})[drop:] if drop else None
    out = {}
    for m in metas:
        if keep is not None and m["run"] not in keep:
            continue
        if keep_runs is not None and m["run"] not in keep_runs:
            continue
        d = parse_time_per_query(m["path"], m["config"])[warmup:]
        if len(d):
            out[m["run"]] = d
    return out


def coverage_by_run(metas, warmup=WARMUP, drop_runs=None, keep_runs=None):
    """{run_id: (median_in_step_ns, median_period_ns)} — how much of the wall
    clock the step marker actually covers."""
    drop = DROP_RUNS if drop_runs is None else drop_runs
    keep = sorted({m["run"] for m in metas})[drop:] if drop else None
    out = {}
    for m in metas:
        if keep is not None and m["run"] not in keep:
            continue
        if keep_runs is not None and m["run"] not in keep_runs:
            continue
        spans = parse_step_spans(m["path"], m["config"])[warmup:]
        if len(spans) < 2:
            continue
        durs = np.asarray([e - s for s, e in spans], dtype=np.float64)
        per = np.diff(np.asarray([s for s, _ in spans], dtype=np.float64))
        out[m["run"]] = (float(np.median(durs)), float(np.median(per)))
    return out


def load(results_dir, machine):
    metas = []
    for p in sorted(glob.glob(os.path.join(results_dir, "mod_*.csv"))):
        m = parse_filename(p)
        if m and m["machine"] == machine:
            metas.append(m)
    return metas


def select(metas, config=None, model=None, batch=None):
    out = []
    for m in metas:
        if config is not None and m["config"] != config:
            continue
        if model is not None and m["model"] != model:
            continue
        if batch is not None and m["batch"] != batch:
            continue
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
_BOOT_WORK_BUDGET = 5e7


def _arrs(by_run):
    return [np.asarray(by_run[k], dtype=np.float64) for k in sorted(by_run) if len(by_run[k])]


def summarize(by_run, unit_ns=NS_PER_MS, n_boot=10000, seed=0):
    """median/mean/p95 + hierarchical bootstrap CI (runs resampled, then steps)."""
    arrs = _arrs(by_run)
    if not arrs:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "run_medians": []}
    a = np.concatenate(arrs)
    pooled_n = a.size
    n_eff = int(min(n_boot, max(1000, _BOOT_WORK_BUDGET // max(pooled_n, 1))))
    rng = np.random.default_rng(seed)
    R = len(arrs)
    boots = np.empty(n_eff)
    for i in range(n_eff):
        parts = [arrs[j][rng.integers(0, arrs[j].size, arrs[j].size)]
                 for j in rng.integers(0, R, R)]
        boots[i] = np.median(np.concatenate(parts))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": int(pooled_n), "median": float(np.median(a)) / unit_ns,
            "mean": float(a.mean()) / unit_ns,
            "ci_lo": float(lo) / unit_ns, "ci_hi": float(hi) / unit_ns,
            "runs": R,
            "run_medians": [float(np.median(v)) / unit_ns for v in arrs]}


# Point estimator for combining the per-run paired differences.
#   "median" (default) — robust: a single contaminated repetition cannot move it.
#   "mean"             — the classic average; kept for comparison and reported
#                        alongside, since swapping estimators after seeing the
#                        data must be visible, not silent.
# Why median is the default: real collections produce occasional bad runs
# (machine interference, a stray process). At the 2026-08-18 mps anchor, 8 of 9
# repetitions agreed within +/-55 us but one read -1241.8 us, which dragged the
# MEAN to -145 us (a nonsensical "the wrapper makes it faster") while the MEDIAN
# read -6 us. The effect under test is ~50 us, far smaller than one bad run.
ESTIMATOR = "median"


def _combine(vals, estimator=None):
    est = ESTIMATOR if estimator is None else estimator
    return float(np.median(vals)) if est == "median" else float(np.mean(vals))


def paired_overhead(base_by_run, arm_by_run, n_boot=10000, seed=0, estimator=None):
    """Paired across-run overhead — the statistic of record.

    d_i = median(b_i) - median(a_i) over runs both configurations share,
    combined across runs by ESTIMATOR (median by default; see above). The CI
    resamples PAIRS with replacement and re-resamples queries within each chosen
    run, and applies the same estimator to each bootstrap replicate. Also
    reports the difference as a % of the pooled median of the reference (a)."""
    shared = sorted(set(base_by_run) & set(arm_by_run))
    if len(shared) < 2:
        return None
    b = {r: np.asarray(base_by_run[r], dtype=np.float64) for r in shared}
    c = {r: np.asarray(arm_by_run[r], dtype=np.float64) for r in shared}
    med_base = float(np.median(np.concatenate([b[r] for r in shared])))
    d_runs = {r: float(np.median(c[r])) - float(np.median(b[r])) for r in shared}
    d_point = _combine(list(d_runs.values()), estimator)

    rng = np.random.default_rng(seed)
    R = len(shared)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ds = []
        for j in rng.integers(0, R, R):
            r = shared[j]
            rb = b[r][rng.integers(0, b[r].size, b[r].size)]
            rc = c[r][rng.integers(0, c[r].size, c[r].size)]
            ds.append(np.median(rc) - np.median(rb))
        boots[i] = _combine(ds, estimator)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dv = list(d_runs.values())
    return {"pairs": R, "step_base_ms": med_base / NS_PER_MS,
            "estimator": ESTIMATOR if estimator is None else estimator,
            "abs_median_us": float(np.median(dv)) / NS_PER_US,
            "abs_mean_us": float(np.mean(dv)) / NS_PER_US,
            "abs_us": d_point / NS_PER_US,
            "abs_lo_us": float(lo) / NS_PER_US, "abs_hi_us": float(hi) / NS_PER_US,
            "pct": 100.0 * d_point / med_base,
            "pct_lo": 100.0 * float(lo) / med_base, "pct_hi": 100.0 * float(hi) / med_base,
            "within_noise": bool(lo <= 0.0 <= hi),
            "per_pair_us": [d_runs[r] / NS_PER_US for r in shared]}


# ---------------------------------------------------------------------------
# Per-cell assembly
# ---------------------------------------------------------------------------
# Some cells are BISTABLE: a given process launch lands in one of two distinct
# step-time regimes (mps EfficientNetV2-S b4: ~46 ms or ~105 ms, a 2.3x gap, same
# config). If the arms of one repetition land in DIFFERENT regimes, that pair
# measures the regime gap (~60 ms), not the wrapper (~50 us) — it is not a valid
# pair at all. MAX_REGIME_RATIO drops such repetitions.
#
# This filter is deliberately OUTCOME-INDEPENDENT: it looks only at the arms'
# absolute step times, never at the overhead they imply, so it cannot bias the
# result toward the answer we expect. Discarding pairs because their *difference*
# looked wrong would be exactly that bias; this is not that.
MAX_REGIME_RATIO = 0.0          # 0 = disabled; 1.25 = arms must agree within 25%


def _regime_filter(arms, ratio):
    """Drop repetitions whose arms sit in different step-time regimes.
    `arms` = list of {run: steps}; returns (filtered arms, dropped run ids)."""
    if not ratio:
        return arms, []
    shared = set(arms[0])
    for a in arms[1:]:
        shared &= set(a)
    dropped = []
    for r in sorted(shared):
        meds = [float(np.median(a[r])) for a in arms]
        if min(meds) > 0 and max(meds) / min(meds) > ratio:
            dropped.append(r)
    if dropped:
        arms = [{k: v for k, v in a.items() if k not in dropped} for a in arms]
    return arms, dropped


def cell_result(metas, model, batch, warmup):
    """Everything for one (model, batch) cell, or None if incomplete.

    TIME PER QUERY is the metric of record and is available for all three
    configurations. The step's own duration exists only for the monolith --
    Choreo runs with `disable_logs: true` and writes no per-stage rows -- so
    there is deliberately no cross-configuration in-step comparison here. That
    comparison was E2's old headline and it was unusable: the framework's cost
    lands mostly BETWEEN steps, which the step marker excludes by construction,
    so it measured a near-zero difference against run-to-run noise an order of
    magnitude larger, and flipped sign between repetitions.
    """
    sel = lambda c: select(metas, config=c, model=model, batch=batch)
    kw = dict(warmup=warmup)
    # The monolith is measured from its own per-step log -- it is a bare PyTorch
    # loop run with --no-radt and emits no spans. That asymmetry is the point of
    # the control: the reference must not carry the framework's instrument.
    q_mono = periods_by_run(sel(MONOLITH), **kw)
    if not q_mono:
        return None

    dropped = []
    if MAX_REGIME_RATIO and len(q_mono) > 2:
        (q_mono,), dropped = _regime_filter([q_mono], MAX_REGIME_RATIO)
        if not q_mono:
            return None

    # Choreo's side is filled in later, from SPANS, by print_breakdown ->
    # c["breakdown"]["in_pipeline"]. There is deliberately no CSV-derived
    # Choreo timing any more: the pipeline's per-query rows are gated off.
    out = {"model": model, "batch": batch, "regime_dropped": dropped,
           "q_monolith": summarize(q_mono),
           # kept raw so the cost can be taken as a PAIRED difference against
           # Choreo's per-run L_q once the spans have been read
           "q_mono_by_run": q_mono,
           "breakdown": None, "cost": None}

    # The monolith's own split, needed for the identity below: the step it
    # brackets, and the gap between steps where its data loading happens.
    steps = steps_by_run(sel(MONOLITH), warmup)
    out["monolith_step"] = summarize(steps) if steps else None
    out["coverage"] = float("nan")
    cov = coverage_by_run(sel(MONOLITH), **kw)
    if cov:
        out["coverage"] = float(np.median([d / p for d, p in cov.values() if p]))
    return out


def collect_cells(metas, warmup):
    seen = sorted({(m["model"], m["batch"]) for m in metas
                   if m["model"] in MODEL_DISPLAY})
    cells = []
    for model, batch in seen:
        c = cell_result(metas, model, batch, warmup)
        if c:
            cells.append(c)
    return cells


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def print_cells(cells, machine):
    """The monolith reference, per cell.

    Choreo's own numbers are not here: they come from spans and are printed by
    print_breakdown / print_span_cost below. This table exists so the reference
    the comparison rests on is visible, with its run-to-run spread.
    """
    print(f"\n## {machine} — monolith reference (time per query)\n")
    print("| cell | R | monolith (ms) | per-run medians (ms) |")
    print("|---|--:|--:|---|")
    for c in cells:
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        mono = c["q_monolith"]
        runs = " / ".join(f"{v:.2f}" for v in mono.get("run_medians", []))
        print(f"| {name} | {mono.get('runs', 0)} | {mono['median']:.2f} | {runs} |")
    print("\n(the monolith is a bare PyTorch loop run with --no-radt: no spans, no "
          "framework. Its per-step timing comes from its own log, which is the only "
          "instrument it has. Time per query = start-to-start between consecutive "
          "steps = 1/throughput.)")
    for c in cells:
        if c.get("regime_dropped"):
            print(f"- MIXED-REGIME repetitions dropped, "
                  f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}: "
                  f"{c['regime_dropped']}")


# ---------------------------------------------------------------------------
# Query latency breakdown, from the spans of the traced configuration
# ---------------------------------------------------------------------------
def _span_runs(machine, store=None):
    """{label: run_id} from the local MLflow store this machine recorded to."""
    import mlflow
    db = store or os.path.join(HERE, f"mlruns_e2_{machine}.db")
    if not os.path.exists(db):
        return {}, None
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    c = mlflow.MlflowClient()
    out = {}
    for r in c.search_runs(["0"], max_results=5000):
        name = r.data.tags.get("mlflow.runName", "")
        out[name.split(" | ")[0]] = r.info.run_id
    return out, c


def breakdown_by_run(machine, model, batch, runs, warmup, store=None):
    """Per-run query latency breakdown from the traced configuration's spans.

    Six components per query, in order: entry, dataloader, handoff, training,
    exit, turnaround. They are successive instants within ONE query on ONE
    clock, so unlike any difference taken across two processes they are
    non-negative by construction and carry no run-level term at all.
    """
    # Running this file as a script puts ITS directory on sys.path, not the repo
    # root, so utils/ is not importable without help.
    root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils.span_reader import read_dir
    labels, client = _span_runs(machine, store)
    if not labels:
        return {}
    out = {}
    for r in runs:
        lab = f"mod_m{model}_b{batch}_{SUITE_TRACED}_{machine}_r{r}"
        rid = labels.get(lab)
        if rid is None:
            continue
        t = read_dir(client.download_artifacts(run_id=rid, path="radt-trace"))
        pq  = t.by_query("pipeline query");  pqp = t.by_query("pipeline query processed")
        dlr = t.by_query(f"{DATALOADER_STAGE}.run")
        dlp = t.by_query(f"{DATALOADER_STAGE}.push_to_outputs")
        trr = t.by_query(f"{TRAIN_STAGE}.run")
        trp = t.by_query(f"{TRAIN_STAGE}.push_to_outputs")
        qs = sorted([q for q in pq if q in pqp and q in dlr and q in dlp
                     and q in trr and q in trp],
                    key=lambda q: pq[q].perf_start_ns)[warmup:]
        if len(qs) < 2:
            continue
        P = lambda d, q: d[q].perf_start_ns
        comp = {
            "entry":      [P(dlr, q) - P(pq, q) for q in qs],
            "dataloader": [P(dlp, q) - P(dlr, q) for q in qs],
            "handoff":    [P(trr, q) - P(dlp, q) for q in qs],
            "training":   [P(trp, q) - P(trr, q) for q in qs],
            "exit":       [P(pqp, q) - P(trp, q) for q in qs],
            "turnaround": [P(pq, qs[i + 1]) - P(pqp, qs[i]) for i in range(len(qs) - 1)],
        }
        # These intervals cannot legitimately be negative: they are successive
        # instants in a strictly sequential pipeline. A negative one means the
        # spans were mis-paired, which would silently shift every later query,
        # so refuse the run rather than take a median over it.
        bad = {k: sum(1 for v in vs if v < 0) for k, vs in comp.items()}
        if any(bad.values()):
            print(f"  !! run {r}: NEGATIVE intervals {bad} — run excluded; "
                  f"spans are mis-paired and a median over them would be wrong")
            continue
        out[r] = comp
    return out


def attach_breakdown(machine, cells, metas, warmup, store=None):
    """Read the spans once and hang everything derived from them on the cells.

    Two things come out of it. The BREAKDOWN -- six components per query, all
    from one clock inside one process. And the COST of decomposition, taken as a
    paired across-run difference between the monolith's time per query and
    Choreo's in-pipeline L_q, so it is the same statistic of record E1 uses and
    carries a bootstrap CI rather than a bare median difference.
    """
    for c in cells:
        runs = sorted({m["run"] for m in select(metas, config=SUITE_TRACED,
                                                model=c["model"], batch=c["batch"])})
        if DROP_RUNS:
            runs = runs[DROP_RUNS:]
        bd = breakdown_by_run(machine, c["model"], c["batch"], runs, warmup, store)
        if not bd:
            continue
        c["breakdown_runs"] = bd
        med = {k: np.median([np.median(bd[r][k]) for r in bd]) / NS_PER_US
               for k in ("entry", "dataloader", "handoff", "training", "exit",
                         "turnaround")}
        # `turnaround` is scheduler admission -- the gap between one query being
        # counted processed and the next being admitted. It is NOT the pipeline
        # executing anything, and the monolith has no analogue for it (its loop
        # runs continuously, with no loadgen). Charging it to the framework
        # measures the harness, not the framework's inflation of the pipeline,
        # so it is excluded from both the framework total and L_q and reported
        # separately.
        med["framework"] = med["entry"] + med["handoff"] + med["exit"]
        med["in_pipeline"] = med["framework"] + med["dataloader"] + med["training"]
        c["breakdown"] = med
        # Per-query in-pipeline L_q, in ns, per run: the same shape the monolith
        # side already has, so the existing paired statistic applies unchanged.
        lq_by_run = {r: [sum(v) for v in zip(bd[r]["entry"], bd[r]["dataloader"],
                                             bd[r]["handoff"], bd[r]["training"],
                                             bd[r]["exit"])]
                     for r in bd}
        c["cost"] = paired_overhead(c["q_mono_by_run"], lq_by_run)
        c["framework_ci"] = framework_ci(bd)


def framework_ci(bd, n_boot=10000, seed=0):
    """CI on the in-process framework term (entry + handoff + exit) per query.

    This is the quantity E2 actually resolves. It is measured WITHIN one
    process on one clock, so it carries none of the run-to-run drift that
    dominates the monolith-vs-Choreo difference, and the interval is three
    orders of magnitude tighter. Bootstrap is hierarchical to match the rest of
    the file: repetitions resampled, then queries within each chosen repetition.
    """
    runs = sorted(bd)
    fw = {r: np.asarray([e + h + x for e, h, x in
                         zip(bd[r]["entry"], bd[r]["handoff"], bd[r]["exit"])],
                        dtype=np.float64) for r in runs}
    lq = {r: np.asarray([e + d + h + t + x for e, d, h, t, x in
                         zip(bd[r]["entry"], bd[r]["dataloader"], bd[r]["handoff"],
                             bd[r]["training"], bd[r]["exit"])], dtype=np.float64)
          for r in runs}
    rng = np.random.default_rng(seed)
    R = len(runs)
    # Same work budget as summarize(): a replicate here costs 2R medians over
    # ~n queries, so cap the replicate count rather than let a heavy cell run
    # for minutes to sharpen an interval that is already a few percent wide.
    pooled = sum(fw[r].size for r in runs)
    n_eff = int(min(n_boot, max(1000, _BOOT_WORK_BUDGET // max(2 * pooled, 1))))
    abs_b = np.empty(n_eff); pct_b = np.empty(n_eff)
    for i in range(n_eff):
        fs, ps = [], []
        for j in rng.integers(0, R, R):
            r = runs[j]
            idx = rng.integers(0, fw[r].size, fw[r].size)
            mf = np.median(fw[r][idx])
            fs.append(mf); ps.append(100.0 * mf / np.median(lq[r][idx]))
        abs_b[i] = np.median(fs); pct_b[i] = np.median(ps)
    a_lo, a_hi = np.percentile(abs_b, [2.5, 97.5])
    p_lo, p_hi = np.percentile(pct_b, [2.5, 97.5])
    point = float(np.median([np.median(fw[r]) for r in runs]))
    pct = float(np.median([100.0 * np.median(fw[r]) / np.median(lq[r]) for r in runs]))
    return {"runs": R, "n_boot": n_eff,
            "abs_us": point / NS_PER_US,
            "abs_lo_us": float(a_lo) / NS_PER_US, "abs_hi_us": float(a_hi) / NS_PER_US,
            "pct": pct, "pct_lo": float(p_lo), "pct_hi": float(p_hi)}


def print_breakdown(machine, cells):
    """Where a query's latency goes, split into stage work and framework cost."""
    print(f"\n## {machine} — query latency breakdown (spans)\n")
    print("| cell | R | entry | dataloader | handoff | training | exit "
          "| **framework** | **in-pipeline L_q** | framework % | (scheduling) |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    any_row = False
    for c in cells:
        med = c.get("breakdown")
        if not med:
            continue
        any_row = True
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        print(f"| {name} | {len(c['breakdown_runs'])} | {med['entry']:.1f} "
              f"| {med['dataloader']:.1f} | {med['handoff']:.1f} "
              f"| {med['training']:.1f} | {med['exit']:.1f} "
              f"| {med['framework']:.1f} | {med['in_pipeline']:.1f} "
              f"| {100 * med['framework'] / med['in_pipeline']:.2f}% "
              f"| {med['turnaround']:.1f} |")
    if not any_row:
        print("| (no traced spans found for this machine) |")
        return
    print("\n(microseconds per query, all from SPANS. `dataloader` and `training` "
          "are the stages doing real work; `entry`, `handoff` and `exit` are the "
          "framework moving the query between them, and are what decomposition "
          "adds INSIDE the pipeline. Their sum plus the stages is `in-pipeline "
          "L_q` = `pipeline query` start to `pipeline query processed` start.\n"
          "\n`scheduling` is the trailing turnaround to the next query -- loadgen "
          "admission, outside the pipeline's execution of this query, and with no "
          "monolith analogue since a bare loop has no admission step. It is "
          "listed for completeness and excluded from the framework total: the "
          "question is whether the framework inflates the pipeline, not what the "
          "harness costs around it. L_q + scheduling = the start-to-start time "
          "per query.)")


def print_span_cost(machine, cells):
    """Cost of decomposition — the headline of E2.

    Choreo's side comes from SPANS: in-pipeline L_q, `pipeline query` start to
    `pipeline query processed` start. It excludes loadgen admission, which is
    not the pipeline executing anything and has no monolith analogue.

    The monolith's side is its own per-step timing, because it is a bare PyTorch
    loop run with --no-radt and emits no spans at all. That asymmetry is
    unavoidable and is the point of the control: the reference must not carry
    the framework's instrument.

    The difference is PAIRED by repetition — monolith run r against Choreo run r,
    which ran minutes apart on the same machine in the same session — and the CI
    resamples those pairs. Choreo's L_q is from the traced runs, so the number
    reported is the cost of decomposition WITH tracing on, which is the only
    configuration E2 collects and the one the paper claims.
    """
    rows = [c for c in cells if c.get("cost") and c.get("breakdown")]
    if not rows:
        return
    print(f"\n## {machine} — cost of decomposition, in-pipeline only\n")
    print("| cell | R | monolith (ms) | choreo L_q (ms) | cost (µs) | 95% CI (µs) "
          "| as % | framework term (µs) | 95% CI (µs) | as % of L_q |")
    print("|---|--:|--:|--:|--:|---|--:|--:|---|--:|")
    for c in rows:
        o, b_ = c["cost"], c["breakdown"]
        mono = c["q_monolith"]["median"]
        lq = b_["in_pipeline"] / 1000.0
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        flag = " (n.s.)" if o["within_noise"] else ""
        f = c.get("framework_ci") or {}
        fci = (f"[{f['abs_lo_us']:.1f}, {f['abs_hi_us']:.1f}]" if f else "—")
        fpct = (f"{f['pct']:.3f}%" if f else "—")
        print(f"| {name} | {o['pairs']} | {mono:.2f} | {lq:.2f} | "
              f"{o['abs_us']:+.1f} | [{o['abs_lo_us']:+.1f}, {o['abs_hi_us']:+.1f}]"
              f"{flag} | {o['pct']:+.3f}% | {b_['framework']:.1f} | {fci} | {fpct} |")
    print("\n(Choreo from spans, monolith from its own per-step log -- it runs "
          "with --no-radt by design and has no spans. Loadgen admission is "
          "excluded from the Choreo side; the monolith has no admission step, so "
          "this is the like-for-like pairing. Cost is the "
          f"{ESTIMATOR} of per-repetition paired differences, CI bootstrapped "
          "over those pairs; `n.s.` marks a cell whose interval contains zero, "
          "i.e. the cost is smaller than this apparatus resolves there. A "
          "NEGATIVE cost is not a speed-up and is printed as measured rather "
          "than clipped. `framework term` is the span-measured "
          "entry+handoff+exit, i.e. what the framework adds from the INSIDE, "
          "measured within one process and so free of the cross-process noise "
          "the paired difference carries.)")
    for c in rows:
        pp = " / ".join(f"{v:+.1f}" for v in c["cost"]["per_pair_us"])
        print(f"- per-repetition paired differences (µs), "
              f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}: {pp}")


def print_identity(machine, cells):
    """Reconcile the measured cost of choreo against the breakdown.

    Decomposing does not only ADD scaffolding; it also relocates the stage
    bodies, and those two effects can be of similar size and opposite sign. If
    only the net were reported, a cell where they cancel would read as
    "modularity is free" when in fact both terms are large.

        cost of choreo = (dataloader − monolith gap)
                       + (training   − monolith step)
                       + framework scaffolding
    """
    rows = [c for c in cells if c.get("breakdown") and c.get("monolith_step")
            and c.get("cost")]
    if not rows:
        return
    print(f"\n## {machine} — where the cost of decomposition comes from\n")
    print("| cell | dl − gap | training − step | framework | sum | measured | "
          "residual |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for c in rows:
        b = c["breakdown"]
        step = c["monolith_step"]["median"] * (NS_PER_MS / NS_PER_US)
        gap = c["q_monolith"]["median"] * (NS_PER_MS / NS_PER_US) - step
        d_dl = b["dataloader"] - gap
        d_tr = b["training"] - step
        fw = b["framework"]          # entry + handoff + exit; scheduling excluded
        measured = c["cost"]["abs_us"]
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        print(f"| {name} | {d_dl:+.1f} | {d_tr:+.1f} | {fw:+.1f} | "
              f"{d_dl + d_tr + fw:+.1f} | {measured:+.1f} | "
              f"{measured - (d_dl + d_tr + fw):+.1f} |")
    print("\n(microseconds. The monolith's gap is its time per query minus its "
          "step, i.e. where its own data loading happens. A large residual means "
          "the medians are being combined across quantities that do not add "
          "exactly; a large `dl − gap` means decomposition MOVED work rather "
          "than adding it, which the net alone would hide.)")


def print_sweeps(cells, machine):
    """The amortization claim: a fixed per-query cost against a growing query.

    Led by the framework term, because that is the quantity with a usable
    interval. The cross-process cost is carried alongside so the reader can see
    that it is consistent with the framework term and simply too noisy to
    resolve a trend -- it is not quietly dropped because it is inconvenient.
    """
    have = [c for c in cells if c.get("framework_ci")]
    bs = sorted([c for c in have if c["model"] == ANCHOR_MODEL],
                key=lambda c: c["batch"])
    ms = sorted([c for c in have if c["batch"] == ANCHOR_BATCH],
                key=lambda c: c["q_monolith"]["median"])
    for title, sel_, label in (("Batch sweep (EfficientNetV2-S)", bs, "batch"),
                               ("Model sweep (batch 8)", ms, "model")):
        if len(sel_) < 2:
            continue
        print(f"\n### {machine} — {title}\n")
        print(f"| {label} | query latency L_q (ms) | framework (µs) | as % of L_q "
              f"| (cross-process cost, µs) |")
        print("|---|--:|--:|--:|--:|")
        for c in sel_:
            key = c["batch"] if label == "batch" else MODEL_DISPLAY.get(c["model"], c["model"])
            f = c["framework_ci"]
            xp = f"{c['cost']['abs_us']:+.0f}" if c.get("cost") else "—"
            print(f"| {key} | {c['breakdown']['in_pipeline'] / 1000.0:.2f} | "
                  f"{f['abs_us']:.1f} | {f['pct']:.3f}% | {xp} |")
        qs = [c["breakdown"]["in_pipeline"] / 1000.0 for c in sel_]
        fw = [c["framework_ci"] for c in sel_]
        print(f"\nquery latency {qs[0]:.1f} → {qs[-1]:.1f} ms ({qs[-1] / qs[0]:.1f}×): "
              f"framework {fw[0]['abs_us']:.0f} → {fw[-1]['abs_us']:.0f} µs "
              f"({fw[-1]['abs_us'] / fw[0]['abs_us']:.2f}×), "
              f"i.e. {fw[0]['pct']:.3f}% → {fw[-1]['pct']:.3f}% of the query.")


def print_latex(cells, machine):
    print("% --- E2 modularity overhead: time per query ---")
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Cost of decomposition, as time per query.}")
    print("\\label{tab:modularity-sweep}")
    print("\\begin{tabular}{lrrr}\n\\toprule")
    print("Cell & Time per query (\\si{\\milli\\second}) & "
          "Cost (\\si{\\micro\\second}) & \\% \\\\\n\\midrule")
    for c in cells:
        o = c.get("cost")
        if not o:
            continue
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        print(f"{name} & {c['q_monolith']['median']:.2f} & "
              f"{o['abs_us']:+.1f} & {o['pct']:+.3f} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


def make_breakdown_figure(per_machine, fig_dir):
    """What the framework adds to a query, and which part of it adds what.

    One panel per machine: the three auxiliary components stacked, per cell,
    ordered by how heavy the query is. The dataloader and training stages are
    the real work and are left out entirely -- at b64 on the Mac they are 1.08 s
    against 0.3 ms of scaffolding, so nothing else would be visible.

    The amortization curve is NOT repeated here; it is the right panel of
    e2_modularity_scale.png. This figure answers the different question of
    WHERE the fixed cost sits, which is the one that differs between machines:
    the hand-off between stage threads is ~3x more expensive on the GB10 than
    on the Mac and dominates its total.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    machines = [m for m in per_machine
                if any(c.get("breakdown") for c in per_machine[m])]
    if not machines:
        return None
    # Scheduling (turnaround) is deliberately not stacked here: it is loadgen
    # admission outside the pipeline, not framework work on this query.
    parts = ("entry", "handoff", "exit")
    colors = {"entry": "tab:blue", "handoff": "tab:orange", "exit": "tab:green"}

    fig, ax = plt.subplots(1, len(machines), figsize=(5.4 * len(machines), 4.2),
                           squeeze=False)
    ax = ax[0]
    for i, machine in enumerate(machines):
        cells = sorted([c for c in per_machine[machine] if c.get("breakdown")],
                       key=lambda c: c["breakdown"]["in_pipeline"])
        labels = [MODEL_DISPLAY.get(c["model"], c["model"]).replace("EfficientNetV2-", "")
                  + f"\nb{c['batch']}" for c in cells]
        bottom = [0.0] * len(cells)
        for part in parts:
            vals = [c["breakdown"][part] for c in cells]
            ax[i].bar(labels, vals, bottom=bottom, label=part,
                      color=colors[part], edgecolor="white", linewidth=0.5)
            bottom = [b + v for b, v in zip(bottom, vals)]
        # The machine goes on the axis label, not in a title.
        ax[i].set_xlabel(f"cell, ordered by query latency — {machine}")
        ax[i].set_ylabel("framework overhead per query (µs)")
        ax[i].tick_params(axis="x", labelsize=7)
        ax[i].grid(alpha=0.3, axis="y")
        ax[i].legend(fontsize=8)
    # One shared y-scale, so the machines are actually comparable by eye.
    top = max(a_.get_ylim()[1] for a_ in ax)
    for a_ in ax:
        a_.set_ylim(0, top)

    fig.tight_layout()
    out = os.path.join(fig_dir, "e2_query_latency_breakdown.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_figure(per_machine, fig_dir):
    """What decomposition costs, against how heavy the query is.

    The quantity plotted is the IN-PROCESS framework term -- entry + handoff +
    exit, measured from spans on one clock inside the Choreo process. It is the
    part of E2 the apparatus actually resolves: its interval is a few percent
    wide, where the monolith-vs-Choreo difference in the table above has an
    interval that contains zero at most cells. Both are reported; only this one
    is worth a figure.

    Left: absolute, showing it is a fixed per-query cost that barely moves as
    the query gets 40x heavier. Right: the same number as a share of the query,
    which is the amortization claim -- a fixed cost against a growing query.
    The batch sweep is drawn as a line (one workload, scaled); the other two
    models are drawn as unconnected hollow markers, because they sit on a
    different axis and joining them would imply a trend that is not there.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    style = {"m3pro": ("tab:blue", "o"), "gb10": ("tab:orange", "s")}

    def series(cs, a_, key, lo_k, hi_k, color, marker, connect, label):
        if not cs:
            return
        xs = [c["breakdown"]["in_pipeline"] / 1000.0 for c in cs]
        f = [c["framework_ci"] for c in cs]
        ys = [d[key] for d in f]
        yerr = [[d[key] - d[lo_k] for d in f], [d[hi_k] - d[key] for d in f]]
        a_.errorbar(xs, ys, yerr=yerr, color=color, ms=5, capsize=3,
                    fmt=(marker + "-") if connect else marker,
                    mfc=color if connect else "none",
                    lw=1.3 if connect else 0, label=label)

    for machine, cells in per_machine.items():
        color, marker = style.get(machine, ("tab:green", "^"))
        have = [c for c in cells if c.get("framework_ci")]
        # The batch sweep is ONE workload getting heavier, so a line through it
        # means something. The model-sweep cells are different networks that
        # happen to land at similar latencies; joining them to the batch sweep
        # would draw a trend across two unrelated axes, which is how the
        # earlier version of this figure grew a spurious zig-zag near 500 ms.
        batch = sorted([c for c in have if c["model"] == ANCHOR_MODEL],
                       key=lambda c: c["batch"])
        model = sorted([c for c in have if c["model"] != ANCHOR_MODEL],
                       key=lambda c: c["breakdown"]["in_pipeline"])
        for a_, key, lo_k, hi_k in ((ax[0], "abs_us", "abs_lo_us", "abs_hi_us"),
                                    (ax[1], "pct", "pct_lo", "pct_hi")):
            series(batch, a_, key, lo_k, hi_k, color, marker, True,
                   f"{machine} — batch sweep")
            series(model, a_, key, lo_k, hi_k, color, marker, False,
                   f"{machine} — other models, b8")
    ax[0].set_ylabel("framework cost (µs/query)")
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylabel("framework cost (% of query latency)")
    ax[1].set_yscale("log")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("query latency (ms, log scale)")
        a_.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(fig_dir, "e2_modularity_scale.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main():
    global DROP_RUNS, ESTIMATOR, MAX_REGIME_RATIO
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--machines", nargs="+", default=["m3pro", "gb10"],
                    help="which machines to report (the filename token)")
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="queries dropped at the head of each run")
    ap.add_argument("--drop-runs", type=int, default=DROP_RUNS,
                    help="discard the first N repetitions of each cell entirely. "
                         "Defaults to 1 and should stay there: the first "
                         "repetition is slower for its WHOLE duration, and on a "
                         "smoke collection it moved one term by 800 us.")
    ap.add_argument("--max-regime-ratio", type=float, default=MAX_REGIME_RATIO,
                    help="drop repetitions whose configurations differ in time "
                         "per query by more than this ratio (e.g. 1.25); catches "
                         "bistable cells. 0 disables. Outcome-independent: it "
                         "looks at the times, never at the cost they imply.")
    ap.add_argument("--estimator", choices=["median", "mean"], default=ESTIMATOR,
                    help="how per-run paired differences are combined "
                         "(median = robust to one contaminated repetition)")
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "paper_assets"))
    ap.add_argument("--no-breakdown", action="store_true",
                    help="skip the span-derived breakdown (it reads the local "
                         "MLflow store, which is slower than the CSV tables)")
    ap.add_argument("--latex", metavar="MACHINE", nargs="?", const="gb10", default=None)
    args = ap.parse_args()
    DROP_RUNS = args.drop_runs
    ESTIMATOR = args.estimator
    MAX_REGIME_RATIO = args.max_regime_ratio

    if args.latex is not None:
        metas = load(args.results_dir, args.latex)
        cells = collect_cells(metas, args.warmup)
        if not cells:
            sys.exit(f"no complete cells for {args.latex} in {args.results_dir}")
        attach_breakdown(args.latex, cells, metas, args.warmup)
        print_latex(cells, args.latex)
        return

    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    print("# E2 — the cost of decomposition (EfficientNetV2 / Imagenette)\n")
    print("Two things are run per cell: the **monolith** — a bare PyTorch loop, "
          "run with --no-radt, no framework and no tracing — and "
          "**choreo-traced**, the same work expressed as a two-stage graph in "
          "the framework with tracing on. Their order is rotated every "
          "repetition so neither always absorbs the warm-up.\n")
    print("Choreo writes NO per-query CSV rows: both `disable_logs` flags are "
          "set, so its every number here comes from spans. The monolith has no "
          "spans by construction and is timed from its own log, which is the "
          "only instrument it carries.\n")
    print(f"Metric of record: TIME PER QUERY — start-to-start between consecutive "
          f"queries, i.e. 1/throughput, covering the whole cycle including data "
          f"loading and preprocessing. On the Choreo side this is in-pipeline "
          f"L_q, which excludes loadgen admission (the monolith has no admission "
          f"step). Estimator: {ESTIMATOR} of per-run paired "
          f"differences; 95% CI bootstrapped over run pairs. Queries dropped at "
          f"the head of each run: {args.warmup}. Repetitions dropped as system "
          f"warm-up: {DROP_RUNS}.\n")

    per_machine = {}
    for machine in args.machines:
        metas = load(args.results_dir, machine)
        if not metas:
            print(f"\n## {machine}: no CSVs in {args.results_dir}\n")
            continue
        cells = collect_cells(metas, args.warmup)
        if not cells:
            print(f"\n## {machine}: no complete cells yet "
                  f"({len(metas)} CSVs present)\n")
            continue
        per_machine[machine] = cells
        print(f"\n# ===== {machine} ({len(metas)} CSVs, {len(cells)} cells) =====")
        print_cells(cells, machine)
        if not args.no_breakdown:
            attach_breakdown(machine, cells, metas, args.warmup)
            print_breakdown(machine, cells)
            print_span_cost(machine, cells)
            print_identity(machine, cells)
            print_sweeps(cells, machine)

    if per_machine:
        figs = [make_figure(per_machine, fig_dir),
                make_breakdown_figure(per_machine, fig_dir)]
        for f in figs:
            if f:
                print(f"\n**Figure:** `{f}`")


if __name__ == "__main__":
    main()
