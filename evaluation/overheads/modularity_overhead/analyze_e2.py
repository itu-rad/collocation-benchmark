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
framework overheads (entry, handoff, exit, turnaround). It sums to the time per
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
MONOLITH, CHOREO, CHOREO_TRACED = "monolith", "choreo", "choreo-traced"
CONFIGS = (MONOLITH, CHOREO, CHOREO_TRACED)
CONFIG_DESC = {
    MONOLITH:      "bare PyTorch loop, no framework",
    CHOREO:        "the framework, tracing off",
    CHOREO_TRACED: "the framework, tracing on",
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
    q_mono = periods_by_run(sel(MONOLITH), **kw)
    q_cho  = periods_by_run(sel(CHOREO), **kw)
    q_tra  = periods_by_run(sel(CHOREO_TRACED), **kw)
    if not q_mono or not q_cho:
        return None

    dropped = []
    if MAX_REGIME_RATIO:
        arms = [q_mono, q_cho, q_tra] if q_tra else [q_mono, q_cho]
        arms, dropped = _regime_filter(arms, MAX_REGIME_RATIO)
        if q_tra:
            q_mono, q_cho, q_tra = arms
        else:
            q_mono, q_cho = arms
        if not q_mono or not q_cho:
            return None

    out = {"model": model, "batch": batch, "regime_dropped": dropped,
           "q_monolith": summarize(q_mono), "q_choreo": summarize(q_cho),
           "q_traced": summarize(q_tra) if q_tra else None,
           # what decomposing costs, and what tracing costs on top of it
           "cost_of_choreo": paired_overhead(q_mono, q_cho),
           "cost_of_tracing": paired_overhead(q_cho, q_tra) if q_tra else None}

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
def _ov(o):
    if o is None:
        return "—", "—"
    return (f"{o['abs_us']:+.1f} [{o['abs_lo_us']:+.1f}, {o['abs_hi_us']:+.1f}]",
            f"{o['pct']:+.3f}% [{o['pct_lo']:+.3f}, {o['pct_hi']:+.3f}]")


def print_cells(cells, machine):
    """Time per query per configuration, and what each layer costs."""
    print(f"\n## {machine} — time per query, and what decomposition costs\n")
    print("| cell | R | monolith (ms) | choreo (ms) | cost of choreo (µs) | as % "
          "| cost of tracing (µs) | as % |")
    print("|---|--:|--:|--:|---|--:|---|--:|")
    for c in cells:
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        ch_abs, ch_pct = _ov(c["cost_of_choreo"])
        tr_abs, tr_pct = _ov(c["cost_of_tracing"])
        R = c["cost_of_choreo"]["pairs"] if c["cost_of_choreo"] else 0
        print(f"| {name} | {R} | {c['q_monolith']['median']:.2f} | "
              f"{c['q_choreo']['median']:.2f} | {ch_abs} | "
              f"{ch_pct.split(' ')[0]} | {tr_abs} | {tr_pct.split(' ')[0]} |")
    print("\n(time per query = start-to-start between consecutive queries = "
          "1/throughput, covering the whole cycle including data loading. "
          "'cost of choreo' = choreo − monolith; 'cost of tracing' = "
          "choreo-traced − choreo. Brackets: 95% CI, bootstrap over run pairs.)")
    print("\nA NEGATIVE cost is not a speed-up: it means the difference is "
          "smaller than what this apparatus resolves at that cell. It is printed "
          "as measured rather than clipped.")
    for c in cells:
        if c.get("regime_dropped"):
            print(f"- MIXED-REGIME repetitions dropped, "
                  f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}: "
                  f"{c['regime_dropped']}")
    for c in cells:
        if c["cost_of_choreo"]:
            pp = " / ".join(f"{v:+.1f}" for v in c["cost_of_choreo"]["per_pair_us"])
            print(f"- per-run paired differences (µs), "
                  f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}: {pp}")


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
        lab = f"mod_m{model}_b{batch}_{CHOREO_TRACED}_{machine}_r{r}"
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


def print_breakdown(machine, cells, metas, warmup, store=None):
    """Where a query's latency goes, split into stage work and framework cost."""
    print(f"\n## {machine} — query latency breakdown (traced configuration)\n")
    print("| cell | R | entry | dataloader | handoff | training | exit | turnaround "
          "| framework | total | framework % |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    any_row = False
    for c in cells:
        runs = sorted({m["run"] for m in select(metas, config=CHOREO_TRACED,
                                                model=c["model"], batch=c["batch"])})
        if DROP_RUNS:
            runs = runs[DROP_RUNS:]
        bd = breakdown_by_run(machine, c["model"], c["batch"], runs, warmup, store)
        if not bd:
            continue
        any_row = True
        med = {k: np.median([np.median(bd[r][k]) for r in bd]) / NS_PER_US
               for k in ("entry", "dataloader", "handoff", "training", "exit",
                         "turnaround")}
        fw = med["entry"] + med["handoff"] + med["exit"] + med["turnaround"]
        tot = fw + med["dataloader"] + med["training"]
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        print(f"| {name} | {len(bd)} | {med['entry']:.1f} | {med['dataloader']:.1f} "
              f"| {med['handoff']:.1f} | {med['training']:.1f} | {med['exit']:.1f} "
              f"| {med['turnaround']:.1f} | {fw:.1f} | {tot:.1f} | "
              f"{100 * fw / tot:.2f}% |")
        c["breakdown"] = med
    if not any_row:
        print("| (no traced spans found for this machine) |")
        return
    print("\n(microseconds per query. `dataloader` and `training` are the stages "
          "doing real work; `entry`, `handoff`, `exit` and `turnaround` are the "
          "framework moving the query between them, and are what decomposition "
          "adds. They sum to the time per query by construction.)")


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
    rows = [c for c in cells if c.get("breakdown") and c.get("monolith_step")]
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
        fw = b["entry"] + b["handoff"] + b["exit"] + b["turnaround"]
        measured = c["cost_of_choreo"]["abs_us"] if c["cost_of_choreo"] else float("nan")
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
    """The amortization claim: relative cost shrinks as the query gets heavier."""
    bs = sorted([c for c in cells if c["model"] == ANCHOR_MODEL],
                key=lambda c: c["batch"])
    ms = sorted([c for c in cells if c["batch"] == ANCHOR_BATCH],
                key=lambda c: c["q_monolith"]["median"])
    for title, sel_, label in (("Batch sweep (EfficientNetV2-S)", bs, "batch"),
                               ("Model sweep (batch 8)", ms, "model")):
        if len(sel_) < 2:
            continue
        print(f"\n### {machine} — {title}\n")
        print(f"| {label} | time per query (ms) | cost of choreo (µs) | as % |")
        print("|---|--:|--:|--:|")
        for c in sel_:
            key = c["batch"] if label == "batch" else MODEL_DISPLAY.get(c["model"], c["model"])
            o = c["cost_of_choreo"]
            print(f"| {key} | {c['q_monolith']['median']:.2f} | "
                  f"{o['abs_us']:+.1f} | {o['pct']:+.3f}% |")
        qs = [c["q_monolith"]["median"] for c in sel_]
        pcts = [c["cost_of_choreo"]["pct"] for c in sel_]
        print(f"\ntime per query {qs[0]:.1f} → {qs[-1]:.1f} ms ({qs[-1]/qs[0]:.1f}×): "
              f"cost {pcts[0]:+.3f}% → {pcts[-1]:+.3f}%")


def print_latex(cells, machine):
    print("% --- E2 modularity overhead: time per query ---")
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Cost of decomposition, as time per query.}")
    print("\\label{tab:modularity-sweep}")
    print("\\begin{tabular}{lrrr}\n\\toprule")
    print("Cell & Time per query (\\si{\\milli\\second}) & "
          "Cost (\\si{\\micro\\second}) & \\% \\\\\n\\midrule")
    for c in cells:
        o = c["cost_of_choreo"]
        if not o:
            continue
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        print(f"{name} & {c['q_monolith']['median']:.2f} & "
              f"{o['abs_us']:+.1f} & {o['pct']:+.3f} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


def make_breakdown_figure(per_machine, fig_dir):
    """Where a query's latency goes, and what the framework adds to it.

    Panels 1-2, one per machine: the four AUXILIARY components stacked, per
    cell, ordered by how heavy the query is. They are what decomposition adds --
    the dataloader and training stages are the real work and are left out so the
    scaffolding is legible at all. Panel 3: that scaffolding as a share of the
    query, which is the amortization claim.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    machines = [m for m in per_machine
                if any(c.get("breakdown") for c in per_machine[m])]
    if not machines:
        return None
    parts = ("entry", "handoff", "exit", "turnaround")
    colors = {"entry": "tab:blue", "handoff": "tab:orange",
              "exit": "tab:green", "turnaround": "tab:red"}

    fig, ax = plt.subplots(1, len(machines) + 1,
                           figsize=(5.2 * (len(machines) + 1), 4.4))
    for i, machine in enumerate(machines):
        cells = sorted([c for c in per_machine[machine] if c.get("breakdown")],
                       key=lambda c: c["q_monolith"]["median"])
        labels = [f"{MODEL_DISPLAY.get(c['model'], c['model']).replace('EfficientNetV2-', '')}"
                  f"\nb{c['batch']}" for c in cells]
        bottom = [0.0] * len(cells)
        for part in parts:
            vals = [c["breakdown"][part] for c in cells]
            ax[i].bar(labels, vals, bottom=bottom, label=part,
                      color=colors[part], edgecolor="white", linewidth=0.5)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax[i].set_title(machine, fontsize=10)
        ax[i].set_ylabel("framework overhead per query (µs)")
        ax[i].tick_params(axis="x", labelsize=7)
        ax[i].grid(alpha=0.3, axis="y")
        ax[i].legend(fontsize=8)

    for machine in machines:
        cells = sorted([c for c in per_machine[machine] if c.get("breakdown")],
                       key=lambda c: c["q_monolith"]["median"])
        xs = [c["q_monolith"]["median"] for c in cells]
        ys = [100.0 * sum(c["breakdown"][k] for k in parts)
              / (sum(c["breakdown"][k] for k in parts)
                 + c["breakdown"]["dataloader"] + c["breakdown"]["training"])
              for c in cells]
        marker = {"m2pro": "o-", "gb10": "s-"}.get(machine, "^-")
        ax[-1].plot(xs, ys, marker, ms=5, lw=1.3, label=machine)
    ax[-1].set_xscale("log")
    ax[-1].set_yscale("log")
    ax[-1].set_xlabel("time per query (ms, log scale)")
    ax[-1].set_ylabel("framework overhead (% of query, log scale)")
    ax[-1].grid(alpha=0.3, which="both")
    ax[-1].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(fig_dir, "e2_query_latency_breakdown.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_figure(per_machine, fig_dir):
    """Cost of decomposition against how heavy the query is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    style = {"m2pro": ("tab:blue", "o"), "gb10": ("tab:orange", "s")}
    for machine, cells in per_machine.items():
        color, marker = style.get(machine, ("tab:green", "^"))
        cs = sorted([c for c in cells if c["cost_of_choreo"]],
                    key=lambda c: c["q_monolith"]["median"])
        if not cs:
            continue
        xs = [c["q_monolith"]["median"] for c in cs]
        for a, key, lo_k, hi_k in ((ax[0], "pct", "pct_lo", "pct_hi"),
                                   (ax[1], "abs_us", "abs_lo_us", "abs_hi_us")):
            ys = [c["cost_of_choreo"][key] for c in cs]
            lo = [c["cost_of_choreo"][key] - c["cost_of_choreo"][lo_k] for c in cs]
            hi = [c["cost_of_choreo"][hi_k] - c["cost_of_choreo"][key] for c in cs]
            a.errorbar(xs, ys, yerr=[lo, hi], fmt=marker + "-", color=color,
                       ms=5, lw=1.3, capsize=3, label=machine)
    ax[0].set_ylabel("cost of decomposition (% of time per query)")
    ax[1].set_ylabel("cost of decomposition (µs/query)")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("time per query (ms, log scale)")
        a.axhline(0, color="k", lw=0.8, alpha=0.5)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
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
    ap.add_argument("--machines", nargs="+", default=["m2pro", "gb10"],
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
        print_latex(cells, args.latex)
        return

    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    print("# E2 — the cost of decomposition (EfficientNetV2 / Imagenette)\n")
    print("Three things are run per cell: the **monolith** (a bare PyTorch loop), "
          "**choreo** (the framework, tracing off), and **choreo-traced** (the "
          "framework, tracing on). Their order is rotated every repetition so "
          "none of them always absorbs the warm-up.\n")
    print(f"Metric of record: TIME PER QUERY — start-to-start between consecutive "
          f"queries, i.e. 1/throughput, covering the whole cycle including data "
          f"loading and preprocessing. Estimator: {ESTIMATOR} of per-run paired "
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
            print_breakdown(machine, cells, metas, args.warmup)
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
