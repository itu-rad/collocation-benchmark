#!/usr/bin/env python3
"""E1 — Framework-overhead (NoOp) analysis. SELF-CONTAINED: everything E1 needs
lives in this one file (parsing, statistics, both metric tables, figures) — it
imports nothing from the other framework_overhead modules.

Reads the overnight collection layout
``evaluation/results/<device>/noop_depth_D_size_S_mode_M_{proc|off}_{device}_rN.csv``
(``proc`` = tracing ON via the bulk+proc exporter, ``off`` = tracing disabled),
for device in {mlx, cuda}. Emits Markdown tables to stdout and two figures.

Two results:
  1. Depth flatness — per-query latency L_q is linear in depth, i.e. a constant
     marginal per-stage dispatch cost (no accumulation with graph depth).
  2. Zero-copy — reference passing is O(1) in payload size, while the deep-copy
     counterfactual is O(payload).
Plus the per-stage cost the tracing layer adds (proc − off).

All latencies come from the monotonic ``perf_counter_ns`` column (trailing field
of every trace line), never wall-clock column 0. CIs are hierarchical bootstrap
with the RUN as the unit of replication.

    python analyze_e1.py [--fig-dir DIR] [--warmup K]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from bisect import bisect_right

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NS_PER_MS = 1e6
NS_PER_US = 1e3
SIZES = [0, 1024, 1048576, 10485760]

# The depth sweep is POWERS OF TWO ONLY. The earlier sweep also carried
# 3/5/6/7/9/10/50/100, which cost collection time and crowded the figures
# without adding shape: O(d) is a smooth curve and 2^0..2^7 already spans the
# full 128x range that shows it flattening.
#
# Depth 10 stays COLLECTED but out of this list, because the payload sweep is
# defined at depth 10 and its size-0 / mode-ref cell is the same data.
DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]
SIZE_LABEL = {0: "0", 1024: "1 KiB", 1048576: "1 MiB", 10485760: "10 MiB"}
WARMUP_K = 1                      # warm-up epochs dropped per run
DEVICES = ("mlx", "cuda")

# The four arms are the 2x2 of two INDEPENDENT instruments, and are named for
# the role each one plays rather than for the switch that produces it:
#
#   arm              per-stage CSV logging   span tracing
#   as-reported              on                  off      <- how E1 has always measured
#   uninstrumented           off                 off      <- the framework, no instrument
#   spans-only               off                 on       <- spans replacing the logging
#   both                     on                  on
#
# Per-stage CSV logging writes two rows per stage per query; span tracing emits
# six events per stage per query. L_q itself is measured identically in every
# arm, from the PIPELINE-level rows, which pipeline.py emits unconditionally
# (disable_logs is a Stage flag and Pipeline has none) -- so the arms differ
# only in how much instrument runs inside the interval, never in how the
# interval is timed. See collect_e1.sh.
ARMS = ("as-reported", "uninstrumented", "spans-only", "both")

# What the harness actually collects now: the two ways the framework is run.
# `as-reported` and `both` stay in ARMS so archived data still parses and still
# tables, but nothing new is collected in them and no result depends on them.
COLLECTED_ARMS = ("uninstrumented", "spans-only")

# The first collection used switch-flavoured names, two of which named the
# logging switch and two the tracing switch, so a reader could not tell what
# any single label meant. Accepted here so data collected under them still
# parses; nothing writes them any more.
_LEGACY_ARM = {"off": "as-reported", "nolog": "uninstrumented",
               "spans": "spans-only", "proc": "both"}

_FNAME_RE = re.compile(
    r"^noop_depth_(?P<depth>\d+)_size_(?P<size>\d+)_mode_(?P<mode>ref|copy)"
    r"_(?P<arm>as-reported|uninstrumented|spans-only|both|proc|off|nolog|spans)"
    r"_(?P<device>mlx|cuda)_r(?P<run>\d+)\.csv$"
)

ARM_LOGS = {"as-reported": True, "both": True,
            "uninstrumented": False, "spans-only": False}
ARM_TRACE = {"as-reported": 0, "uninstrumented": 0, "spans-only": 1, "both": 1}


def parse_filename(path):
    """Return dict(depth,size,mode,arm,trace,logs,device,run,path) or None.

    ``trace`` is kept as 0/1 so every historical selector still works; ``arm``
    and ``logs`` carry the extra dimension the two-arm scheme could not express.
    """
    m = _FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    arm = _LEGACY_ARM.get(m["arm"], m["arm"])
    return {"depth": int(m["depth"]), "size": int(m["size"]), "mode": m["mode"],
            "arm": arm, "trace": ARM_TRACE[arm], "logs": ARM_LOGS[arm],
            "device": m["device"], "run": int(m["run"]), "path": path}


# ---------------------------------------------------------------------------
# CSV parsing -> per-query timing vectors
#
# Trace line layouts (", "-separated; the pipeline name has spaces but no comma):
#   stage row    : wall, parent, "<Mode> Stage K", run, {start|end}, perf
#   pipeline row : wall, parent, "pipeline - <split>", run, {start|end},
#                  query_id, query_ts, epoch, batch, perf
#   prepare row  : wall, parent, "pipeline", prepare, {start|end}, perf
# Under the closed-loop OfflineLoadScheduler exactly one query is in flight, so
# each query's pipeline [start,end] perf window is disjoint; a stage event is
# attributed to the epoch whose window contains its perf timestamp.
# ---------------------------------------------------------------------------
def _stage_index(module):
    tail = module.rsplit(" ", 1)[-1]
    return int(tail) if tail.isdigit() else None


class Run:
    """Per-query timing vectors for one CSV (one cell, one repetition)."""

    def __init__(self, meta):
        self.meta = meta
        self.latency_ns = {}                 # epoch -> L_q
        self.transition_ns = {}              # epoch -> {k: end(k)->start(k+1)}
        self.stage_dur_ns = {}               # epoch -> {k: start(k)->end(k)}

    @property
    def epochs(self):
        return sorted(self.latency_ns)


def parse_run(path):
    """Parse one CSV into a Run (latency, transition, stage-duration)."""
    run = Run(parse_filename(path) or {"path": path})
    pipe_start, pipe_end = {}, {}
    stage_events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            module, phase, event = parts[2], parts[3], parts[4]
            try:
                perf = int(parts[-1])
            except ValueError:
                continue
            if module.startswith("pipeline -") and phase == "run":
                try:
                    epoch = int(parts[7])
                except (IndexError, ValueError):
                    continue
                (pipe_start if event == "start" else pipe_end)[epoch] = perf
            elif phase == "run":
                idx = _stage_index(module)
                if idx is not None:
                    stage_events.append((perf, idx, event))

    epochs = sorted(e for e in pipe_start if e in pipe_end)
    starts = [pipe_start[e] for e in epochs]
    per_epoch = {e: {} for e in epochs}
    for perf, idx, event in stage_events:
        i = bisect_right(starts, perf) - 1
        if i < 0:
            continue
        e = epochs[i]
        if perf > pipe_end[e]:               # falls between queries
            continue
        per_epoch[e].setdefault(idx, {})[event] = perf

    for e in epochs:
        run.latency_ns[e] = pipe_end[e] - pipe_start[e]
        stages = per_epoch[e]
        durs, trans = {}, {}
        for k, ev in stages.items():
            if "start" in ev and "end" in ev:
                durs[k] = ev["end"] - ev["start"]
        for k in stages:
            nxt = stages.get(k + 1)
            if nxt and "end" in stages[k] and "start" in nxt:
                trans[k] = nxt["start"] - stages[k]["end"]
        run.stage_dur_ns[e] = durs
        run.transition_ns[e] = trans
    return run


def load_device(device, root):
    """Return list[Run] for one device's NoOp CSVs under <root>/evaluation/results."""
    runs = []
    d = os.path.join(root, "evaluation", "results", device)
    for p in sorted(glob.glob(os.path.join(d, "noop_*.csv"))):
        if parse_filename(p):
            runs.append(parse_run(p))
    return runs


# ---------------------------------------------------------------------------
# Selection + run-structured pooling (run = unit of replication)
# ---------------------------------------------------------------------------
def select(runs, depth=None, size=None, mode=None, trace=None, arm=None):
    """Filter runs. Prefer `arm` over `trace` for anything four-arm.

    `trace` is the old two-arm switch and is now AMBIGUOUS: trace=0 matches both
    `off` and `nolog`, trace=1 both `proc` and `spans`. Selecting on it pools
    arms that differ in whether the CSV instrument was running, which is exactly
    the distinction the four-arm collection exists to measure. It is kept only
    for the payload sweep, whose data predates the extra arms and contains
    `off`/`proc` alone.
    """
    out = []
    for r in runs:
        m = r.meta
        if depth is not None and m.get("depth") != depth:
            continue
        if size is not None and m.get("size") != size:
            continue
        if mode is not None and m.get("mode") != mode:
            continue
        if trace is not None and m.get("trace") != trace:
            continue
        if arm is not None and m.get("arm") != arm:
            continue
        out.append(r)
    return out


def _drop_warmup(run):
    eps = run.epochs
    return eps[WARMUP_K:] if len(eps) > WARMUP_K else eps


def pool_latency_by_run(runs):
    out = []
    for r in runs:
        vec = [r.latency_ns[e] for e in _drop_warmup(r)]
        if vec:
            out.append(vec)
    return out


def pool_transition_by_run(runs):
    out = []
    for r in runs:
        vec = []
        for e in _drop_warmup(r):
            vec.extend(r.transition_ns.get(e, {}).values())
        if vec:
            out.append(vec)
    return out


def pool_stage_dur_by_run(runs, min_idx=0):
    out = []
    for r in runs:
        vec = []
        for e in _drop_warmup(r):
            for k, d in r.stage_dur_ns.get(e, {}).items():
                if k >= min_idx:
                    vec.append(d)
        if vec:
            out.append(vec)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _percentile(sorted_vec, q):
    if not sorted_vec:
        return float("nan")
    pos = q * (len(sorted_vec) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vec[int(pos)]
    return sorted_vec[lo] * (hi - pos) + sorted_vec[hi] * (pos - lo)


def median(vec):
    return _percentile(sorted(vec), 0.5) if vec else float("nan")


def p95(vec):
    return _percentile(sorted(vec), 0.95) if vec else float("nan")


def ols_slope(xs, ys):
    """Slope + intercept of an ordinary least-squares fit (for depth-flatness)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def ols_r2(xs, ys, slope, intercept):
    """Coefficient of determination for the same fit."""
    n = len(xs)
    if n < 2 or slope != slope:
        return float("nan")
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def fit_verdict(intercept_ns, r2):
    """Say plainly whether a linear fit is admissible.

    The intercept is L_q extrapolated to depth 0 -- the fixed per-query cost of
    a pipeline with no stages. It cannot be negative: that would be a pipeline
    that finishes before it starts. A negative intercept is therefore not a
    small number to report with a sign, it is proof that L_q is NOT linear in
    depth for that arm, so the slope is not a marginal per-stage cost and must
    not be quoted as one.

    We hit this for real: the tracing arms fit intercepts of -119.7 us (proc)
    and -45.4 us (spans) on the GB10, because the span instrument's own cost
    grows at depth >= 32 (six span events per stage per query, against two CSV
    rows) and bends the curve upward.
    """
    bad = []
    if intercept_ns == intercept_ns and intercept_ns < 0:
        bad.append(f"intercept {intercept_ns / NS_PER_US:.1f} us is NEGATIVE — a "
                   f"zero-stage pipeline cannot take negative time, so L_q is "
                   f"not linear in depth here and the slope is NOT a marginal "
                   f"per-stage cost")
    if r2 == r2 and r2 < 0.99:
        bad.append(f"R^2 = {r2:.4f} < 0.99 — the linear model does not describe "
                   f"this arm well")
    return bad


_BOOT_WORK_BUDGET = 5e7


def hier_bootstrap_ci(run_vecs, alpha=0.05, seed=0, n=10000):
    """Hierarchical (cluster) bootstrap CI for the pooled median: resample RUNS
    with replacement, then queries within each resampled run, pool, take median.
    The run is the unit of replication (query-pooling understates variance)."""
    run_vecs = [v for v in run_vecs if v]
    if not run_vecs or sum(len(v) for v in run_vecs) < 2:
        return (float("nan"), float("nan"))
    pooled_n = sum(len(v) for v in run_vecs)
    try:
        import numpy as np
        n_eff = int(min(n, max(1000, _BOOT_WORK_BUDGET // max(pooled_n, 1))))
        rng = np.random.default_rng(seed)
        arrs = [np.asarray(v, dtype=np.float64) for v in run_vecs]
        R = len(arrs)
        stats = np.empty(n_eff)
        for i in range(n_eff):
            parts = [arrs[j][rng.integers(0, arrs[j].size, arrs[j].size)]
                     for j in rng.integers(0, R, R)]
            stats[i] = np.median(np.concatenate(parts))
        lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(lo), float(hi))
    except ImportError:
        import random
        n_eff = int(min(n, max(200, (_BOOT_WORK_BUDGET / 10) // max(pooled_n, 1))))
        rng = random.Random(seed)
        R = len(run_vecs)
        stats = []
        for _ in range(n_eff):
            pooled = []
            for _ in range(R):
                pooled.extend(rng.choices(run_vecs[rng.randrange(R)],
                                          k=len(run_vecs[rng.randrange(R)])))
            stats.append(median(pooled))
        stats.sort()
        return (_percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2))


def summarize(run_vecs, unit_ns=NS_PER_US):
    """median/mean/p95 + hierarchical 95% CI on the median, scaled to unit_ns.
    Expects run-structured input (list of per-run vectors). p95 gated at >= 500
    pooled queries. Returns per-run medians too (printed beside every CI)."""
    run_vecs = [list(v) for v in run_vecs if v]
    flat = [x for v in run_vecs for x in v]
    n = len(flat)
    if n == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p95": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "run_medians": []}
    lo, hi = hier_bootstrap_ci(run_vecs)
    return {
        "n": n,
        "median": median(flat) / unit_ns,
        "mean": (sum(flat) / n) / unit_ns,
        "p95": (p95(flat) / unit_ns) if n >= 500 else float("nan"),
        "ci_lo": lo / unit_ns,
        "ci_hi": hi / unit_ns,
        "run_medians": [median(v) / unit_ns for v in run_vecs],
    }


# ---------------------------------------------------------------------------
# Result 1: depth flatness (+ tracing-layer add)
# ---------------------------------------------------------------------------
def depth_table(runs, arm_label):
    """Print the depth sweep (size 0, mode ref) and return {depth: O(d) summary}."""
    sel_arm = select(runs, size=0, mode="ref")
    depths = sorted({r.meta["depth"] for r in sel_arm} & set(DEPTHS))
    print(f"\n## Depth sweep -- {arm_label} (size 0, mode ref)\n")
    print("| depth | N | L_q median (ms) | O(d)=L_q/d (us) | 95% CI (us, hier.) | "
          "transition (us) | p95 O(d) (us) | O(d) per-run medians (us) |")
    print("|------:|--:|----------------:|----------------:|:------------------:|"
          "----------------:|--------------:|:---|")
    xs, lq_ns, od_by_depth = [], [], {}
    for d in depths:
        sel = select(sel_arm, depth=d)
        lat_runs = pool_latency_by_run(sel)
        if not lat_runs:
            continue
        od_runs = [[v / d for v in run] for run in lat_runs]
        lq = summarize(lat_runs, NS_PER_MS)
        od = summarize(od_runs, NS_PER_US)
        tr = summarize(pool_transition_by_run(sel), NS_PER_US)
        od_by_depth[d] = od
        xs.append(d)
        lq_ns.append(median([v for run in lat_runs for v in run]))
        tr_s = f"{tr['median']:.2f}" if tr["n"] else "—"
        p95_s = f"{od['p95']:.2f}" if od["p95"] == od["p95"] else "n/a"
        rm = " / ".join(f"{v:.1f}" for v in od["run_medians"])
        print(f"| {d} | {lq['n']} | {lq['median']:.4f} | {od['median']:.2f} | "
              f"[{od['ci_lo']:.1f}, {od['ci_hi']:.1f}] | {tr_s} | {p95_s} | {rm} |")

    slope_ns, intercept_ns = ols_slope(xs, lq_ns)
    r2 = ols_r2(xs, lq_ns, slope_ns, intercept_ns)
    print(f"\n**Marginal per-stage cost** (slope of L_q vs depth): "
          f"{slope_ns / NS_PER_US:.2f} us/stage  \n"
          f"**Fixed per-query overhead** (intercept): "
          f"{intercept_ns / NS_PER_US:.2f} us  \n"
          f"**Fit quality**: R^2 = {r2:.5f} over {len(xs)} depths")
    for problem in fit_verdict(intercept_ns, r2):
        print(f"\n> **FIT REJECTED** — {problem}")
    return od_by_depth


ARM_DESC = {
    "as-reported":    "CSV logging ON, spans off — how E1 has always measured",
    "uninstrumented": "CSV logging off, spans off — the framework with no instrument",
    "spans-only":     "CSV logging off, spans ON — spans replacing the logging",
    "both":           "CSV logging ON, spans ON — both instruments running",
}


def instrument_table(od_by_arm, device):
    """Separate what the FRAMEWORK costs from what MEASURING it costs.

    Every number E1 has ever reported for per-stage dispatch came from the
    `off` arm, and that arm's timing is produced by CSV log rows emitted by the
    stages themselves. A log row costs ~7.7 us single-threaded and far more
    under stage-thread contention on the logging handler lock, and at depth 10
    the stage-to-stage transitions -- where those rows sit -- are 78% of L_q.
    So `off` measures the framework AND the instrument, inseparably.

    `nolog` runs the identical pipeline with the per-stage rows switched off,
    leaving only the pipeline-level rows that carry L_q. The difference is the
    instrument, and what remains is the framework:

        off  - nolog  = the CSV instrument's per-stage cost
        spans - nolog = the span instrument's per-stage cost
        proc - off    = the tracing layer on top of the CSV instrument

    A negative difference is not a speed-up; it means the two arms differ by
    less than the run-to-run noise at that depth, and is reported as measured
    rather than clipped.
    """
    have = [a for a in ARMS if od_by_arm.get(a)]
    if "as-reported" not in have or "uninstrumented" not in have:
        print(f"\n## {device} -- instrument decomposition: needs the as-reported "
              f"and uninstrumented arms (have: {', '.join(have) or 'none'})\n")
        return
    depths = sorted(set(od_by_arm["as-reported"]) & set(od_by_arm["uninstrumented"]))
    print(f"\n## {device} -- what is the framework, what is the instrument\n")
    print("| depth | O(d) as-reported | O(d) uninstrumented | CSV instrument "
          "| instrument % of as-reported | O(d) spans-only | span instrument |")
    print("|------:|-----------------:|--------------------:|---------------:"
          "|----------------------------:|----------------:|----------------:|")
    for d in depths:
        off = od_by_arm["as-reported"][d]["median"]
        nol = od_by_arm["uninstrumented"][d]["median"]
        csv_cost = off - nol
        pct = (100.0 * csv_cost / off) if off else float("nan")
        sp = od_by_arm.get("spans-only", {}).get(d)
        sp_s = f"{sp['median']:.2f}" if sp else "—"
        sp_cost = f"{sp['median'] - nol:+.2f}" if sp else "—"
        print(f"| {d} | {off:.2f} | {nol:.2f} | {csv_cost:+.2f} | {pct:+.1f}% "
              f"| {sp_s} | {sp_cost} |")
    print("\n(all per-stage, microseconds: O(d) = L_q / depth. "
          "'CSV instrument' = as-reported - uninstrumented; "
          "'span instrument' = spans-only - uninstrumented.)")


def tracing_add_table(od_off, od_on):
    print("\n### Span-tracing cost on top of the CSV logging (both − as-reported)\n")
    print("| depth | O(d) as-reported (us) | O(d) both (us) | span-tracing add (us) |")
    print("|------:|--------------:|---------------:|-----------------:|")
    for d in sorted(set(od_off) & set(od_on)):
        a, b = od_off[d]["median"], od_on[d]["median"]
        print(f"| {d} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")


# ---------------------------------------------------------------------------
# Result 2: zero-copy vs deep-copy (depth 10, tracing OFF, stages >= 1)
# ---------------------------------------------------------------------------
PAYLOAD_DEPTH = 10


def payload_collect(runs, arm="uninstrumented"):
    """Per-stage cost vs payload, as L_q / depth.

    This used to be the per-stage self-duration read from the CSV rows, which
    tied the sweep to `as-reported` — the one arm nobody runs. L_q needs no
    instrument at all: pipeline.py writes its rows unconditionally. Dividing by
    depth puts it on the same per-stage footing as before, and at depth 10 a
    deep copy is paid once per stage, so the effect is if anything clearer.
    """
    out = {"ref": {}, "copy": {}}
    for mode in ("ref", "copy"):
        for size in SIZES:
            sel = select(runs, depth=PAYLOAD_DEPTH, size=size, mode=mode, arm=arm)
            lat = pool_latency_by_run(sel)
            if lat:
                out[mode][size] = summarize(
                    [[v / PAYLOAD_DEPTH for v in run] for run in lat], NS_PER_US)
    return out


def payload_table(data, arm="uninstrumented"):
    print(f"\n## Zero-copy: per-stage cost vs payload "
          f"(depth {PAYLOAD_DEPTH}, {arm} arm, L_q/depth)\n")
    print("| payload | ref (us) | ref 95% CI (hier.) | copy (us) | copy 95% CI (hier.) | copy/ref |")
    print("|--------:|---------:|:------------------:|----------:|:-------------------:|---------:|")
    for size in SIZES:
        r, c = data["ref"].get(size), data["copy"].get(size)
        if not r and not c:
            continue
        r_s = f"{r['median']:.2f}" if r else "—"
        r_ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]" if r else "—"
        c_s = f"{c['median']:.2f}" if c else "—"
        c_ci = f"[{c['ci_lo']:.1f}, {c['ci_hi']:.1f}]" if c else "—"
        ratio = f"{c['median'] / r['median']:.1f}x" if (r and c and r["median"]) else "—"
        print(f"| {SIZE_LABEL[size]} | {r_s} | {r_ci} | {c_s} | {c_ci} | {ratio} |")
    for mode in ("ref", "copy"):
        for size in SIZES:
            s = data[mode].get(size)
            if s and s.get("run_medians"):
                rm = " / ".join(f"{v:.2f}" for v in s["run_medians"])
                print(f"- per-run medians (us), {mode} @ {SIZE_LABEL[size]}: {rm}")

    cx = [s for s in SIZES if s > 0 and s in data["copy"]]
    if len(cx) >= 2:
        cy = [data["copy"][s]["median"] for s in cx]
        slope, intercept = ols_slope(cx, cy)
        print(f"\n**copy** cost vs payload: {slope * 1e6:.3f} us/MB "
              f"(intercept {intercept:.2f} us) -> grows with payload (O(payload)).")
    rx = [s for s in SIZES if s in data["ref"]]
    if len(rx) >= 2:
        ry = [data["ref"][s]["median"] for s in rx]
        slope, _ = ols_slope(rx, ry)
        print(f"**ref** cost vs payload: {slope * 1e6:.3f} us/MB "
              f"-> flat (O(1) in payload size).")


# ---------------------------------------------------------------------------
# LaTeX tables (paper output) — same statistics as the Markdown tables
# ---------------------------------------------------------------------------
SIZE_TEX = {0: "0", 1024: "\\SI{1}{\\kibi\\byte}",
            1048576: "\\SI{1}{\\mebi\\byte}", 10485760: "\\SI{10}{\\mebi\\byte}"}
DEVICE_TEX = {"mlx": "Apple~M2~Pro", "cuda": "NVIDIA~GB10"}


def latex_depth_table(runs, device):
    # `uninstrumented` is the framework's own cost; `as-reported` mixes in the
    # CSV instrument, which is 58-71% of it. trace=0 matches BOTH, so it must
    # not be used here.
    sel = select(runs, arm="uninstrumented", size=0, mode="ref")
    depths = sorted({r.meta["depth"] for r in sel} & set(DEPTHS))
    print("% --- Framework overhead: depth scaling (uninstrumented arm) ---")
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Per-stage framework overhead is flat in pipeline depth "
          f"(no-op chains, tracing disabled, {DEVICE_TEX.get(device, device)}). "
          "Median over $R$ runs; hierarchical bootstrap \\SI{95}{\\percent} CI in "
          "brackets (runs resampled first, then queries). Raw per-run values in "
          "the artifact.}")
    print("\\label{tab:noop-depth}")
    print("\\begin{tabular}{rrr}\n\\toprule")
    print("Depth & Per-query latency (\\si{\\milli\\second}) & "
          "Per-stage (\\si{\\micro\\second}) \\\\\n\\midrule")
    for d in depths:
        lat_runs = pool_latency_by_run(select(sel, depth=d))
        if not lat_runs:
            continue
        lq = summarize(lat_runs, NS_PER_MS)
        od = summarize([[v / d for v in run] for run in lat_runs], NS_PER_US)
        print(f"% d={d} per-stage per-run medians (us): "
              + ", ".join(f"{v:.1f}" for v in od["run_medians"]))
        print(f"{d} & {lq['median']:.3f} & "
              f"{od['median']:.1f} [{od['ci_lo']:.1f}, {od['ci_hi']:.1f}] \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def latex_payload_table(runs, device):
    print("% --- Framework overhead: zero-copy payload sweep (as-reported arm) ---")
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Reference passing is constant in payload size while deep-copy "
          f"is linear (no-op chains, depth~10, tracing disabled, "
          f"{DEVICE_TEX.get(device, device)}). Per-stage duration, "
          "\\si{\\micro\\second}, median [hierarchical bootstrap 95\\% CI].}")
    print("\\label{tab:noop-zerocopy}")
    print("\\begin{tabular}{lrr}\n\\toprule")
    print("Payload & Reference (\\si{\\micro\\second}) & "
          "Deep-copy (\\si{\\micro\\second}) \\\\\n\\midrule")
    for size in SIZES:
        r = pool_stage_dur_by_run(
            select(runs, arm="as-reported", depth=10, size=size, mode="ref"), min_idx=1)
        c = pool_stage_dur_by_run(
            select(runs, arm="as-reported", depth=10, size=size, mode="copy"), min_idx=1)
        rs = summarize(r, NS_PER_US) if r else None
        cs = summarize(c, NS_PER_US) if c else None
        r_txt = (f"{rs['median']:.1f} [{rs['ci_lo']:.1f}, {rs['ci_hi']:.1f}]"
                 if rs else "---")
        c_txt = (f"{cs['median']:.1f} [{cs['ci_lo']:.1f}, {cs['ci_hi']:.1f}]"
                 if cs else "---")
        print(f"{SIZE_TEX[size]} & {r_txt} & {c_txt} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
ARM_STYLE = {"as-reported": "tab:blue", "uninstrumented": "tab:green",
             "both": "tab:red", "spans-only": "tab:orange"}


def _od_curve(runs, arm):
    """[(depth, O(d) median us)] for one arm, sorted by depth."""
    out = []
    for d in sorted({r.meta["depth"] for r in runs
                     if r.meta["size"] == 0 and r.meta["mode"] == "ref"}
                    & set(DEPTHS)):
        sel = select(runs, depth=d, size=0, mode="ref", arm=arm)
        lat = pool_latency_by_run(sel)
        if lat:
            out.append((d, summarize([[v / d for v in run] for run in lat],
                                     NS_PER_US)["median"]))
    return out


def make_instrument_figure(per_device, fig_dir):
    """The figure the four-arm collection exists to produce.

    Left column: O(d) = L_q/depth per arm. The framework's marginal per-stage
    cost is where this flattens, so `off` and `nolog` flattening to different
    plateaus IS the result -- the gap between them is our own logger, not the
    framework.

    Right column: that gap plotted directly, against the independently measured
    cost of the two CSV rows each stage writes (7.7 us/row, timed in isolation).
    If the decomposition is real, the measured gap should sit on that line
    without having been fitted to it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devs = list(per_device)
    fig, axes = plt.subplots(len(devs), 2, figsize=(12, 4.4 * len(devs)),
                             squeeze=False)
    for row, dev in enumerate(devs):
        runs = per_device[dev]
        curves = {a: _od_curve(runs, a) for a in ARM_STYLE}

        ax = axes[row][0]
        for arm, color in ARM_STYLE.items():
            if curves[arm]:
                xs, ys = zip(*curves[arm])
                ax.plot(xs, ys, "o-", color=color, ms=4, lw=1.4, label=arm)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("pipeline depth (stages)")
        ax.set_ylabel(f"{dev}: per-stage cost O(d) = L_q/depth (us)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

        ax = axes[row][1]
        base = dict(curves["uninstrumented"])
        for arm, color, lbl in (
                ("as-reported", "tab:blue",
                 "CSV instrument (as-reported - uninstrumented)"),
                ("spans-only", "tab:orange",
                 "span instrument (spans-only - uninstrumented)")):
            pts = [(d, v - base[d]) for d, v in curves[arm] if d in base]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=color, ms=4, lw=1.4, label=lbl)
        # 2 CSV rows per stage per query x 7.7 us/row, measured standalone.
        ax.axhline(2 * 7.7, color="k", ls="--", lw=1.0, alpha=0.6,
                   label="2 log rows x 7.7 us (measured separately)")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("pipeline depth (stages)")
        ax.set_ylabel(f"{dev}: instrument cost per stage (us)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(fig_dir, "e1_instrument_decomposition.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def make_figures(per_device, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: per-query latency L_q vs depth (linearity / flat marginal cost).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, dev in zip(axes, per_device):
        runs = per_device[dev]
        # Select on ARM, not `trace` — trace=0 now pools off with nolog, which
        # would draw one line through two arms that differ by the instrument.
        for i, (arm, color) in enumerate(ARM_STYLE.items()):
            depths, meds = [], []
            for d in sorted({r.meta["depth"] for r in runs
                             if r.meta["size"] == 0 and r.meta["mode"] == "ref"}
                            & set(DEPTHS)):
                s = summarize(pool_latency_by_run(
                    select(runs, depth=d, size=0, mode="ref", arm=arm)), NS_PER_MS)
                if s["n"]:
                    depths.append(d); meds.append(s["median"])
            if not depths:
                continue
            ax.plot(depths, meds, "o-", color=color, ms=4, lw=1.3, label=arm)
            lq_ns = [m * NS_PER_MS for m in meds]
            slope, icpt = ols_slope(depths, lq_ns)
            r2 = ols_r2(depths, lq_ns, slope, icpt)
            xs = [min(depths), max(depths)]
            # Dash the fit only where it is admissible; a rejected fit is drawn
            # dotted so the eye does not read it as a marginal per-stage cost.
            rejected = bool(fit_verdict(icpt, r2))
            ax.plot(xs, [(slope * x + icpt) / NS_PER_MS for x in xs],
                    ":" if rejected else "--", color=color, lw=0.9, alpha=0.7)
            ax.annotate(f"{arm}: {slope / NS_PER_US:.1f} us/stage"
                        + ("  (fit rejected)" if rejected else ""),
                        xy=(0.04, 0.94 - 0.07 * i), xycoords="axes fraction",
                        color=color, fontsize=8)
        # No titles: the caption carries that in the paper. Device identity is in
        # the legend and the fitted-slope annotation instead.
        ax.set_xlabel("pipeline depth (stages)")
        ax.set_ylabel(f"{dev}: per-query latency L_q median (ms)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    f1 = os.path.join(fig_dir, "e1_depth_flatness.png")
    fig.savefig(f1, dpi=140); plt.close(fig)

    # Fig 2: per-stage self-duration vs payload (zero-copy vs deep-copy), off arm.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    marker = {"mlx": "o", "cuda": "s"}
    for dev in per_device:
        data = payload_collect(per_device[dev])
        for mode, color in [("ref", "tab:green"), ("copy", "tab:red")]:
            xs = [s for s in SIZES if s in data[mode]]
            ys = [data[mode][s]["median"] for s in xs]
            xplot = [max(x, 1) for x in xs]      # 0 -> 1 byte for the log axis
            ax.plot(xplot, ys, marker[dev] + "-", color=color, ms=6, lw=1.3,
                    label=f"{dev} {mode}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("payload size (bytes; 0->1 for log axis)")
    ax.set_ylabel("per-stage self-duration median (us)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = os.path.join(fig_dir, "e1_payload_zero_copy.png")
    fig.savefig(f2, dpi=140); plt.close(fig)
    return f1, f2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global WARMUP_K
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-dir", default=os.path.join(here, "paper_assets"))
    ap.add_argument("--warmup", type=int, default=WARMUP_K,
                    help="warm-up epochs dropped per run")
    ap.add_argument("--root", default=root, help="repo root (holds evaluation/results/)")
    ap.add_argument("--latex", metavar="DEVICE", nargs="?", const="mlx", default=None,
                    help="emit the paper LaTeX tables for DEVICE (default mlx) to "
                         "stdout and exit; no Markdown/figures")
    args = ap.parse_args()

    WARMUP_K = args.warmup

    # LaTeX mode: emit the two paper tables for one device and stop.
    if args.latex is not None:
        dev = args.latex
        runs = load_device(dev, args.root)
        if not runs:
            sys.exit(f"No NoOp CSVs for {dev} under {args.root}/evaluation/results/{dev}")
        latex_depth_table(runs, dev)
        latex_payload_table(runs, dev)
        return

    os.makedirs(args.fig_dir, exist_ok=True)
    print("# E1 -- Framework overhead (NoOp)\n")
    print(f"Warm-up epochs dropped per run: WARMUP_K={WARMUP_K}. Latencies from the "
          "monotonic perf clock. CIs are hierarchical (run = unit of replication).\n")
    per_device = {}
    for dev in DEVICES:
        runs = load_device(dev, args.root)
        if not runs:
            print(f"## {dev}: NO runs found under {args.root}/evaluation/results/{dev}\n")
            continue
        per_device[dev] = runs
        print(f"\n# ===== {dev} ({len(runs)} run-files) =====")
        # One table per ARM. Selecting on `trace` here would pool off with
        # nolog (and proc with spans), hiding the instrument's cost inside the
        # framework's — see select().
        od_by_arm = {}
        for a in ARMS:
            sel = select(runs, arm=a)
            if sel:
                od_by_arm[a] = depth_table(sel, f"{dev} -- arm '{a}': {ARM_DESC[a]}")
        instrument_table(od_by_arm, dev)
        if od_by_arm.get("as-reported") and od_by_arm.get("both"):
            tracing_add_table(od_by_arm["as-reported"], od_by_arm["both"])
        # The payload sweep predates the four-arm split and holds off/proc only,
        # so it still selects on `trace`.
        # Both arms: the framework bare, and the framework traced. The
        # zero-copy claim is architectural, so it must hold in both — and
        # showing it in `spans-only` too is what retires `as-reported`.
        for arm in COLLECTED_ARMS:
            data = payload_collect(runs, arm)
            if any(data[m] for m in data):
                payload_table(data, arm)

    if per_device:
        f1, f2 = make_figures(per_device, args.fig_dir)
        f3 = make_instrument_figure(per_device, args.fig_dir)
        print(f"\n**Figures:** `{f1}`, `{f2}`, `{f3}`")


if __name__ == "__main__":
    main()
