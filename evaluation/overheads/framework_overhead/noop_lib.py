"""Shared parsing + statistics for the framework-overhead (NoOp) microbenchmark.

Both the Markdown write-up and the LaTeX tables consume THIS module so they can
never disagree. All latencies are derived from the monotonic ``perf_counter_ns``
column (the trailing field of every trace line), never the wall-clock column 0
(which exists only for RadT cross-process alignment).

Trace line layouts (fields are ``", "``-separated; the pipeline name contains
spaces but no commas, so a plain comma split is safe):

  stage row    : wall, parent, "<Mode> Stage K", run, {start|end}, perf
  pipeline row : wall, parent, "pipeline - <split>", run, {start|end},
                 query_id, query_ts, epoch, batch, perf
  prepare row  : wall, parent, "pipeline", prepare, {start|end}, perf

Stage rows carry NO epoch. Under the closed-loop OfflineLoadScheduler exactly one
query is in flight, so each query's pipeline ``[start, end]`` perf window is
disjoint; we attribute a stage event to the epoch whose window contains its perf
timestamp. (Depth-1 has no inter-stage transition; handled as undefined.)
"""

from __future__ import annotations

import glob
import math
import os
import re
from bisect import bisect_right

_FNAME_RE = re.compile(
    r"^noop_d(?P<depth>\d+)_s(?P<size>\d+)_m(?P<mode>ref|copy)"
    r"_t(?P<trace>[01])_r(?P<run>\d+)\.csv$"
)

NS_PER_MS = 1e6
NS_PER_US = 1e3


def parse_filename(path):
    """Return dict(depth,size,mode,trace,run) or None if it isn't a matrix CSV."""
    m = _FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return {
        "depth": int(m["depth"]), "size": int(m["size"]), "mode": m["mode"],
        "trace": int(m["trace"]), "run": int(m["run"]), "path": path,
    }


def _stage_index(module):
    """Trailing integer of a stage module name, or None (name-agnostic)."""
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
    """Parse one matrix CSV into a Run (latency, transition, stage-duration)."""
    meta = parse_filename(path)
    run = Run(meta or {"path": path})

    pipe_start, pipe_end = {}, {}            # epoch -> perf ns
    stage_events = []                        # (perf, idx, event)
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

    # Bracket stage events into the epoch whose [start,end] perf window holds them.
    epochs = sorted(e for e in pipe_start if e in pipe_end)
    starts = [pipe_start[e] for e in epochs]
    per_epoch = {e: {} for e in epochs}      # epoch -> {idx: {start,end}}
    for perf, idx, event in stage_events:
        i = bisect_right(starts, perf) - 1
        if i < 0:
            continue
        e = epochs[i]
        if perf > pipe_end[e]:               # falls in the gap between queries
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


# --- collection helpers ------------------------------------------------------

def load_matrix(results_dir):
    """Return list[Run] for every matrix CSV in results_dir."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "noop_d*_t*_r*.csv"))):
        if parse_filename(path):
            runs.append(parse_run(path))
    return runs


def select(runs, depth=None, size=None, mode=None, trace=None):
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
        out.append(r)
    return out


def _drop_warmup(run):
    """Epochs of a run with the first (smallest) epoch dropped as warm-up."""
    eps = run.epochs
    return eps[1:] if len(eps) > 1 else eps


def pool_latency(runs):
    """Pooled per-query latency (ns) across runs, warm-up dropped."""
    return [v for run in pool_latency_by_run(runs) for v in run]


def pool_transition(runs):
    """Pooled per-stage transition cost (ns) across runs, warm-up dropped."""
    return [v for run in pool_transition_by_run(runs) for v in run]


def pool_stage_dur(runs, min_idx=0):
    """Pooled stage self-duration (ns); min_idx skips e.g. the injector stage 0."""
    return [v for run in pool_stage_dur_by_run(runs, min_idx) for v in run]


# Run-structured variants: one vector per run, preserving the cluster structure
# the hierarchical bootstrap needs (the run, not the query, is the unit of
# replication — see experimental_setup.tex §Statistics).

def pool_latency_by_run(runs):
    """Per-run per-query latency vectors (ns), warm-up dropped, empties removed."""
    out = []
    for r in runs:
        vec = [r.latency_ns[e] for e in _drop_warmup(r)]
        if vec:
            out.append(vec)
    return out


def pool_transition_by_run(runs):
    """Per-run per-stage transition vectors (ns), warm-up dropped."""
    out = []
    for r in runs:
        vec = []
        for e in _drop_warmup(r):
            vec.extend(r.transition_ns.get(e, {}).values())
        if vec:
            out.append(vec)
    return out


def pool_stage_dur_by_run(runs, min_idx=0):
    """Per-run stage self-duration vectors (ns); min_idx skips the injector."""
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


# --- statistics --------------------------------------------------------------

def _percentile(sorted_vec, q):
    if not sorted_vec:
        return float("nan")
    pos = q * (len(sorted_vec) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vec[int(pos)]
    return sorted_vec[lo] * (hi - pos) + sorted_vec[hi] * (pos - lo)


def median(vec):
    return _percentile(sorted(vec), 0.5) if vec else float("nan")


def p95(vec):
    return _percentile(sorted(vec), 0.95) if vec else float("nan")


def bootstrap_ci(vec, stat=median, n=10000, alpha=0.05, seed=0):
    """LEGACY query-pooled percentile bootstrap CI for ``stat``.

    Treats pooled per-query values as independent — pseudoreplication when they
    come from R correlated runs; kept only for flat-vector callers. New code
    passes run-structured data to :func:`summarize` (-> hierarchical CI).
    """
    import random
    if len(vec) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(vec)
    stats = []
    for _ in range(n):
        sample = [vec[rng.randrange(m)] for _ in range(m)]
        stats.append(stat(sample))
    stats.sort()
    return (_percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2))


# Cap on (replicates x pooled size) so deep-pipeline cells stay tractable; the
# replicate count never drops below 1000 (Monte-Carlo error << 5-cluster
# uncertainty) nor rises above 10^4.
_BOOT_WORK_BUDGET = 5e7


def hier_bootstrap_ci(run_vecs, alpha=0.05, seed=0, n=10000):
    """Hierarchical (cluster) bootstrap CI for the pooled median.

    Resamples RUNS with replacement first, then queries within each resampled
    run, pools, and takes the median (experimental_setup.tex §Statistics). This
    is the CI of record; query-pooled bootstrapping understates variance.
    Deterministic via fixed seed. Uses numpy when available (the torch envs all
    ship it); pure-stdlib fallback with a reduced work budget otherwise.
    """
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
            parts = []
            for j in rng.integers(0, R, R):
                a = arrs[j]
                parts.append(a[rng.integers(0, a.size, a.size)])
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
                run = run_vecs[rng.randrange(R)]
                pooled.extend(rng.choices(run, k=len(run)))
            stats.append(median(pooled))
        stats.sort()
        return (_percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2))


def summarize(vec, unit_ns=NS_PER_US):
    """median/mean/p95 + 95% CI on the median, scaled to ``unit_ns``.

    Pass RUN-STRUCTURED data (list of per-run vectors, from the ``*_by_run``
    helpers) to get the hierarchical bootstrap CI of record plus the raw
    per-run medians (``run_medians``, printed beside every CI per the paper's
    statistics rules). A flat vector falls back to the legacy query-pooled CI
    (``ci_kind`` says which you got). p95 is gated at >= 500 pooled queries.
    """
    nested = bool(vec) and isinstance(vec[0], (list, tuple))
    if nested:
        run_vecs = [list(v) for v in vec if v]
        flat = [x for v in run_vecs for x in v]
    else:
        run_vecs = None
        flat = list(vec)
    n = len(flat)
    if n == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p95": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "ci_kind": "none", "run_medians": []}
    if nested:
        lo, hi = hier_bootstrap_ci(run_vecs)
        ci_kind = "hierarchical"
        run_medians = [median(v) / unit_ns for v in run_vecs]
    else:
        lo, hi = bootstrap_ci(flat)
        ci_kind = "pooled-legacy"
        run_medians = []
    return {
        "n": n,
        "median": median(flat) / unit_ns,
        "mean": (sum(flat) / n) / unit_ns,
        "p95": (p95(flat) / unit_ns) if n >= 500 else float("nan"),
        "ci_lo": lo / unit_ns,
        "ci_hi": hi / unit_ns,
        "ci_kind": ci_kind,
        "run_medians": run_medians,
    }


def ols_slope(xs, ys):
    """Slope + intercept of an ordinary least-squares fit (for depth-flatness)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def default_results_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
