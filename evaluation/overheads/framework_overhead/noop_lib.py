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
    vec = []
    for r in runs:
        for e in _drop_warmup(r):
            vec.append(r.latency_ns[e])
    return vec


def pool_transition(runs):
    """Pooled per-stage transition cost (ns) across runs, warm-up dropped."""
    vec = []
    for r in runs:
        for e in _drop_warmup(r):
            vec.extend(r.transition_ns.get(e, {}).values())
    return vec


def pool_stage_dur(runs, min_idx=0):
    """Pooled stage self-duration (ns); min_idx skips e.g. the injector stage 0."""
    vec = []
    for r in runs:
        for e in _drop_warmup(r):
            for k, d in r.stage_dur_ns.get(e, {}).items():
                if k >= min_idx:
                    vec.append(d)
    return vec


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
    """Percentile bootstrap CI for ``stat``. Deterministic via fixed seed.

    Uses the stdlib RNG (no numpy dependency). Returns (lo, hi); (nan, nan) when
    the sample is too small to resample meaningfully.
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


def summarize(vec, unit_ns=NS_PER_US):
    """median/mean/p95 + bootstrap CI on the median, scaled to ``unit_ns``."""
    n = len(vec)
    if n == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p95": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    lo, hi = bootstrap_ci(vec)
    return {
        "n": n,
        "median": median(vec) / unit_ns,
        "mean": (sum(vec) / n) / unit_ns,
        "p95": (p95(vec) / unit_ns) if n >= 100 else float("nan"),
        "ci_lo": lo / unit_ns,
        "ci_hi": hi / unit_ns,
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
