"""Shared parsing + statistics for the staged contention analyzer (Stages A-D).

Design of record: CONTENTION_EXPERIMENTS_REDESIGN.md (E3'/E6', staged form,
signed off 2026-07-14). This module owns everything analyze_staged.py needs
that is *format*-shaped: trace parsing with per-pipeline separation, arrivals
sidecar matching (coordinated-omission-safe response anchoring), AMC bandwidth
CSV joins, and the hierarchical (run-cluster) bootstrap machinery, which
originated in the E1 overhead analysis (now analyze_e1.py).

Trace CSV row layouts (fields are comma-separated; pipeline and stage names
contain spaces but never commas, so a plain comma split is safe; the trailing
field is always perf_counter_ns stamped by utils/logger.py PERF_FORMAT):

  pipeline row : wall, <pipeline name>, "pipeline - <split>", run,
                 {start|end}, query_id, query_submitted_ts, epoch, batch, perf
                 (emitted by pipeline/pipeline.py:268/166; batch is 1-based
                  there, the arrivals sidecar counter is 0-based)
  stage row    : wall, <pipeline name>, <stage name>, run, {start|end}, perf
                 (emitted by stages/stage.py log_phase_single; NO epoch)
  prepare rows : phase == "prepare" — ignored here.

Multi-pipeline separation: column 1 (the pipeline name) is authoritative.
Orchestrated staged cells may curate either ONE merged CSV per run or one CSV
per pipeline process; parse_trace_files() accepts a list of files and merges
by pipeline name, so both layouts analyze identically.

Stage-duration attribution: a stage is a single thread that processes queries
in FIFO order, so its run start/end events alternate in perf order; we pair
them sequentially and attribute the k-th execution to the k-th query submitted
to that pipeline. (This holds under overlap/pipelining, where E1's
window-bracketing approach would not.)

Response-time anchoring (coordinated omission): the response of query i is
    end_wall - intended_arrival(i)        when an arrivals sidecar matches,
    end_wall - query_submitted_ts(i)      otherwise (actual-arrival anchor).
Arrivals rows are rank-matched to queries in submission order and validated
against the trace's own submitted timestamps. NOTE (verify on real staged
data): in multi-pipeline cells every open-loop scheduler in the process tree
writes the SAME <label>_arrivals.csv path, so the surviving sidecar's owner
is inferred by timestamp agreement, never assumed.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "pilots"))

import pilot_lib  # noqa: E402  (parse_arrivals, load_knobs/get_knob)

NS = 1e9

parse_arrivals = pilot_lib.parse_arrivals
load_knobs = pilot_lib.load_knobs
get_knob = pilot_lib.get_knob


# Inlined from what used to be framework_overhead/noop_lib.py. E1 was
# consolidated into a single self-contained analyze_e1.py and the shared library
# went away with it; these three are the only pieces anything outside E1 used.
# Linear-interpolated percentile, matching numpy's default and pilot_lib's.
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


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

@dataclass
class QueryRec:
    epoch: int
    batch: int                    # 1-based, as logged
    query_id: str = ""
    submitted_ts: float = float("nan")   # wall, loadgen enqueue time
    start_wall: float = float("nan")
    end_wall: float = float("nan")
    start_perf: int = -1
    end_perf: int = -1

    @property
    def complete(self) -> bool:
        return self.start_perf >= 0 and self.end_perf >= 0

    @property
    def latency_s(self) -> float:
        """Pipeline-internal latency (dequeue -> completion), perf clock."""
        return (self.end_perf - self.start_perf) / NS if self.complete else float("nan")


@dataclass
class PipelineTrace:
    name: str
    queries: list[QueryRec] = field(default_factory=list)   # submission order
    # stage name -> list of (start_perf, end_perf, start_wall, end_wall),
    # sequential-pairing order == FIFO query order through that stage
    stage_execs: dict[str, list[tuple[int, int, float, float]]] = field(
        default_factory=dict)
    stage_unpaired: dict[str, int] = field(default_factory=dict)
    # stage name -> per-query generated-token counts ("n_generated_tokens"
    # rows from the instrumented generator stages), FIFO order matching
    # stage_execs; empty for uninstrumented traces
    stage_token_counts: dict[str, list[int]] = field(default_factory=dict)

    @property
    def completed(self) -> list[QueryRec]:
        return [q for q in self.queries if q.complete]

    def span_wall(self) -> tuple[float, float]:
        """(first submission, last completion) wall window; NaNs if empty."""
        qs = self.completed
        if not qs:
            return (float("nan"), float("nan"))
        t0 = min(q.submitted_ts if not math.isnan(q.submitted_ts) else q.start_wall
                 for q in qs)
        t1 = max(q.end_wall for q in qs)
        return (t0, t1)


def parse_trace_files(paths) -> dict[str, PipelineTrace]:
    """Parse one or more trace CSVs into per-pipeline traces (merged).

    Returns {pipeline_name: PipelineTrace}. Pipeline names come from trace
    column 1 (never from filenames), so merged and per-pipeline file layouts
    are equivalent.
    """
    # (pipeline, epoch, batch) -> partial QueryRec
    qrecs: dict[tuple[str, int, int], QueryRec] = {}
    # (pipeline, stage) -> [(perf, wall, event)]
    stage_events: dict[tuple[str, str], list[tuple[int, float, str]]] = {}
    # (pipeline, stage) -> [(perf, n_tokens)] from "n_generated_tokens" rows
    token_events: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                pipe, module, phase, event = (parts[1], parts[2], parts[3],
                                              parts[4])
                if phase == "n_generated_tokens":
                    # instrumented generator stages log the real per-query
                    # token count: wall, pipe, stage, n_generated_tokens,
                    # <int>, perf
                    try:
                        token_events.setdefault((pipe, module), []).append(
                            (int(parts[-1]), int(event)))
                    except ValueError:
                        pass
                    continue
                if event not in ("start", "end") or phase == "prepare":
                    continue
                try:
                    perf = int(parts[-1])
                    wall = float(parts[0])
                except ValueError:
                    continue
                if module.startswith("pipeline -") and phase == "run":
                    if len(parts) < 10:
                        continue
                    try:
                        key = (pipe, int(parts[7]), int(parts[8]))
                        sub_ts = float(parts[6])
                    except ValueError:
                        continue
                    rec = qrecs.setdefault(
                        key, QueryRec(epoch=key[1], batch=key[2],
                                      query_id=parts[5], submitted_ts=sub_ts))
                    if event == "start":
                        rec.start_perf, rec.start_wall = perf, wall
                        rec.submitted_ts = sub_ts
                    else:
                        rec.end_perf, rec.end_wall = perf, wall
                elif module != "pipeline":            # plain stage row
                    # phase == "run" is today's instrumentation; any other
                    # sub-phase (e.g. a future "prefill"/"first_token" event
                    # from the generator stage) is kept under
                    # "<stage>::<phase>" so Step D's TTFT split lights up the
                    # moment the instrumentation lands.
                    skey = module if phase == "run" else f"{module}::{phase}"
                    stage_events.setdefault((pipe, skey), []).append(
                        (perf, wall, event))

    traces: dict[str, PipelineTrace] = {}
    for (pipe, epoch, batch), rec in qrecs.items():
        traces.setdefault(pipe, PipelineTrace(pipe)).queries.append(rec)
    for pt in traces.values():
        pt.queries.sort(key=lambda q: (q.start_perf if q.start_perf >= 0
                                       else (q.epoch, q.batch)))
    for (pipe, stage), evs in stage_events.items():
        pt = traces.setdefault(pipe, PipelineTrace(pipe))
        evs.sort(key=lambda t: t[0])
        pairs, pending, unpaired = [], None, 0
        for perf, wall, event in evs:
            if event == "start":
                if pending is not None:
                    unpaired += 1        # start without end (crash/drop)
                pending = (perf, wall)
            elif event == "end":
                if pending is None:
                    unpaired += 1
                    continue
                pairs.append((pending[0], perf, pending[1], wall))
                pending = None
        if pending is not None:
            unpaired += 1
        pt.stage_execs[stage] = pairs
        pt.stage_unpaired[stage] = unpaired
    for (pipe, stage), evs in token_events.items():
        pt = traces.setdefault(pipe, PipelineTrace(pipe))
        evs.sort(key=lambda t: t[0])
        pt.stage_token_counts[stage] = [c for _, c in evs]
    return traces


def stage_durations_by_query(pt: PipelineTrace, stage_name: str) -> list[float]:
    """Per-query durations (s) of one stage, aligned to pt.queries order.

    Sequential FIFO pairing (see module docstring). If execution and query
    counts disagree, aligns the common prefix and pads with NaN.
    """
    execs = pt.stage_execs.get(stage_name, [])
    out = []
    for i, _q in enumerate(pt.queries):
        if i < len(execs):
            s, e, _, _ = execs[i]
            out.append((e - s) / NS)
        else:
            out.append(float("nan"))
    return out


# ---------------------------------------------------------------------------
# Arrivals sidecar: owner inference + anchored responses
# ---------------------------------------------------------------------------

def arrival_match_fraction(pt: PipelineTrace, arr, tol_s: float = 0.05) -> float:
    """Fraction of rank-matched (query, arrival) pairs whose trace submitted_ts
    agrees with the sidecar's actual_ts within tol_s."""
    qs = sorted(pt.queries, key=lambda q: q.submitted_ts)
    n = min(len(qs), arr.n)
    if n == 0:
        return 0.0
    hits = sum(1 for i in range(n)
               if abs(qs[i].submitted_ts - arr.actual_ts[i]) <= tol_s)
    return hits / n


def infer_arrivals_owner(traces: dict[str, PipelineTrace], arr,
                         tol_s: float = 0.05):
    """Best-matching pipeline for a sidecar: (name, match_fraction)."""
    best, best_frac = None, 0.0
    for name, pt in traces.items():
        frac = arrival_match_fraction(pt, arr, tol_s)
        if frac > best_frac:
            best, best_frac = name, frac
    return best, best_frac


def anchored_responses(pt: PipelineTrace, arr=None,
                       min_match: float = 0.90) -> tuple[list[float], str, dict]:
    """Per-query response times (s) for a pipeline, in submission order.

    Anchor precedence:
      * "intended"  — arrivals sidecar rank-matched (>= min_match agreement):
                      response = end_wall - intended_ts (coordinated-omission-
                      safe: queue-blocked submissions are charged in full)
      * "submitted" — trace's own query_submitted_ts (actual arrival)
    Returns (responses, anchor_kind, diagnostics).
    """
    qs = sorted(pt.completed, key=lambda q: q.submitted_ts)
    diag = {"n_completed": len(qs), "n_arrivals": arr.n if arr else 0,
            "match_frac": float("nan"), "blocked_puts": arr.blocked_puts if arr else 0,
            "max_block_s": arr.max_block_s if arr else 0.0}
    if arr is not None and arr.n:
        frac = arrival_match_fraction(pt, arr)
        diag["match_frac"] = frac
        if frac >= min_match:
            # rank-match completed queries into the full arrival schedule by
            # nearest actual_ts (some arrivals may have no completed query)
            resp = []
            j = 0
            for q in qs:
                # advance to the arrival row closest to this submission
                while (j + 1 < arr.n and
                       abs(arr.actual_ts[j + 1] - q.submitted_ts)
                       <= abs(arr.actual_ts[j] - q.submitted_ts)):
                    j += 1
                resp.append(q.end_wall - arr.intended_ts[j])
                j = min(j + 1, arr.n - 1)
            return resp, "intended", diag
    return [q.end_wall - q.submitted_ts for q in qs], "submitted", diag


# ---------------------------------------------------------------------------
# AMC bandwidth CSV join (M2 counter-backed path; scripts/amc_bandwidth_sampler)
# ---------------------------------------------------------------------------

AMC_ENGINES = ("cpu", "gpu", "ane", "other")


def load_bandwidth_csv(path) -> list[dict]:
    """Rows of an AMC sampler CSV: timestamp,dt_s,{eng}_{rd,wr}...,total_gbps."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (TypeError, ValueError):
                continue
    return rows


def bandwidth_window_stats(rows: list[dict], t0: float, t1: float) -> dict:
    """Mean GB/s per engine bucket (+ total) over the wall window [t0, t1].

    Sampler timestamps are wall-clock (same base as trace column 0), stamped
    at the END of each dt_s interval; a row is included when its interval
    overlaps the window.
    """
    sel = [r for r in rows
           if (r["timestamp"] > t0) and (r["timestamp"] - r["dt_s"] < t1)]
    if not sel or math.isnan(t0):
        return {}
    dt = sum(r["dt_s"] for r in sel)
    out = {}
    for eng in AMC_ENGINES:
        out[f"{eng}_gbps"] = sum(r[f"{eng}_rd"] + r[f"{eng}_wr"]
                                 for r in sel) / dt / 1e9
    out["total_gbps"] = sum(r["total_gbps"] * r["dt_s"] for r in sel) / dt
    out["bw_samples"] = float(len(sel))
    return out


# ---------------------------------------------------------------------------
# Hierarchical (run-cluster) bootstrap — same construction as the overhead
# analyzers' hier_bootstrap_ci (analyze_e1.py / analyze_e2.py)
# to (a) arbitrary statistics and (b) run-level scalar tables for slopes.
# ---------------------------------------------------------------------------

_BOOT_WORK_BUDGET = 5e7


def hier_boot_ci(run_vecs, stat_fn, alpha=0.05, seed=0, n=10000):
    """CI of stat_fn(pooled values) under run-then-query resampling.

    Runs are the cluster/replication unit (experimental_setup.tex
    §Statistics); resampling queries inside each resampled run mirrors
    E1's hier_bootstrap_ci exactly, with the same work budget.
    """
    import numpy as np
    run_vecs = [np.asarray([v for v in vec if not math.isnan(v)], dtype=float)
                for vec in run_vecs]
    run_vecs = [v for v in run_vecs if v.size]
    if not run_vecs or sum(v.size for v in run_vecs) < 2:
        return (float("nan"), float("nan"))
    pooled_n = sum(v.size for v in run_vecs)
    n_eff = int(min(n, max(1000, _BOOT_WORK_BUDGET // max(pooled_n, 1))))
    rng = np.random.default_rng(seed)
    R = len(run_vecs)
    stats = np.empty(n_eff)
    for i in range(n_eff):
        parts = []
        for j in rng.integers(0, R, R):
            a = run_vecs[j]
            parts.append(a[rng.integers(0, a.size, a.size)])
        stats[i] = stat_fn(np.concatenate(parts))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def run_level_ci(values, alpha=0.05, seed=0, n=10000, stat=None):
    """Bootstrap CI for a run-level scalar (e.g. per-run throughput): resample
    the R run values with replacement; stat defaults to the mean."""
    import numpy as np
    vals = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if vals.size < 2:
        return (float("nan"), float("nan"))
    stat = stat or np.mean
    rng = np.random.default_rng(seed)
    stats = np.array([stat(vals[rng.integers(0, vals.size, vals.size)])
                      for _ in range(n)])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def slope_boot_samples(points, seed=0, n=4000, normalize_at_zero=True):
    """Bootstrap distribution of the dose-response slope.

    points: list of (x, [per-run y values]) — one entry per ladder level
    (x = bytes/s or ops/s; y = the per-run foreground statistic). Each
    bootstrap replicate resamples the run values within every level (runs are
    independent across levels — each cell is its own set of runs), recomputes
    level means, and fits OLS. Returns (slope_samples, point_slope,
    point_intercept, norm_slope_samples, point_norm_slope) where the
    normalized slope is b / a (fractional change per unit x, using the fitted
    intercept a as the zero-dose baseline) — the quantity the pre-registered
    [2/3, 3/2] engine-ratio band applies to. Requires >= 2 usable levels.
    """
    import numpy as np
    pts = [(x, np.asarray([v for v in ys if not math.isnan(v)], dtype=float))
           for x, ys in points]
    pts = [(x, ys) for x, ys in pts if not math.isnan(x) and ys.size]
    if len(pts) < 2:
        return None
    xs = np.array([x for x, _ in pts])
    slope, intercept = ols_slope(list(xs), [float(ys.mean()) for _, ys in pts])
    rng = np.random.default_rng(seed)
    samples = np.empty(n)
    nsamples = np.empty(n)
    for i in range(n):
        ys_mean = np.array([ys[rng.integers(0, ys.size, ys.size)].mean()
                            for _, ys in pts])
        b, a = ols_slope(list(xs), list(ys_mean))
        samples[i] = b
        nsamples[i] = b / a if (normalize_at_zero and a) else float("nan")
    norm_slope = slope / intercept if (normalize_at_zero and intercept) \
        else float("nan")
    return samples, slope, intercept, nsamples, norm_slope


def ci_of(samples, alpha=0.05):
    import numpy as np
    s = np.asarray([v for v in samples if not math.isnan(v)])
    if s.size < 10:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def ratio_ci(samples_a, samples_b, alpha=0.05):
    """CI of slope_a / slope_b from two independent bootstrap distributions
    (paired per replicate; sign-unstable replicates where b crosses 0 yield
    huge ratios and correctly widen the CI)."""
    import numpy as np
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    m = min(a.size, b.size)
    if m < 10:
        return (float("nan"), float("nan"))
    with_ratio = a[:m] / b[:m]
    with_ratio = with_ratio[~(np.isnan(with_ratio) | np.isinf(with_ratio))]
    return ci_of(with_ratio, alpha)


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def default_results_dir(device: str) -> str:
    # Results live beside the experiment that produced them; the old
    # evaluation/collect/results tree was removed.
    return str(REPO_ROOT / "evaluation" / "contention" / "results" / device)


def global_results_dir() -> str:
    # Was evaluation/results, which was removed when results were moved to sit
    # beside their experiment.
    return str(REPO_ROOT / "evaluation" / "contention" / "results")
