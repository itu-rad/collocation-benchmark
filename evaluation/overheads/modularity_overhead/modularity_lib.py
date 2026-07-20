"""Shared parsing + statistics for the modularity-overhead experiment.

Compares per-step training latency of a hand-written PyTorch EfficientNetV2-S
fine-tune ("baseline") against the same workload as a Choreo pipeline. All
latencies come from the monotonic ``perf_counter_ns`` column (trailing CSV
field), never wall-clock column 0 (that exists only for RadT alignment, and its
multi-thread jitter is exactly what produced the old nonsensical Table 2).

Trace lines are ``", "``-separated. Relevant rows:

  baseline step : wall, baseline_finetune, training_step, run, {start|end}, perf
  choreo train  : wall, <pipeline>, EfficientNet training, run, {start|end}, perf
  choreo load   : wall, <pipeline>, Load Imagenette samples from TorchVision Dataset, run, {start|end}, perf
  choreo pipe   : wall, <pipeline>, pipeline - <split>, run, {start|end}, qid, qts, epoch, batch, perf

The metric of record is the TRAINING-STAGE step: baseline ``training_step`` vs
Choreo ``EfficientNet training``, both bracketing identical GPU work
(.to->zero_grad->fwd->bwd->step->synchronize), both excluding data loading. Under
the closed-loop OfflineLoadScheduler (one query in flight) the stage's start/end
rows strictly alternate, so consecutive pairing is exact.

Uses numpy (always present in the torch env these runs require) for a fast
two-independent-sample bootstrap.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

_FNAME_RE = re.compile(
    r"^mod_(?P<impl>baseline|choreo)(?:_t(?P<trace>[01]))?"
    r"_d(?P<dev>[a-z0-9]+)_r(?P<run>\d+)\.csv$"
)

NS_PER_MS = 1e6
NS_PER_US = 1e3

TRAIN_STAGE = "EfficientNet training"
LOAD_STAGE = "Load Imagenette samples from TorchVision Dataset"
BASELINE_STEP = "training_step"


def default_results_dir():
    return os.path.join(_HERE, "results")


def parse_filename(path):
    """Return dict(impl, trace, dev, run, path) or None."""
    m = _FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return {
        "impl": m["impl"],
        "trace": int(m["trace"]) if m["trace"] is not None else None,
        "dev": m["dev"],
        "run": int(m["run"]),
        "path": path,
    }


# --- parsing -----------------------------------------------------------------

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


def _stage_durations(path, module_name):
    """Per-step durations (ns) for one stage: pair alternating start/end by perf."""
    evs = [(perf, event) for (mod, phase, event, perf) in _rows(path)
           if mod == module_name and phase == "run" and event in ("start", "end")]
    evs.sort()
    durs, i = [], 0
    while i < len(evs) - 1:
        if evs[i][1] == "start" and evs[i + 1][1] == "end":
            durs.append(evs[i + 1][0] - evs[i][0])
            i += 2
        else:
            i += 1  # skip a stray/unpaired event
    return durs


BASELINE_LOOP = "training_loop"


def parse_baseline_steps(path):
    """Per-step durations for the LAST session in the file.

    baseline_finetune historically opened its log in append mode, so re-runs
    accumulated stale sessions (the baseline-append bug — it can flip the core
    overhead negative). Each session emits one 'training_loop, run, start'
    marker; we reset at every marker so only the final session's steps count.
    Robust even against pre-fix contaminated files."""
    evs = []
    for (mod, phase, event, perf) in _rows(path):
        if mod == BASELINE_LOOP and phase == "run" and event == "start":
            evs = []  # new session marker: discard everything before it
        elif mod == BASELINE_STEP and phase == "run" and event in ("start", "end"):
            evs.append((perf, event))
    evs.sort()
    durs, i = [], 0
    while i < len(evs) - 1:
        if evs[i][1] == "start" and evs[i + 1][1] == "end":
            durs.append(evs[i + 1][0] - evs[i][0])
            i += 2
        else:
            i += 1
    return durs


def parse_choreo_train_steps(path):
    return _stage_durations(path, TRAIN_STAGE)


def parse_choreo_load_steps(path):
    return _stage_durations(path, LOAD_STAGE)


def parse_pipeline_latency(path):
    """End-to-end per-query latency (ns): pipeline run start->end paired per query.

    For the training pipeline, ``epoch`` is the training-epoch number (constant
    within a run) and ``batch`` is the per-query index, so the unique key is
    ``(epoch, batch)`` (parts[7], parts[8])."""
    starts, ends = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10 or not parts[2].startswith("pipeline -") or parts[3] != "run":
                continue
            try:
                key = (int(parts[7]), int(parts[8]))
                perf = int(parts[-1])
            except (IndexError, ValueError):
                continue
            (starts if parts[4] == "start" else ends)[key] = perf
    return [ends[k] - starts[k] for k in sorted(starts) if k in ends]


# --- collection --------------------------------------------------------------

def load_matrix(results_dir, device=None):
    metas = []
    for p in sorted(glob.glob(os.path.join(results_dir, "mod_*.csv"))):
        m = parse_filename(p)
        if m and (device is None or m["dev"] == device):
            metas.append(m)
    return metas


def select(metas, impl=None, trace=None):
    out = []
    for m in metas:
        if impl is not None and m["impl"] != impl:
            continue
        if trace is not None and m["trace"] != trace:
            continue
        out.append(m)
    return out


def pool_steps(metas, parse_fn, warmup=200):
    """Pool per-step durations across runs, dropping the first `warmup` steps/run.

    Default warmup=200 matches the setup text (E2: drop the first 200 steps as
    kernel-autotuning/first-call warm-up over the 1,100-step epoch)."""
    vec = []
    for _, run_vec in sorted(steps_by_run(metas, parse_fn, warmup).items()):
        vec.extend(run_vec)
    return vec


def steps_by_run(metas, parse_fn, warmup=200):
    """Per-run step durations keyed by run id: {run: [ns, ...]}.

    Preserves the cluster structure for the hierarchical bootstrap and the run
    pairing (interleaved arms share run ids) for the paired difference."""
    out = {}
    for m in metas:
        d = parse_fn(m["path"])[warmup:]
        if d:
            out[m["run"]] = d
    return out


# --- statistics --------------------------------------------------------------

_BOOT_WORK_BUDGET = 5e7  # cap replicates x pooled size; floor 1000 replicates


def _as_run_arrays(steps):
    """Normalize input to a list of per-run float arrays.

    Accepts {run: vec} (from steps_by_run), list-of-vectors, or a flat vector
    (legacy — treated as ONE cluster, which degenerates to the pooled CI)."""
    if isinstance(steps, dict):
        vecs = [steps[k] for k in sorted(steps)]
    elif steps and isinstance(steps[0], (list, tuple, np.ndarray)):
        vecs = list(steps)
    else:
        vecs = [steps] if len(steps) else []
    return [np.asarray(v, dtype=np.float64) for v in vecs if len(v)]


def _hier_boot_medians(arrs, n_boot, seed):
    """Bootstrap replicates of the pooled median: resample runs, then steps."""
    pooled_n = int(sum(a.size for a in arrs))
    n_eff = int(min(n_boot, max(1000, _BOOT_WORK_BUDGET // max(pooled_n, 1))))
    rng = np.random.default_rng(seed)
    R = len(arrs)
    out = np.empty(n_eff)
    for i in range(n_eff):
        parts = [arrs[j][rng.integers(0, arrs[j].size, arrs[j].size)]
                 for j in rng.integers(0, R, R)]
        out[i] = np.median(np.concatenate(parts))
    return out


def summarize(steps, unit_ns=NS_PER_US, n_boot=10000, seed=0):
    """median/mean/p95 + 95% CI on the median, scaled to unit_ns.

    Pass run-structured data ({run: vec} from steps_by_run, or list of per-run
    vectors) to get the hierarchical (cluster) bootstrap CI of record — runs are
    resampled first, then steps within runs — plus the raw per-run medians
    (``run_medians``), which the paper prints beside every CI. A flat vector is
    legacy query-pooled behavior (``ci_kind`` reports which). p95 gates at
    >= 500 pooled steps.
    """
    arrs = _as_run_arrays(steps)
    if not arrs:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p95": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "ci_kind": "none", "run_medians": []}
    a = np.concatenate(arrs)
    n = a.size
    hierarchical = len(arrs) > 1
    boots = _hier_boot_medians(arrs, n_boot, seed)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(n),
        "median": float(np.median(a)) / unit_ns,
        "mean": float(a.mean()) / unit_ns,
        "p95": float(np.percentile(a, 95)) / unit_ns if n >= 500 else float("nan"),
        "ci_lo": float(lo) / unit_ns,
        "ci_hi": float(hi) / unit_ns,
        "ci_kind": "hierarchical" if hierarchical else "pooled-single-cluster",
        "run_medians": [float(np.median(v)) / unit_ns for v in arrs],
    }


def overhead_ratio_ci(base, choreo, n_boot=10000, seed=0):
    """Unpaired overhead CI: (median_c - median_b) / median_b + absolute diff.

    Accepts run-structured input ({run: vec} / list of per-run vectors), in
    which case each arm is resampled with the HIERARCHICAL bootstrap (runs
    first, then steps) — the CI of record for unpaired comparisons. Flat
    vectors fall back to the legacy pooled resampling (pseudoreplication;
    kept for compatibility only). 'Within noise' = ratio CI contains 0."""
    ab = _as_run_arrays(base)
    ac = _as_run_arrays(choreo)
    b = np.concatenate(ab)
    c = np.concatenate(ac)
    med_b, med_c = float(np.median(b)), float(np.median(c))
    mb = _hier_boot_medians(ab, n_boot, seed)
    mc = _hier_boot_medians(ac, n_boot, seed + 1)
    m = min(mb.size, mc.size)
    ratios = np.sort((mc[:m] - mb[:m]) / mb[:m])
    absds = np.sort(mc[:m] - mb[:m])
    rlo, rhi = np.percentile(ratios, [2.5, 97.5])
    alo, ahi = np.percentile(absds, [2.5, 97.5])
    return {
        "median_base_ns": med_b, "median_choreo_ns": med_c,
        "ratio": (med_c - med_b) / med_b,
        "ratio_lo": float(rlo), "ratio_hi": float(rhi),
        "abs_ns": med_c - med_b, "abs_lo": float(alo), "abs_hi": float(ahi),
        "within_noise": bool(rlo <= 0.0 <= rhi),
        "ci_kind": "hierarchical" if (len(ab) > 1 and len(ac) > 1) else "pooled-legacy",
    }


def paired_overhead_ci(base_by_run, choreo_by_run, n_boot=10000, seed=0):
    """Paired across-run overhead — the statistic of record for E2.

    The arms were interleaved run-by-run (shared conditions), so runs pair by
    id: d_i = median(choreo_i) - median(base_i). Reports the mean paired
    difference (and its ratio to the pooled baseline median) with a bootstrap
    CI that resamples PAIRS with replacement and re-resamples steps within each
    chosen run — more powerful than comparing two marginal intervals for small
    fixed effects. The raw per-pair differences are returned for printing.
    """
    shared = sorted(set(base_by_run) & set(choreo_by_run))
    if len(shared) < 2:
        return None
    b = {r: np.asarray(base_by_run[r], dtype=np.float64) for r in shared}
    c = {r: np.asarray(choreo_by_run[r], dtype=np.float64) for r in shared}
    med_base = float(np.median(np.concatenate(list(b.values()))))
    d_runs = {r: float(np.median(c[r])) - float(np.median(b[r])) for r in shared}
    d_point = float(np.mean(list(d_runs.values())))

    rng = np.random.default_rng(seed)
    R = len(shared)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, R, R)
        ds = []
        for j in idx:
            r = shared[j]
            rb = b[r][rng.integers(0, b[r].size, b[r].size)]
            rc = c[r][rng.integers(0, c[r].size, c[r].size)]
            ds.append(np.median(rc) - np.median(rb))
        boots[i] = np.mean(ds)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "pairs": R,
        "d_ns": d_point, "d_lo": float(lo), "d_hi": float(hi),
        "ratio": d_point / med_base,
        "ratio_lo": float(lo) / med_base, "ratio_hi": float(hi) / med_base,
        "within_noise": bool(lo <= 0.0 <= hi),
        "d_runs_ns": [d_runs[r] for r in shared],
        "run_ids": shared,
    }
