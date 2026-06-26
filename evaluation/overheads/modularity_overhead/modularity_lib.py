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


def parse_baseline_steps(path):
    return _stage_durations(path, BASELINE_STEP)


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


def pool_steps(metas, parse_fn, warmup=100):
    """Pool per-step durations across runs, dropping the first `warmup` steps/run."""
    vec = []
    for m in metas:
        d = parse_fn(m["path"])
        vec.extend(d[warmup:])
    return vec


# --- statistics --------------------------------------------------------------

def summarize(vec_ns, unit_ns=NS_PER_US, n_boot=10000, seed=0):
    """median/mean/p95 + bootstrap 95% CI on the median, scaled to unit_ns."""
    a = np.asarray(vec_ns, dtype=np.float64)
    n = a.size
    if n == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p95": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.median(a[rng.integers(0, n, size=(n_boot, n))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(n),
        "median": float(np.median(a)) / unit_ns,
        "mean": float(a.mean()) / unit_ns,
        "p95": float(np.percentile(a, 95)) / unit_ns if n >= 100 else float("nan"),
        "ci_lo": lo / unit_ns,
        "ci_hi": hi / unit_ns,
    }


def overhead_ratio_ci(base_ns, choreo_ns, n_boot=10000, seed=0):
    """Two-independent-sample bootstrap of (median_c-median_b)/median_b and the
    absolute difference. CIs come from resampling EACH pool independently (the two
    arms are separate process executions; not paired). 'Within noise' = ratio CI
    contains 0 (abs CI contains 0)."""
    b = np.asarray(base_ns, dtype=np.float64)
    c = np.asarray(choreo_ns, dtype=np.float64)
    med_b, med_c = float(np.median(b)), float(np.median(c))
    ratio = (med_c - med_b) / med_b
    abs_ns = med_c - med_b
    rng = np.random.default_rng(seed)
    mb = np.median(b[rng.integers(0, b.size, size=(n_boot, b.size))], axis=1)
    mc = np.median(c[rng.integers(0, c.size, size=(n_boot, c.size))], axis=1)
    ratios = np.sort((mc - mb) / mb)
    absds = np.sort(mc - mb)
    rlo, rhi = np.percentile(ratios, [2.5, 97.5])
    alo, ahi = np.percentile(absds, [2.5, 97.5])
    within_noise = bool(rlo <= 0.0 <= rhi)
    return {
        "median_base_ns": med_b, "median_choreo_ns": med_c,
        "ratio": ratio, "ratio_lo": float(rlo), "ratio_hi": float(rhi),
        "abs_ns": abs_ns, "abs_lo": float(alo), "abs_hi": float(ahi),
        "within_noise": within_noise,
    }
