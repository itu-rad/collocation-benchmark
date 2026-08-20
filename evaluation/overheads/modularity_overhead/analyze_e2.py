#!/usr/bin/env python3
"""E2 — Modularity-overhead analysis. SELF-CONTAINED: parsing, statistics,
tables, LaTeX and figures all live in this one file (imports nothing local).

Compares the per-step training latency of a hand-written PyTorch fine-tune (the
bare monolith, "baseline") against the identical workload wrapped as a Choreo
pipeline, across a scale sweep (batch 1..64 at EfficientNetV2-S; model S/M/L +
ConvNeXt-L at batch 8):

    core   = choreo t0 (tracing off) - baseline   -> the framework wrapper itself
    total  = choreo t2 (bulk+proc tracing) - baseline -> wrapper + span export

Result: the RELATIVE overhead amortizes toward zero as the step grows, while the
ABSOLUTE overhead stays a roughly fixed per-step cost — a small O(1) tax, not a
scaling one.

Metric of record: the TRAINING-STAGE step — baseline `training_step` vs Choreo
`EfficientNet training` — both bracketing identical GPU work and both excluding
data loading. All latencies come from the monotonic perf_counter_ns column (the
trailing CSV field), never wall-clock column 0.

Statistic of record: the PAIRED across-run difference. The arms are interleaved
run-by-run (collect.sh runs baseline, t0, t2 back-to-back within each repetition),
so runs pair by id: d_i = median(choreo_i) - median(base_i). The CI resamples
PAIRS with replacement and re-resamples steps within each chosen run.

    python analyze_e2.py [--device mps cuda] [--warmup 200] [--fig-dir DIR] [--latex DEVICE]
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
WARMUP = 200                      # steps dropped per run (kernel autotune / first-call)

TRAIN_STAGE = "EfficientNet training"     # Choreo training stage
BASELINE_STEP = "training_step"           # monolith per-step marker
BASELINE_LOOP = "training_loop"           # session marker (see parse_baseline_steps)

# arm t0 = tracing off (core wrapper); t2 = radt bulk+proc span export.
# (t1, the superseded in-process mlflow exporter, is not collected.)
ARM_CORE, ARM_TRACED = 0, 2

MODEL_DISPLAY = {"effv2s": "EfficientNetV2-S", "effv2m": "EfficientNetV2-M",
                 "effv2l": "EfficientNetV2-L", "convnextl": "ConvNeXt-L"}
MODEL_ORDER = ["effv2s", "effv2m", "convnextl", "effv2l"]   # by step-time, re-sorted at runtime
BATCHES = [1, 2, 4, 8, 16, 32, 64]
ANCHOR_MODEL, ANCHOR_BATCH = "effv2s", 8

_FNAME_RE = re.compile(
    r"^mod_(?P<impl>baseline|choreo)(?:_t(?P<trace>\d))?"
    r"_m(?P<model>[a-z0-9]+)_b(?P<batch>\d+)"
    r"_d(?P<dev>[a-z0-9]+)_r(?P<run>\d+)\.csv$"
)


def parse_filename(path):
    m = _FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return {"impl": m["impl"],
            "trace": int(m["trace"]) if m["trace"] is not None else None,
            "model": m["model"], "batch": int(m["batch"]),
            "dev": m["dev"], "run": int(m["run"]), "path": path}


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


def _pair(evs):
    """Pair alternating (perf, start/end) events into durations."""
    evs.sort()
    durs, i = [], 0
    while i < len(evs) - 1:
        if evs[i][1] == "start" and evs[i + 1][1] == "end":
            durs.append(evs[i + 1][0] - evs[i][0])
            i += 2
        else:
            i += 1                      # skip a stray/unpaired event
    return durs


def parse_baseline_steps(path):
    """Per-step durations for the LAST session in the file.

    baseline_finetune historically opened its log in append mode, so re-runs
    could accumulate stale sessions (the append bug — it can flip the measured
    overhead negative). Each session emits one `training_loop, run, start`
    marker; reset at every marker so only the final session counts. Robust even
    against pre-fix contaminated files."""
    evs = []
    for (mod, phase, event, perf) in _rows(path):
        if mod == BASELINE_LOOP and phase == "run" and event == "start":
            evs = []
        elif mod == BASELINE_STEP and phase == "run" and event in ("start", "end"):
            evs.append((perf, event))
    return _pair(evs)


def parse_choreo_steps(path):
    """Per-step durations of the Choreo training stage."""
    return _pair([(perf, ev) for (mod, ph, ev, perf) in _rows(path)
                  if mod == TRAIN_STAGE and ph == "run" and ev in ("start", "end")])


# Repetitions discarded as SYSTEM warm-up (whole runs, not steps). The first
# repetition of a cell is measurably slower for its ENTIRE duration — e.g. the
# 2026-08-18 mps anchor: baseline r1 = 97.2 ms/step vs 89.2 for r2..r6, still
# 97.4 in its last 300 steps, so per-step warm-up dropping cannot remove it
# (cold page cache / power state on the collection's first process launch).
# Left at 0 = report everything; the per-run medians printed under each table
# make such an outlier visible rather than silently dropped.
DROP_RUNS = 0


def steps_by_run(metas, warmup=WARMUP, drop_runs=None):
    """{run_id: [step_ns, ...]} with the first `warmup` steps dropped per run,
    and the lowest `drop_runs` run ids discarded entirely as system warm-up."""
    drop = DROP_RUNS if drop_runs is None else drop_runs
    keep = sorted({m["run"] for m in metas})[drop:] if drop else None
    out = {}
    for m in metas:
        if keep is not None and m["run"] not in keep:
            continue
        fn = parse_baseline_steps if m["impl"] == "baseline" else parse_choreo_steps
        d = fn(m["path"])[warmup:]
        if d:
            out[m["run"]] = d
    return out


def load(results_dir, device):
    metas = []
    for p in sorted(glob.glob(os.path.join(results_dir, "mod_*.csv"))):
        m = parse_filename(p)
        if m and m["dev"] == device:
            metas.append(m)
    return metas


def select(metas, impl=None, trace=None, model=None, batch=None):
    out = []
    for m in metas:
        if impl is not None and m["impl"] != impl:
            continue
        if trace is not None and m["trace"] != trace:
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

    d_i = median(arm_i) - median(base_i) over runs shared by both arms, combined
    across runs by ESTIMATOR (median by default; see above). The CI resamples
    PAIRS with replacement and re-resamples steps within each chosen run, and
    applies the same estimator to each bootstrap replicate. Also reports the
    difference as a % of the pooled baseline median."""
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
    """core/total overhead for one (model,batch) cell, or None if incomplete."""
    base = steps_by_run(select(metas, impl="baseline", model=model, batch=batch), warmup)
    t0 = steps_by_run(select(metas, impl="choreo", trace=ARM_CORE, model=model, batch=batch), warmup)
    t2 = steps_by_run(select(metas, impl="choreo", trace=ARM_TRACED, model=model, batch=batch), warmup)
    if not base or not t0:
        return None
    dropped = []
    if MAX_REGIME_RATIO:
        arms = [base, t0, t2] if t2 else [base, t0]
        arms, dropped = _regime_filter(arms, MAX_REGIME_RATIO)
        if t2:
            base, t0, t2 = arms
        else:
            base, t0 = arms
        if not base or not t0:
            return None
    out = {"model": model, "batch": batch, "regime_dropped": dropped,
           "base": summarize(base), "t0": summarize(t0),
           "core": paired_overhead(base, t0)}
    out["t2"] = summarize(t2) if t2 else None
    out["total"] = paired_overhead(base, t2) if t2 else None
    # tracing layer alone = traced arm vs core arm (both wrapped)
    out["tracing"] = paired_overhead(t0, t2) if t2 else None
    return out


def collect_cells(metas, warmup):
    seen = sorted({(m["model"], m["batch"]) for m in metas})
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


def print_cells(cells, device):
    print(f"\n## {device} — per-cell overhead (paired across runs)\n")
    print(f"| cell | R | step (ms) | core (µs/step, {ESTIMATOR}) | core % | "
          f"core (mean) | +tracing (µs/step) | total % |")
    print("|---|--:|--:|---|--:|--:|---|--:|")
    for c in cells:
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        core_abs, core_pct = _ov(c["core"])
        tot_abs, tot_pct = _ov(c["total"])
        R = c["core"]["pairs"] if c["core"] else 0
        cmean = f"{c['core']['abs_mean_us']:+.1f}" if c["core"] else "—"
        print(f"| {name} | {R} | {c['base']['median']:.2f} | {core_abs} | "
              f"{core_pct.split(' ')[0]} | {cmean} | {tot_abs} | "
              f"{tot_pct.split(' ')[0]} |")
    print("\n(core = Choreo wrapper, tracing off, vs the bare monolith; "
          "+tracing = wrapper + radt bulk/proc span export. "
          "Brackets: 95% CI, bootstrap over run pairs.)")
    for c in cells:
        if c.get("regime_dropped"):
            print(f"- MIXED-REGIME repetitions dropped, "
                  f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}: "
                  f"{c['regime_dropped']} (arms landed in different step-time regimes)")
    for c in cells:
        if c["core"]:
            pp = " / ".join(f"{v:+.1f}" for v in c["core"]["per_pair_us"])
            print(f"- per-run paired core diffs (µs), "
                  f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}: {pp}")


def print_sweeps(cells, device):
    """The amortization claim: relative shrinks with step size, absolute ~fixed."""
    bs = sorted([c for c in cells if c["model"] == ANCHOR_MODEL], key=lambda c: c["batch"])
    ms = sorted([c for c in cells if c["batch"] == ANCHOR_BATCH],
                key=lambda c: c["base"]["median"])
    for title, sel, label in (("Batch sweep (EfficientNetV2-S)", bs, "batch"),
                              ("Model sweep (batch 8)", ms, "model")):
        if len(sel) < 2:
            continue
        print(f"\n### {device} — {title}\n")
        print(f"| {label} | step (ms) | core µs/step | core % of step | total µs/step | total % |")
        print("|---|--:|--:|--:|--:|--:|")
        for c in sel:
            key = c["batch"] if label == "batch" else MODEL_DISPLAY.get(c["model"], c["model"])
            core = c["core"]; tot = c["total"]
            print(f"| {key} | {c['base']['median']:.2f} | "
                  f"{core['abs_us']:+.1f} | {core['pct']:+.3f}% | "
                  f"{tot['abs_us']:+.1f} | {tot['pct']:+.3f}% |" if tot else
                  f"| {key} | {c['base']['median']:.2f} | {core['abs_us']:+.1f} | "
                  f"{core['pct']:+.3f}% | — | — |")
        if len(sel) >= 3:
            steps = [c["base"]["median"] for c in sel]
            pcts = [c["core"]["pct"] for c in sel]
            abss = [c["core"]["abs_us"] for c in sel]
            # Report what the numbers do; do NOT assert the expected conclusion.
            # The claim (relative shrinks, absolute ~fixed) only holds if the
            # per-cell spread is small next to the effect — say so either way.
            spread = max(abss) - min(abss)
            rel_drops = abs(pcts[-1]) < abs(pcts[0])
            print(f"\nstep {steps[0]:.1f} → {steps[-1]:.1f} ms "
                  f"({steps[-1]/max(steps[0],1e-9):.1f}×): "
                  f"core {pcts[0]:+.3f}% → {pcts[-1]:+.3f}% "
                  f"({'relative shrinks' if rel_drops else 'relative does NOT shrink'}), "
                  f"absolute {abss[0]:+.1f} → {abss[-1]:+.1f} µs/step "
                  f"(spread {spread:.1f} µs across cells; "
                  f"{'consistent with a ~fixed cost' if spread < 200 else 'TOO NOISY to call fixed'}).")


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def print_latex(cells, device):
    print("% --- E2 modularity overhead: scale sweep ---")
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Wrapping real work in Choreo costs a small, roughly FIXED "
          "per-step amount, so the \\emph{relative} overhead amortizes toward zero "
          "as the step grows (" + device + "). Paired across-run difference; "
          "bootstrap \\SI{95}{\\percent} CI over run pairs.}")
    print("\\label{tab:modularity-sweep}")
    print("\\begin{tabular}{lrrr}\n\\toprule")
    print("Cell & Step (\\si{\\milli\\second}) & Core (\\si{\\micro\\second}/step) & "
          "Core (\\si{\\percent}) \\\\\n\\midrule")
    for c in cells:
        name = f"{MODEL_DISPLAY.get(c['model'], c['model'])} b{c['batch']}"
        o = c["core"]
        print(f"{name} & {c['base']['median']:.2f} & "
              f"{o['abs_us']:+.1f} [{o['abs_lo_us']:+.1f}, {o['abs_hi_us']:+.1f}] & "
              f"{o['pct']:+.3f} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


# ---------------------------------------------------------------------------
# Figure: 2x2 amortization panels
# ---------------------------------------------------------------------------
def make_figure(per_device, fig_dir):
    """Overhead vs STEP TIME — the variable the amortization claim is about.

    Plots 95% CIs, because most sweep cells have only 3 usable repetitions and
    their intervals cross zero: without error bars a noisy cell would look
    exactly as authoritative as the R=9 anchor. No titles — the caption carries
    that in the paper; panels are identified by their axis labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"mps": ("tab:blue", "o", "M2 Pro (mps)"),
             "cuda": ("tab:orange", "s", "GB10 (cuda)")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    for dev, cells in per_device.items():
        color, marker, label = style.get(dev, ("tab:gray", "^", dev))
        cs = sorted([c for c in cells if c["core"]], key=lambda c: c["base"]["median"])
        xs = [c["base"]["median"] for c in cs]
        for a, key, lo_k, hi_k in ((ax[0], "pct", "pct_lo", "pct_hi"),
                                   (ax[1], "abs_us", "abs_lo_us", "abs_hi_us")):
            ys = [c["core"][key] for c in cs]
            lo = [c["core"][key] - c["core"][lo_k] for c in cs]
            hi = [c["core"][hi_k] - c["core"][key] for c in cs]
            a.errorbar(xs, ys, yerr=[lo, hi], fmt=marker + "-", color=color,
                       ms=5, lw=1.3, capsize=3, elinewidth=1, alpha=0.9,
                       label=f"{label} core")
        # ring the high-confidence anchor cell (R=9)
        for c in cs:
            if c["model"] == ANCHOR_MODEL and c["batch"] == ANCHOR_BATCH:
                for a, key in ((ax[0], "pct"), (ax[1], "abs_us")):
                    a.plot([c["base"]["median"]], [c["core"][key]], marker,
                           color=color, ms=12, mfc="none", mew=2,
                           label=f"{label} anchor (R=9)")

    # A single pathological cell (mps b2, CI to +25%) would squash every other
    # point, so clip each axis to the bulk of the data and mark any interval that
    # runs off-scale rather than silently cropping it.
    for a, key, lo_k, hi_k in ((ax[0], "pct", "pct_lo", "pct_hi"),
                               (ax[1], "abs_us", "abs_lo_us", "abs_hi_us")):
        pts, los, his = [], [], []
        for cells in per_device.values():
            for c in cells:
                if c["core"]:
                    pts.append(c["core"][key])
                    los.append(c["core"][lo_k])
                    his.append(c["core"][hi_k])
        if not pts:
            continue
        span = max(max(pts) - min(pts), 1e-9)
        lim_lo = min(min(pts) - 0.6 * span, np.percentile(los, 25))
        lim_hi = max(max(pts) + 0.6 * span, np.percentile(his, 75))
        a.set_ylim(lim_lo, lim_hi)
        n_clip = sum(1 for l, h in zip(los, his) if l < lim_lo or h > lim_hi)
        if n_clip:
            a.annotate(f"{n_clip} interval(s) extend off-scale",
                       xy=(0.02, 0.03), xycoords="axes fraction",
                       fontsize=7, style="italic", color="0.35")

    ax[0].set_ylabel("core overhead (% of step)")
    ax[1].set_ylabel("core overhead (µs/step)")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("baseline step time (ms, log scale)")
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
    ap.add_argument("--devices", nargs="+", default=["mps", "cuda"])
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="steps dropped per run (within-run warm-up)")
    ap.add_argument("--drop-runs", type=int, default=DROP_RUNS,
                    help="discard the first N repetitions of each cell entirely "
                         "(SYSTEM warm-up: the first run of a collection is slower "
                         "for its whole duration; see DROP_RUNS)")
    ap.add_argument("--max-regime-ratio", type=float, default=MAX_REGIME_RATIO,
                    help="drop repetitions whose arms differ in step time by more "
                         "than this ratio (e.g. 1.25); catches bistable cells. "
                         "0 disables. Outcome-independent: looks at step times, "
                         "not at the measured overhead.")
    ap.add_argument("--estimator", choices=["median", "mean"], default=ESTIMATOR,
                    help="how per-run paired differences are combined "
                         "(median = robust to a contaminated repetition)")
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "..", "paper_assets"))
    ap.add_argument("--latex", metavar="DEVICE", nargs="?", const="cuda", default=None)
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
    print("# E2 — Modularity overhead (real workload, scale sweep)\n")
    print(f"Estimator: {ESTIMATOR} of per-run paired differences "
          f"(runs dropped as system warm-up: {DROP_RUNS}).")
    print(f"Warm-up steps dropped per run: {args.warmup}. Metric: training-stage step "
          "(monotonic perf clock). Statistic: paired across-run difference "
          "(arms interleaved per repetition); 95% CI bootstrapped over run pairs.\n")
    per_device = {}
    for dev in args.devices:
        metas = load(args.results_dir, dev)
        if not metas:
            print(f"\n## {dev}: no CSVs in {args.results_dir}\n")
            continue
        cells = collect_cells(metas, args.warmup)
        if not cells:
            print(f"\n## {dev}: no complete cells yet ({len(metas)} CSVs present)\n")
            continue
        per_device[dev] = cells
        print(f"\n# ===== {dev} ({len(metas)} CSVs, {len(cells)} cells) =====")
        print_cells(cells, dev)
        print_sweeps(cells, dev)
    if per_device:
        print(f"\n**Figure:** `{make_figure(per_device, fig_dir)}`")


if __name__ == "__main__":
    main()
