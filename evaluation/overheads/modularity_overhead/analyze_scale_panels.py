#!/usr/bin/env python3
"""Two-axis view of the modularity scale sweep, with per-run mean/std statistics.

Splits the sweep into the two clean 1-D studies it actually is:

  (batch) overhead vs BATCH  — EfficientNetV2-S, batch 1..64  (model fixed)
  (model) overhead vs MODEL  — S / M / L at batch 8            (batch fixed)

For each cell it reports, over the R runs (arms interleaved run-by-run):
  * step time (ms)            — mean +/- std of the per-run baseline median step
  * core overhead (us)        — mean +/- std of the per-run paired diff
                                 median(choreo tracing-off) - median(baseline)
  * core+trace overhead (us)  — same, for choreo tracing-on
  * core %, core+trace %      — the mean absolute overhead as a % of the mean step

ConvNeXt-L is excluded from the model axis (224 px vs the 384-480 px EfficientNets:
it conflates architecture/resolution with size and is not a monotonic step-time
point). Reuses modularity_lib for parsing + the paired hierarchical bootstrap.

    python analyze_scale_panels.py --results-dir <dir> --device cuda [--fig out.png] [--latex]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modularity_lib as ml  # noqa: E402

MODEL_DISPLAY = {"effv2s": "EfficientNetV2-S", "effv2m": "EfficientNetV2-M",
                 "effv2l": "EfficientNetV2-L"}  # convnextl excluded (resolution-confounded)
SWEEP_MODELS = set(MODEL_DISPLAY)


def _runs_us(paired, unpaired):
    """Per-run overhead diffs in us (from the paired stat), or a single unpaired
    point if <2 runs. Returns (mean_us, std_us)."""
    if paired is not None:
        v = np.asarray(paired["d_runs_ns"], dtype=float) / ml.NS_PER_US
        return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else float("nan"))
    return unpaired["abs_ns"] / ml.NS_PER_US, float("nan")


def cell_overhead(cell_metas, warmup):
    base = ml.steps_by_run(ml.select(cell_metas, impl="baseline"), ml.parse_baseline_steps, warmup=warmup)
    off = ml.steps_by_run(ml.select(cell_metas, impl="choreo", trace=0), ml.parse_choreo_train_steps, warmup=warmup)
    on = ml.steps_by_run(ml.select(cell_metas, impl="choreo", trace=1), ml.parse_choreo_train_steps, warmup=warmup)
    if not base or not off:
        return None
    # per-run baseline median step (ms)
    step_ms = np.array([np.median(base[r]) for r in sorted(base)], dtype=float) / ml.NS_PER_MS
    step_mean = float(step_ms.mean())
    step_std = float(step_ms.std(ddof=1)) if step_ms.size > 1 else float("nan")

    core_mean, core_std = _runs_us(ml.paired_overhead_ci(base, off), ml.overhead_ratio_ci(base, off))
    if on:
        total_mean, total_std = _runs_us(ml.paired_overhead_ci(base, on), ml.overhead_ratio_ci(base, on))
    else:
        total_mean = total_std = float("nan")

    step_us = step_mean * 1000.0
    return {"step_mean": step_mean, "step_std": step_std,
            "core_mean": core_mean, "core_std": core_std,
            "total_mean": total_mean, "total_std": total_std,
            "core_pct": core_mean / step_us * 100,
            "total_pct": total_mean / step_us * 100}


def collect(results_dir, device, warmup):
    metas = ml.load_matrix(results_dir, device=device)
    rows = {}
    for (model, batch, dev), cm in ml.cells(metas):
        if model not in SWEEP_MODELS:
            continue
        o = cell_overhead(cm, warmup)
        if o:
            rows[(model, batch)] = o
    batch_sweep = sorted(((b, o) for (m, b), o in rows.items() if m == "effv2s"), key=lambda x: x[0])
    model_sweep = sorted(((m, o) for (m, b), o in rows.items() if b == 8), key=lambda x: x[1]["step_mean"])
    return batch_sweep, model_sweep


def print_latex(batch_sweep, model_sweep):
    def row(model, batch, o):
        return (f"{model} & {batch} & "
                f"${o['step_mean']:.1f} \\pm {o['step_std']:.1f}$ & "
                f"${o['core_mean']:.0f} \\pm {o['core_std']:.0f}$ & "
                f"${o['total_mean']:.0f} \\pm {o['total_std']:.0f}$ & "
                f"{o['core_pct']:+.2f} & {o['total_pct']:+.2f} \\\\")

    print("\n% --- modularity scale sweep (GB10, R=5); step/overhead as mean $\\pm$ std over runs ---")
    print("\\begin{tabular}{llrrrrr}")
    print("\\toprule")
    print("model & batch & step (ms) & core (\\si{\\micro s}) & core+trace (\\si{\\micro s}) "
          "& core \\% & core+trace \\% \\\\")
    print("\\midrule")
    print("\\multicolumn{7}{l}{\\emph{Batch sweep — EfficientNetV2-S}} \\\\")
    for b, o in batch_sweep:
        print(row("EfficientNetV2-S", b, o))
    print("\\midrule")
    print("\\multicolumn{7}{l}{\\emph{Model sweep — batch 8}} \\\\")
    for m, o in model_sweep:
        if m == "effv2s":
            continue  # already the b8 row in the batch sweep
        print(row(MODEL_DISPLAY[m], 8, o))
    print("\\bottomrule")
    print("\\end{tabular}")


def print_markdown(batch_sweep, model_sweep):
    def show(rows, header):
        print(f"\n## {header}\n")
        print("| cell | step ms (µ±σ) | core µs (µ±σ) | core+trace µs (µ±σ) | core % | core+trace % |")
        print("|---|--:|--:|--:|--:|--:|")
        for lab, o in rows:
            print(f"| {lab} | {o['step_mean']:.1f} ± {o['step_std']:.1f} | "
                  f"{o['core_mean']:.0f} ± {o['core_std']:.0f} | "
                  f"{o['total_mean']:.0f} ± {o['total_std']:.0f} | "
                  f"{o['core_pct']:+.2f} | {o['total_pct']:+.2f} |")
    show([(f"EffNetV2-S b{b}", o) for b, o in batch_sweep], "Batch sweep (EfficientNetV2-S)")
    show([(MODEL_DISPLAY[m] + " b8", o) for m, o in model_sweep if m != "effv2s"], "Model sweep (batch 8)")


def make_fig(batch_sweep, model_sweep, out, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bx = [b for b, _ in batch_sweep]
    mx = list(range(len(model_sweep)))
    mticklabels = [f"{MODEL_DISPLAY[m]}\n{o['step_mean']:.0f} ms" for m, o in model_sweep]

    def panel(ax, xs, core, total, *, title, xlabel, ylabel, log2=False, xticks=None, xticklabels=None):
        ax.plot(xs, core, "o-", color="tab:blue", label="core wrapper")
        ax.plot(xs, total, "s--", color="tab:red", label="core + tracing")
        if log2:
            ax.set_xscale("log", base=2)
        if xticks is not None:
            ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, fontsize=8)
            elif log2:
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.6))
    panel(ax[0][0], bx, [o["core_pct"] for _, o in batch_sweep], [o["total_pct"] for _, o in batch_sweep],
          title="Relative overhead vs batch (model fixed)", xlabel="batch size (EfficientNetV2-S)",
          ylabel=f"overhead (% of step) — {device}", log2=True, xticks=bx)
    panel(ax[0][1], mx, [o["core_pct"] for _, o in model_sweep], [o["total_pct"] for _, o in model_sweep],
          title="Relative overhead vs model (batch 8)", xlabel="model (ordered by step-time)",
          ylabel="overhead (% of step)", xticks=mx, xticklabels=mticklabels)
    panel(ax[1][0], bx, [o["core_mean"] for _, o in batch_sweep], [o["total_mean"] for _, o in batch_sweep],
          title="Absolute overhead vs batch", xlabel="batch size (EfficientNetV2-S)",
          ylabel="overhead (µs / step)", log2=True, xticks=bx)
    panel(ax[1][1], mx, [o["core_mean"] for _, o in model_sweep], [o["total_mean"] for _, o in model_sweep],
          title="Absolute overhead vs model", xlabel="model (ordered by step-time)",
          ylabel="overhead (µs / step)", xticks=mx, xticklabels=mticklabels)

    fig.suptitle("Modularity overhead: relative (top) amortizes as the step grows; "
                 "absolute (bottom) is a ~fixed per-step cost", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n[fig] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=ml.default_results_dir())
    ap.add_argument("--device", required=True)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    batch_sweep, model_sweep = collect(args.results_dir, args.device, args.warmup)
    if not batch_sweep and not model_sweep:
        sys.exit("no sweep cells found")
    print_markdown(batch_sweep, model_sweep)
    if args.latex:
        print_latex(batch_sweep, model_sweep)
    if args.fig:
        make_fig(batch_sweep, model_sweep, args.fig, args.device)


if __name__ == "__main__":
    main()
