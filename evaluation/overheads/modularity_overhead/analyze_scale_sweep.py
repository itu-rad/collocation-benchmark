#!/usr/bin/env python3
"""Multi-scale modularity-overhead result (E2 expansion): does Choreo's per-step
overhead stay a roughly FIXED cost as the workload gets heavier, so its RELATIVE
size collapses toward zero?

For every scale cell (model, batch, device) in the results dir this runs the same
paired baseline-vs-Choreo comparison as analyze_operational_overhead.py, then
lines the cells up by baseline step-time to show the amortization curve:

  * core wrapper   = Choreo(tracing off) − baseline   (queue/thread dispatch)
  * core + tracing = Choreo(tracing on)  − baseline   (+ one async MLflow span)

The absolute µs overhead should stay ~flat across cells while the % of the step
falls ~1/step_time. The worst realistic case (small model, batch 1) is the
ceiling; every heavier cell is cheaper.

    python analyze_scale_sweep.py [--results-dir DIR] [--device cuda|mps]
                                  [--warmup 200] [--latex] [--fig out.png]

Reuses modularity_lib for all parsing and statistics; matplotlib is imported only
when --fig is given (the other analyzers have no plotting dependency).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modularity_lib as ml  # noqa: E402

# Friendly display names for the short filename tags (canonical = legacy anchor).
MODEL_DISPLAY = {
    "canonical": "EfficientNetV2-S",
    "effv2s": "EfficientNetV2-S",
    "effv2m": "EfficientNetV2-M",
    "effv2l": "EfficientNetV2-L",
    "convnextl": "ConvNeXt-L",
}


def _overhead(base, other):
    """Paired overhead of `other` vs `base` (falls back to unpaired if run ids
    don't align). Returns dict with abs_ns/abs_lo/abs_hi, ratio/ratio_lo/ratio_hi,
    within_noise, paired(bool) — normalized across both code paths."""
    p = ml.paired_overhead_ci(base, other)
    if p is not None:
        return {"abs_ns": p["d_ns"], "abs_lo": p["d_lo"], "abs_hi": p["d_hi"],
                "ratio": p["ratio"], "ratio_lo": p["ratio_lo"],
                "ratio_hi": p["ratio_hi"], "within_noise": p["within_noise"],
                "paired": True}
    o = ml.overhead_ratio_ci(base, other)
    return {"abs_ns": o["abs_ns"], "abs_lo": o["abs_lo"], "abs_hi": o["abs_hi"],
            "ratio": o["ratio"], "ratio_lo": o["ratio_lo"],
            "ratio_hi": o["ratio_hi"], "within_noise": o["within_noise"],
            "paired": False}


def compute_rows(results_dir, device=None, warmup=200):
    """One row per (model, batch, device) cell with step-time + core/total overhead."""
    metas = ml.load_matrix(results_dir, device=device)
    rows = []
    for (model, batch, dev), cell_metas in ml.cells(metas):
        base = ml.steps_by_run(ml.select(cell_metas, impl="baseline"),
                               ml.parse_baseline_steps, warmup=warmup)
        off = ml.steps_by_run(ml.select(cell_metas, impl="choreo", trace=0),
                              ml.parse_choreo_train_steps, warmup=warmup)
        on = ml.steps_by_run(ml.select(cell_metas, impl="choreo", trace=1),
                             ml.parse_choreo_train_steps, warmup=warmup)
        if not base or not off:
            continue  # need at least baseline + core arm to place a point
        b = ml.summarize(base, ml.NS_PER_MS)
        row = {
            "model": model, "batch": batch, "dev": dev,
            "display": MODEL_DISPLAY.get(model, model),
            "n_base": b["n"], "step_ms": b["median"],
            "core": _overhead(base, off),
            "total": _overhead(base, on) if on else None,
        }
        rows.append(row)
    # Order by device, then by step-time (the amortization x-axis).
    rows.sort(key=lambda r: (r["dev"], r["step_ms"]))
    return rows


def print_table(rows):
    print("\n## Multi-scale modularity overhead — overhead vs. step-time\n")
    print("| device | model | batch | N | step (ms) | core µs | core % | "
          "core+trace µs | core+trace % | within noise? |")
    print("|---|---|--:|--:|---------:|------:|------:|------:|------:|:--:|")
    for r in rows:
        c = r["core"]
        t = r["total"]
        core_us = c["abs_ns"] / ml.NS_PER_US
        tot_us = (t["abs_ns"] / ml.NS_PER_US) if t else float("nan")
        tot_pct = f"{t['ratio'] * 100:+.2f}%" if t else "—"
        tot_us_s = f"{tot_us:.0f}" if t else "—"
        noise = "YES" if c["within_noise"] else "no"
        print(f"| {r['dev']} | {r['display']} | {r['batch']} | {r['n_base']} | "
              f"{r['step_ms']:.3f} | {core_us:.0f} | {c['ratio'] * 100:+.2f}% | "
              f"{tot_us_s} | {tot_pct} | {noise} |")
    print("\n*core = Choreo(tracing off) − baseline (queue/thread dispatch); "
          "core+trace = Choreo(tracing on) − baseline (+ one async MLflow span). "
          "Overhead is the paired per-run difference of medians (see "
          "analyze_operational_overhead.py). Read the µs columns as the ~fixed "
          "framework cost and the % columns as its share of the step — the latter "
          "shrinks as the step grows.*")


def print_curve(rows):
    """The amortization-curve data: step-time vs relative overhead, per series."""
    print("\n### Collapse-curve data (step_ms, core_pct, total_pct)\n")
    print("| device | model | batch | step (ms) | core % | core+trace % |")
    print("|---|---|--:|---------:|------:|------:|")
    for r in rows:
        t = r["total"]
        tp = f"{t['ratio'] * 100:.3f}" if t else "—"
        print(f"| {r['dev']} | {r['display']} | {r['batch']} | {r['step_ms']:.3f} | "
              f"{r['core']['ratio'] * 100:.3f} | {tp} |")
    fixed = _fixed_cost_us(rows)
    if fixed is not None:
        print(f"\n*Fixed-cost fit: median core overhead ≈ {fixed:.0f} µs across "
              f"cells ⇒ predicted core %% ≈ {fixed:.0f}µs / (step_ms · 10). "
              f"If the points track this 1/x curve, the overhead is a fixed "
              f"per-step cost and its relative size is fully explained by step-time.*")


def _fixed_cost_us(rows):
    """Median core absolute overhead (µs) — the fixed-cost estimate for the 1/x
    overlay. Uses only cells whose core overhead is positive (a wrapper can only
    add work; non-positive cells are noise-dominated)."""
    import numpy as np
    vals = [r["core"]["abs_ns"] / ml.NS_PER_US for r in rows
            if r["core"]["abs_ns"] > 0]
    return float(np.median(vals)) if vals else None


def make_fig(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4.5))
    devs = sorted({r["dev"] for r in rows})
    markers = {d: m for d, m in zip(devs, ["o", "s", "^", "D"])}
    for dev in devs:
        dr = [r for r in rows if r["dev"] == dev]
        xs = [r["step_ms"] for r in dr]
        ax.plot(xs, [r["core"]["ratio"] * 100 for r in dr], markers[dev] + "-",
                color="tab:blue", label=f"core wrapper ({dev})", alpha=0.85)
        tr = [r for r in dr if r["total"]]
        if tr:
            ax.plot([r["step_ms"] for r in tr],
                    [r["total"]["ratio"] * 100 for r in tr], markers[dev] + "--",
                    color="tab:red", label=f"core + tracing ({dev})", alpha=0.85)

    fixed = _fixed_cost_us(rows)
    if fixed:
        xs = np.logspace(np.log10(min(r["step_ms"] for r in rows)),
                         np.log10(max(r["step_ms"] for r in rows)), 100)
        ax.plot(xs, fixed / (xs * 10.0), ":", color="gray",
                label=f"fixed-cost 1/x ({fixed:.0f} µs)")

    # Annotate the worst realistic case (largest relative core overhead).
    worst = max(rows, key=lambda r: r["core"]["ratio"])
    ax.annotate(f"worst case\n{worst['display']} b{worst['batch']}\n"
                f"{worst['core']['ratio'] * 100:+.2f}%",
                xy=(worst["step_ms"], worst["core"]["ratio"] * 100),
                xytext=(1.3, 0.7), textcoords="offset fontsize",
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xscale("log")
    ax.set_xlabel("baseline step time (ms, log)")
    ax.set_ylabel("Choreo overhead (% of step)")
    ax.set_title("Modularity overhead amortizes as the step grows")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n[fig] wrote {out_path}")


def print_latex(rows):
    print("\n% --- LaTeX: multi-scale modularity overhead ---")
    print("\\begin{tabular}{llrrrrr}")
    print("\\toprule")
    print("device & model & batch & step (ms) & core (\\si{\\micro s}) & "
          "core \\% & core+trace \\% \\\\")
    print("\\midrule")
    for r in rows:
        t = r["total"]
        tp = f"{t['ratio'] * 100:+.2f}" if t else "--"
        print(f"{r['dev']} & {r['display']} & {r['batch']} & {r['step_ms']:.2f} & "
              f"{r['core']['abs_ns'] / ml.NS_PER_US:.0f} & "
              f"{r['core']['ratio'] * 100:+.2f} & {tp} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=ml.default_results_dir())
    ap.add_argument("--device", default=None,
                    help="restrict to one device; default pools both onto the curve")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--latex", action="store_true", help="also emit a LaTeX table")
    ap.add_argument("--fig", default=None, help="write the amortization figure here")
    args = ap.parse_args()

    rows = compute_rows(args.results_dir, device=args.device, warmup=args.warmup)
    if not rows:
        sys.exit(f"No usable scale cells in {args.results_dir} "
                 f"(need baseline + Choreo(off) per cell)")
    print_table(rows)
    print_curve(rows)
    if args.latex:
        print_latex(rows)
    if args.fig:
        make_fig(rows, args.fig)


if __name__ == "__main__":
    main()
