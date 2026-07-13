#!/usr/bin/env python3
"""Emit the paper's Table 2 (modularity overhead) from the same pooled stats the
Markdown analyzers use, so paper and write-up never disagree.

    python generate_latex_results.py [--results-dir DIR] --device cuda|mps > table2.tex
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modularity_lib as ml  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=ml.default_results_dir())
    ap.add_argument("--device", required=True)
    ap.add_argument("--warmup", type=int, default=200,
                    help="steps/run dropped so only steady-state steps count")
    args = ap.parse_args()

    metas = ml.load_matrix(args.results_dir, device=args.device)
    if not metas:
        sys.exit(f"No mod_*_d{args.device}_*.csv in {args.results_dir}")
    w = args.warmup
    base = ml.steps_by_run(ml.select(metas, impl="baseline"), ml.parse_baseline_steps, warmup=w)
    off = ml.steps_by_run(ml.select(metas, impl="choreo", trace=0), ml.parse_choreo_train_steps, warmup=w)
    on = ml.steps_by_run(ml.select(metas, impl="choreo", trace=1), ml.parse_choreo_train_steps, warmup=w)

    sb = ml.summarize(base, ml.NS_PER_MS)
    so = ml.summarize(off, ml.NS_PER_MS)
    # Statistic of record: paired across-run difference (interleaved arms);
    # falls back to the unpaired hierarchical CI if run ids don't align.
    o_off = ml.paired_overhead_ci(base, off)
    if o_off is None:
        o_off = ml.overhead_ratio_ci(base, off)
        o_off.update(d_ns=o_off["abs_ns"], d_lo=o_off["abs_lo"],
                     d_hi=o_off["abs_hi"], d_runs_ns=[], paired=False)
    else:
        o_off["paired"] = True

    if o_off["within_noise"]:
        verdict = "indistinguishable from zero"
    elif o_off["ratio_hi"] <= 0:
        verdict = "no measurable overhead"
    elif o_off["ratio_hi"] < 0.01:  # significant but < 1% of step
        verdict = "negligible"
    else:
        verdict = "significant"
    print(f"% --- Table 2: modularity overhead ({args.device}, EfficientNetV2-S "
          f"batch 8, tracing off, steady state) ---")
    print(f"% per-run medians (ms) — monolith: "
          + ", ".join(f"{v:.3f}" for v in sb["run_medians"])
          + " | Choreo: " + ", ".join(f"{v:.3f}" for v in so["run_medians"]))
    if o_off.get("paired") and o_off.get("d_runs_ns"):
        print("% paired per-run differences (us): "
              + ", ".join(f"{d / ml.NS_PER_US:+.0f}" for d in o_off["d_runs_ns"]))
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Wrapping a real EfficientNetV2-S Imagenette training step in "
          "Choreo's graph/queue/thread structure adds a negligible, fixed per-step "
          "overhead vs.\\ a hand-written monolith. Per-step training latency at "
          "steady state, median over $R$ runs with a hierarchical bootstrap "
          "\\SI{95}{\\percent} CI (runs resampled first, then steps); overhead is "
          f"the paired across-run difference (interleaved arms); {args.device}. "
          "Raw per-run values in the artifact.}")
    print("\\label{tab:modularity}")
    print("\\begin{tabular}{lrr}\n\\toprule")
    print("& Monolith & Choreo \\\\\n\\midrule")
    print(f"Per-step latency (\\si{{\\milli\\second}}) & {sb['median']:.3f} & {so['median']:.3f} \\\\")
    print(f"\\quad 95\\% CI & [{sb['ci_lo']:.3f}, {sb['ci_hi']:.3f}] & "
          f"[{so['ci_lo']:.3f}, {so['ci_hi']:.3f}] \\\\")
    print("\\midrule")
    print(f"\\multicolumn{{3}}{{l}}{{Overhead (paired): "
          f"{o_off['d_ns'] / ml.NS_PER_US:+.1f}\\,\\si{{\\micro\\second}} "
          f"[{o_off['d_lo'] / ml.NS_PER_US:+.1f}, {o_off['d_hi'] / ml.NS_PER_US:+.1f}], "
          f"{o_off['ratio'] * 100:+.2f}\\% "
          f"[{o_off['ratio_lo'] * 100:+.2f}, {o_off['ratio_hi'] * 100:+.2f}] "
          f"--- {verdict}}} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")
    if on:
        son = ml.summarize(on, ml.NS_PER_MS)
        print(f"% tracing-on per-step median: {son['median']:.3f} ms "
              f"(N={son['n']}; per-run: "
              + ", ".join(f"{v:.3f}" for v in son["run_medians"])
              + "); overhead-in-context in true_overhead_analysis.py")


if __name__ == "__main__":
    main()
