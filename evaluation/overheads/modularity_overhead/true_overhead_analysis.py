#!/usr/bin/env python3
"""Overhead-in-context for the modularity-overhead experiment.

Expresses the framework's per-step cost as a fraction of the real GPU training
step, so the absolute µs reads as negligible. Two layers: core dispatch (tracing
off) and the MLflow tracing layer (on − off). Cross-references §exp-noop, which
establishes the per-step framework cost is fixed and step-size-independent.

    python true_overhead_analysis.py [--results-dir DIR] --device cuda|mps
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modularity_lib as ml  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=ml.default_results_dir())
    ap.add_argument("--device", required=True)
    ap.add_argument("--warmup", type=int, default=200,
                    help="steps/run dropped so only steady-state steps count")
    args = ap.parse_args()

    metas = ml.load_matrix(args.results_dir, device=args.device)
    if not metas:
        sys.exit(f"No mod_*_d{args.device}_*.csv in {args.results_dir}")

    w = args.warmup
    base = ml.pool_steps(ml.select(metas, impl="baseline"), ml.parse_baseline_steps, warmup=w)
    off = ml.pool_steps(ml.select(metas, impl="choreo", trace=0), ml.parse_choreo_train_steps, warmup=w)
    on = ml.pool_steps(ml.select(metas, impl="choreo", trace=1), ml.parse_choreo_train_steps, warmup=w)
    if not base or not off:
        sys.exit("Need baseline + Choreo(off) data.")

    step_ms = ml.summarize(base, ml.NS_PER_MS)["median"]
    print(f"\n## Overhead in context ({args.device})\n")
    print(f"Real GPU training step (baseline median): **{step_ms:.2f} ms**\n")
    print("| layer | per-step cost (µs) | as % of a real step |")
    print("|---|---:|---:|")

    core = ml.overhead_ratio_ci(base, off)
    core_us = core["abs_ns"] / ml.NS_PER_US
    print(f"| core dispatch (Choreo off − baseline) | {core_us:.1f} | "
          f"{core_us / 1e3 / step_ms * 100:+.3f}% |")
    if on:
        med_off = ml.summarize(off, ml.NS_PER_US)["median"]
        med_on = ml.summarize(on, ml.NS_PER_US)["median"]
        trace_us = med_on - med_off
        print(f"| MLflow tracing layer (on − off) | {trace_us:.1f} | "
              f"{trace_us / 1e3 / step_ms * 100:+.3f}% |")

    print("\n*The core framework cost is a fixed per-step quantity (see §exp-noop / "
          "the NoOp microbenchmark, which shows it is independent of step size); "
          "against a multi-ms GPU step it is a negligible fraction. The tracing-on "
          "figure uses the async_tracing radt branch (non-blocking span export) to a "
          "local store on the -p 0 path; an orchestrated RadT-server run may differ.*")


if __name__ == "__main__":
    main()
