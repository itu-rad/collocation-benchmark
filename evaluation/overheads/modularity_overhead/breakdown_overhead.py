#!/usr/bin/env python3
"""End-to-end breakdown for the modularity-overhead experiment.

Beyond the per-step training latency (see analyze_operational_overhead.py), the
framework also measures the data-loading stage separately and the full per-query
pipeline latency. This shows the framework's end-to-end visibility (data loading
is a first-class, separately-timed stage) and that end-to-end latency isn't
inflated. Descriptive — not folded into the "overhead" number.

NOTE: under the closed-loop serialized scheduler there is no cross-query overlap;
cross-query dataloader/training pipelining is a separate framework capability.

    python breakdown_overhead.py [--results-dir DIR] --device cuda|mps
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modularity_lib as ml  # noqa: E402


def _row(name, vec, unit=ml.NS_PER_MS):
    if not vec:
        return f"| {name} | 0 | — | — |"
    s = ml.summarize(vec, unit)
    return (f"| {name} | {s['n']} | {s['median']:.3f} | "
            f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] |")


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
    base = ml.select(metas, impl="baseline")
    off = ml.select(metas, impl="choreo", trace=0)

    print(f"\n## End-to-end breakdown ({args.device}, tracing off, median [95% CI] in ms)\n")
    print("| component | N | median (ms) | 95% CI (ms) |")
    print("|---|--:|------------:|:-----------:|")
    print(_row("baseline training step", ml.pool_steps(base, ml.parse_baseline_steps, warmup=w)))
    print(_row("Choreo training stage", ml.pool_steps(off, ml.parse_choreo_train_steps, warmup=w)))
    print(_row("Choreo dataloader stage", ml.pool_steps(off, ml.parse_choreo_load_steps, warmup=w)))
    print(_row("Choreo end-to-end per query", ml.pool_steps(off, ml.parse_pipeline_latency, warmup=w)))

    # Throughput over the steady-state pooled window (steps / total measured time).
    train = ml.pool_steps(off, ml.parse_choreo_train_steps, warmup=w)
    pipe = ml.pool_steps(off, ml.parse_pipeline_latency, warmup=w)
    if train and pipe:
        thr_train = 1e9 / (sum(train) / len(train))
        thr_pipe = 1e9 / (sum(pipe) / len(pipe))
        print(f"\nChoreo training-stage rate: {thr_train:.1f} steps/s; "
              f"end-to-end rate: {thr_pipe:.1f} queries/s "
              f"(end-to-end includes the dataloader stage, serialized).")


if __name__ == "__main__":
    main()
