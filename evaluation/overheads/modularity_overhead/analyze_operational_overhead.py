#!/usr/bin/env python3
"""Headline modularity-overhead result: is wrapping a real GPU training workload
in Choreo's graph/queue/thread structure measurably slower than a hand-written
monolith?

Per-step TRAINING-STAGE latency (monotonic perf column), pooled over R runs with
the first 100 steps/run dropped as warmup. Reports baseline vs Choreo(tracing
off) vs Choreo(tracing on), the overhead as both an absolute cost (us) and a
ratio (%), each with a two-independent-sample bootstrap 95% CI, and an explicit
"within noise" verdict (ratio CI contains 0).

    python analyze_operational_overhead.py [--results-dir DIR] --device cuda|mps
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

    print(f"\n## Modularity overhead -- per-step training latency ({args.device})\n")
    print("| arm | N | median (ms) | 95% CI (ms) | p95 (ms) |")
    print("|---|--:|------------:|:-----------:|---------:|")
    for name, vec in (("baseline (monolith)", base),
                      ("Choreo (tracing off)", off),
                      ("Choreo (tracing on)", on)):
        if not vec:
            print(f"| {name} | 0 | — | — | — |")
            continue
        s = ml.summarize(vec, ml.NS_PER_MS)
        p95 = f"{s['p95']:.3f}" if s['p95'] == s['p95'] else "n/a"
        print(f"| {name} | {s['n']} | {s['median']:.3f} | "
              f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | {p95} |")

    print("\n### Overhead vs the hand-written baseline\n")
    print("| comparison | abs overhead (µs) | 95% CI (µs) | ratio | 95% CI | within noise? |")
    print("|---|---:|:---:|---:|:---:|:---:|")
    for name, vec in (("Choreo(off) − baseline", off), ("Choreo(on) − baseline", on)):
        if not (base and vec):
            continue
        o = ml.overhead_ratio_ci(base, vec)
        print(f"| {name} | {o['abs_ns'] / ml.NS_PER_US:.1f} | "
              f"[{o['abs_lo'] / ml.NS_PER_US:.1f}, {o['abs_hi'] / ml.NS_PER_US:.1f}] | "
              f"{o['ratio'] * 100:+.2f}% | [{o['ratio_lo'] * 100:+.2f}%, {o['ratio_hi'] * 100:+.2f}%] | "
              f"{'YES' if o['within_noise'] else 'no'} |")
    print("\n*Within noise = the ratio CI contains 0 ⇒ Choreo is statistically "
          "indistinguishable from the monolith. A non-positive overhead (upper CI "
          "≤ 0) means the framework adds no measurable per-step cost — Choreo "
          "matches or marginally beats the monolith — not that it is a deliberate "
          "speedup. Read absolute overhead (µs) as the fixed framework cost.*")


if __name__ == "__main__":
    main()
