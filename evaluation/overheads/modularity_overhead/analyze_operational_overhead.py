#!/usr/bin/env python3
"""Headline modularity-overhead result: is wrapping a real GPU training workload
in Choreo's graph/queue/thread structure measurably slower than a hand-written
monolith?

Per-step TRAINING-STAGE latency (monotonic perf column) over R runs with the
first 200 steps/run dropped as warmup. Statistics follow the paper's rules:
per-arm medians carry a HIERARCHICAL bootstrap CI (resample runs, then steps)
with the raw per-run medians printed beside it; the overhead of record is the
PAIRED across-run difference (the arms were interleaved run-by-run), with an
explicit "within noise" verdict (CI contains 0).

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
    base = ml.steps_by_run(ml.select(metas, impl="baseline"), ml.parse_baseline_steps, warmup=w)
    off = ml.steps_by_run(ml.select(metas, impl="choreo", trace=0), ml.parse_choreo_train_steps, warmup=w)
    on = ml.steps_by_run(ml.select(metas, impl="choreo", trace=1), ml.parse_choreo_train_steps, warmup=w)

    print(f"\n## Modularity overhead -- per-step training latency ({args.device})\n")
    print("| arm | N | median (ms) | 95% CI (ms, hier.) | p95 (ms) | per-run medians (ms) |")
    print("|---|--:|------------:|:------------------:|---------:|:---|")
    for name, by_run in (("baseline (monolith)", base),
                         ("Choreo (tracing off)", off),
                         ("Choreo (tracing on)", on)):
        if not by_run:
            print(f"| {name} | 0 | — | — | — | — |")
            continue
        s = ml.summarize(by_run, ml.NS_PER_MS)
        p95 = f"{s['p95']:.3f}" if s['p95'] == s['p95'] else "n/a"
        rm = " / ".join(f"{v:.3f}" for v in s["run_medians"])
        print(f"| {name} | {s['n']} | {s['median']:.3f} | "
              f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | {p95} | {rm} |")

    print("\n### Overhead vs the hand-written baseline "
          "(paired across-run difference — arms interleaved run-by-run)\n")
    print("| comparison | abs overhead (µs) | 95% CI (µs) | ratio | 95% CI | "
          "within noise? | per-pair d (µs) |")
    print("|---|---:|:---:|---:|:---:|:---:|:---|")
    for name, by_run in (("Choreo(off) − baseline", off), ("Choreo(on) − baseline", on)):
        if not (base and by_run):
            continue
        p = ml.paired_overhead_ci(base, by_run)
        if p is None:
            o = ml.overhead_ratio_ci(base, by_run)
            print(f"| {name} (UNPAIRED — run ids don't align) | "
                  f"{o['abs_ns'] / ml.NS_PER_US:.1f} | "
                  f"[{o['abs_lo'] / ml.NS_PER_US:.1f}, {o['abs_hi'] / ml.NS_PER_US:.1f}] | "
                  f"{o['ratio'] * 100:+.2f}% | [{o['ratio_lo'] * 100:+.2f}%, "
                  f"{o['ratio_hi'] * 100:+.2f}%] | {'YES' if o['within_noise'] else 'no'} | — |")
            continue
        ds = " / ".join(f"{d / ml.NS_PER_US:+.0f}" for d in p["d_runs_ns"])
        print(f"| {name} | {p['d_ns'] / ml.NS_PER_US:.1f} | "
              f"[{p['d_lo'] / ml.NS_PER_US:.1f}, {p['d_hi'] / ml.NS_PER_US:.1f}] | "
              f"{p['ratio'] * 100:+.2f}% | [{p['ratio_lo'] * 100:+.2f}%, {p['ratio_hi'] * 100:+.2f}%] | "
              f"{'YES' if p['within_noise'] else 'no'} | {ds} |")
    print("\n*Overhead = mean paired per-run difference of medians (pairs share a "
          "run slot in the interleaved schedule); CI from a pair-resampling "
          "bootstrap with within-run resampling. Within noise = the CI contains "
          "0 ⇒ Choreo is statistically indistinguishable from the monolith. A "
          "non-positive overhead means no measurable per-step cost, not a "
          "deliberate speedup. Read absolute overhead (µs) as the fixed "
          "framework cost.*")


if __name__ == "__main__":
    main()
