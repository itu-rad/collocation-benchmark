#!/usr/bin/env python3
"""Depth-flatness analysis for the framework-overhead microbenchmark.

Shows that the framework's per-stage cost does NOT accumulate with pipeline
depth: per-query latency L_q is linear in depth (constant marginal per-stage
cost), so deep/complex graphs are free of measurement distortion.

Reads the matrix CSVs in ``results/`` (override with ``--results-dir``). With the
tracing-OFF arm present it reports CORE dispatch; with both arms it also reports
the per-stage cost the MLflow tracing layer adds.

    python analyze_noop_results.py [--results-dir DIR] [--arm off|on|both]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noop_lib as nl  # noqa: E402


def _depth_table(runs, arm_label):
    sel_arm = nl.select(runs, size=0, mode="ref")
    depths = sorted({r.meta["depth"] for r in sel_arm})
    print(f"\n## Depth sweep -- {arm_label} (size 0, mode ref)\n")
    print(f"| depth | N | L_q median (ms) | O(d)=L_q/d (us) | 95% CI (us) | "
          f"transition (us) | p95 O(d) (us) |")
    print("|------:|--:|----------------:|----------------:|:-----------:|"
          "----------------:|--------------:|")
    xs, lq_ns = [], []
    od_by_depth = {}
    for d in depths:
        sel = nl.select(sel_arm, depth=d)
        lat = nl.pool_latency(sel)
        if not lat:
            continue
        lq = nl.summarize(lat, nl.NS_PER_MS)
        od = nl.summarize([v / d for v in lat], nl.NS_PER_US)
        tr = nl.summarize(nl.pool_transition(sel), nl.NS_PER_US)
        od_by_depth[d] = od
        xs.append(d)
        lq_ns.append(nl.median(lat))
        tr_s = f"{tr['median']:.2f}" if tr['n'] else "—"
        p95_s = f"{od['p95']:.2f}" if od['p95'] == od['p95'] else "n/a"
        print(f"| {d} | {lq['n']} | {lq['median']:.4f} | {od['median']:.2f} | "
              f"[{od['ci_lo']:.1f}, {od['ci_hi']:.1f}] | {tr_s} | {p95_s} |")

    # Marginal per-stage cost = slope of L_q (ns) vs depth -> us/stage.
    slope_ns, intercept_ns = nl.ols_slope(xs, lq_ns)
    print(f"\n**Marginal per-stage cost** (slope of L_q vs depth): "
          f"{slope_ns / nl.NS_PER_US:.2f} us/stage  \n"
          f"**Fixed per-query overhead** (intercept): "
          f"{intercept_ns / nl.NS_PER_US:.2f} us")
    return od_by_depth


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=nl.default_results_dir())
    ap.add_argument("--arm", choices=["off", "on", "both"], default="both")
    args = ap.parse_args()

    runs = nl.load_matrix(args.results_dir)
    if not runs:
        sys.exit(f"No matrix CSVs found in {args.results_dir}")

    have = {r.meta["trace"] for r in runs}
    od_off = od_on = None
    if args.arm in ("off", "both") and 0 in have:
        od_off = _depth_table(nl.select(runs, trace=0), "tracing OFF (core dispatch)")
    if args.arm in ("on", "both") and 1 in have:
        od_on = _depth_table(nl.select(runs, trace=1), "tracing ON (instrumented)")

    if od_off and od_on:
        print("\n## Tracing-layer per-stage cost (ON − OFF)\n")
        print(f"| depth | O(d) off (us) | O(d) on (us) | tracing add (us) |")
        print("|------:|--------------:|-------------:|-----------------:|")
        for d in sorted(set(od_off) & set(od_on)):
            add = od_on[d]["median"] - od_off[d]["median"]
            print(f"| {d} | {od_off[d]['median']:.2f} | {od_on[d]['median']:.2f} "
                  f"| {add:.2f} |")


if __name__ == "__main__":
    main()
