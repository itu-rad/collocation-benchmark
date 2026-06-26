#!/usr/bin/env python3
"""Emit the paper's LaTeX tables for the framework-overhead microbenchmark.

Consumes the SAME pooled statistics as the Markdown analyzers (via noop_lib) so
the paper tables and the write-up can never disagree. Prints two tables to
stdout:

  * Table: depth scaling (core dispatch, tracing OFF) -- depth, L_q, per-stage.
  * Table: zero-copy payload sweep -- ref vs copy per-stage duration.

    python generate_latex_results.py [--results-dir DIR] > tables.tex
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noop_lib as nl  # noqa: E402

SIZES = [0, 1024, 1048576, 10485760]
SIZE_TEX = {0: "0", 1024: "\\SI{1}{\\kibi\\byte}",
            1048576: "\\SI{1}{\\mebi\\byte}", 10485760: "\\SI{10}{\\mebi\\byte}"}


def depth_table(runs):
    sel = nl.select(runs, trace=0, size=0, mode="ref")
    depths = sorted({r.meta["depth"] for r in sel})
    print("% --- Framework overhead: depth scaling (core dispatch, tracing off) ---")
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Per-stage framework overhead is flat in pipeline depth "
          "(no-op chains, tracing disabled, Apple~M2~Pro). Median over "
          "$R$ runs; \\SI{95}{\\percent} bootstrap CI in brackets.}")
    print("\\label{tab:noop-depth}")
    print("\\begin{tabular}{rrr}\n\\toprule")
    print("Depth & Per-query latency (\\si{\\milli\\second}) & "
          "Per-stage (\\si{\\micro\\second}) \\\\\n\\midrule")
    for d in depths:
        lat = nl.pool_latency(nl.select(sel, depth=d))
        if not lat:
            continue
        lq = nl.summarize(lat, nl.NS_PER_MS)
        od = nl.summarize([v / d for v in lat], nl.NS_PER_US)
        print(f"{d} & {lq['median']:.3f} & "
              f"{od['median']:.1f} [{od['ci_lo']:.1f}, {od['ci_hi']:.1f}] \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def payload_table(runs):
    print("% --- Framework overhead: zero-copy payload sweep (tracing off) ---")
    print("\\begin{table}[t]\n\\centering")
    print("\\caption{Reference passing is constant in payload size while "
          "deep-copy is linear (no-op chains, depth~10, tracing disabled). "
          "Per-stage duration, \\si{\\micro\\second}, median [95\\% CI].}")
    print("\\label{tab:noop-zerocopy}")
    print("\\begin{tabular}{lrr}\n\\toprule")
    print("Payload & Reference (\\si{\\micro\\second}) & "
          "Deep-copy (\\si{\\micro\\second}) \\\\\n\\midrule")
    for size in SIZES:
        r = nl.pool_stage_dur(nl.select(runs, trace=0, depth=10, size=size,
                                        mode="ref"), min_idx=1)
        c = nl.pool_stage_dur(nl.select(runs, trace=0, depth=10, size=size,
                                        mode="copy"), min_idx=1)
        rs = nl.summarize(r, nl.NS_PER_US) if r else None
        cs = nl.summarize(c, nl.NS_PER_US) if c else None
        r_txt = (f"{rs['median']:.1f} [{rs['ci_lo']:.1f}, {rs['ci_hi']:.1f}]"
                 if rs else "---")
        c_txt = (f"{cs['median']:.1f} [{cs['ci_lo']:.1f}, {cs['ci_hi']:.1f}]"
                 if cs else "---")
        print(f"{SIZE_TEX[size]} & {r_txt} & {c_txt} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=nl.default_results_dir())
    args = ap.parse_args()
    runs = nl.load_matrix(args.results_dir)
    if not runs:
        sys.exit(f"No matrix CSVs found in {args.results_dir}")
    depth_table(runs)
    payload_table(runs)


if __name__ == "__main__":
    main()
