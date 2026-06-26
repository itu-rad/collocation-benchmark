#!/usr/bin/env python3
"""Zero-copy analysis for the framework-overhead microbenchmark.

The framework moves data between stages by passing the Query object by reference
(zero-copy). The ``copy`` arm replaces each stage with one that deep-copies the
payload every hop -- the counterfactual for a naive serializing framework. This
script shows reference passing is O(1) in payload size while deep-copy is
O(payload).

Reads the depth-10 payload sweep (tracing OFF) from ``results/``. Stage-duration
is measured over stages >= 1 (stage 0 is the payload injector, not a copy).

    python analyze_payload_results.py [--results-dir DIR] [--fig]

``--fig`` writes payload_zero_copy.{png,pdf} (needs matplotlib; the numeric path
runs without it).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noop_lib as nl  # noqa: E402

SIZES = [0, 1024, 1048576, 10485760]
SIZE_LABEL = {0: "0", 1024: "1 KiB", 1048576: "1 MiB", 10485760: "10 MiB"}


def collect(runs):
    """Return {mode: {size: summary}} of stage self-duration (us)."""
    out = {"ref": {}, "copy": {}}
    for mode in ("ref", "copy"):
        for size in SIZES:
            sel = nl.select(runs, depth=10, size=size, mode=mode, trace=0)
            dur = nl.pool_stage_dur(sel, min_idx=1)
            if dur:
                out[mode][size] = nl.summarize(dur, nl.NS_PER_US)
    return out


def print_table(data):
    print("\n## Zero-copy: per-stage duration vs payload (depth 10, tracing OFF)\n")
    print("| payload | ref (us) | ref 95% CI | copy (us) | copy 95% CI | copy/ref |")
    print("|--------:|---------:|:----------:|----------:|:-----------:|---------:|")
    for size in SIZES:
        r = data["ref"].get(size)
        c = data["copy"].get(size)
        if not r and not c:
            continue
        r_s = f"{r['median']:.2f}" if r else "—"
        r_ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]" if r else "—"
        c_s = f"{c['median']:.2f}" if c else "—"
        c_ci = f"[{c['ci_lo']:.1f}, {c['ci_hi']:.1f}]" if c else "—"
        ratio = f"{c['median'] / r['median']:.1f}x" if (r and c and r['median']) else "—"
        print(f"| {SIZE_LABEL[size]} | {r_s} | {r_ci} | {c_s} | {c_ci} | {ratio} |")

    # Fit copy-arm cost vs bytes (exclude size 0, which does no copy).
    cx = [s for s in SIZES if s > 0 and s in data["copy"]]
    if len(cx) >= 2:
        cy = [data["copy"][s]["median"] for s in cx]   # us
        slope, intercept = nl.ols_slope(cx, cy)
        print(f"\n**copy** cost vs payload: {slope * 1e6:.3f} us/MB "
              f"(intercept {intercept:.2f} us) -> grows with payload (O(payload)).")
    rx = [s for s in SIZES if s in data["ref"]]
    if len(rx) >= 2:
        ry = [data["ref"][s]["median"] for s in rx]
        slope, _ = nl.ols_slope(rx, ry)
        print(f"**ref** cost vs payload: {slope * 1e6:.3f} us/MB "
              f"-> flat (O(1) in payload size).")


def make_figure(data, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib unavailable; skipping figure.", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for mode, marker in (("ref", "o"), ("copy", "s")):
        xs = [s for s in SIZES if s in data[mode]]
        ys = [data[mode][s]["median"] for s in xs]
        lo = [data[mode][s]["median"] - data[mode][s]["ci_lo"] for s in xs]
        hi = [data[mode][s]["ci_hi"] - data[mode][s]["median"] for s in xs]
        # plot size 0 at a small positive x so it shows on the log axis
        px = [s if s > 0 else 256 for s in xs]
        ax.errorbar(px, ys, yerr=[lo, hi], marker=marker, capsize=3, label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("payload size (bytes)")
    ax.set_ylabel("per-stage duration (us)")
    ax.set_title("Reference passing is O(1); deep-copy is O(payload)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"payload_zero_copy.{ext}")
        fig.savefig(path, dpi=150)
        print(f"[fig] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=nl.default_results_dir())
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()

    runs = nl.load_matrix(args.results_dir)
    if not runs:
        sys.exit(f"No matrix CSVs found in {args.results_dir}")
    data = collect(runs)
    print_table(data)
    if args.fig:
        make_figure(data, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
