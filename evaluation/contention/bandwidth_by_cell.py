#!/usr/bin/env python3
"""Per-engine DRAM bandwidth per collocation cell, windowed to the cell's own run.

Windowing is not optional. The bandwidth trace is wall-clock and, until the
sampler learned to die on SIGTERM, an orphaned sampler kept writing into its
cell's file through every later cell -- a 1100s cell held 29737 rows instead of
~2200, and its per-engine medians described whichever workload ran next. Even
with that fixed, the sampler starts before the workload and the model-load phase
is not the measurement.

The window comes from the cell's own foreground trace: first to last per-query
row. Rows outside it are dropped, and a cell whose trace does not overlap its
window at all is reported rather than silently averaged.

    python evaluation/contention/bandwidth_by_cell.py --machine m3pro
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINES = ("cpu", "gpu", "ane", "other")


def run_window(fg_csv):
    """First and last per-query timestamp in a foreground trace."""
    ts = []
    with open(fg_csv) as f:
        for row in csv.reader(f):
            if len(row) > 5 and row[2].strip().startswith("pipeline - ") \
               and row[3].strip() == "run":
                ts.append(float(row[0]))
    return (min(ts), max(ts)) if ts else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default="m3pro")
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--drop-first", type=int, default=1, help="warm-up repetitions to skip")
    args = ap.parse_args()

    root = os.path.join(args.results_dir, args.machine)
    per = defaultdict(lambda: defaultdict(list))
    skipped = []
    for bw in sorted(glob.glob(os.path.join(root, "e5_*_bandwidth.csv"))):
        m = re.match(rf"^e5_(.+?)_{args.machine}_r(\d+)_bandwidth\.csv$", os.path.basename(bw))
        if not m:
            continue
        cell, run = m.group(1), int(m.group(2))
        if run <= args.drop_first:
            continue
        fg = glob.glob(os.path.join(root, f"e5_{cell}_{args.machine}_r{run}_*serve*.csv"))
        if not fg:
            skipped.append((cell, run, "no foreground trace"))
            continue
        win = run_window(fg[0])
        if not win:
            skipped.append((cell, run, "no per-query rows"))
            continue
        lo, hi = win
        rows = [r for r in csv.DictReader(open(bw))
                if lo <= float(r["timestamp"]) <= hi]
        if len(rows) < 10:
            skipped.append((cell, run, f"only {len(rows)} rows inside the run window"))
            continue
        f = lambda k: st.median((float(x[f"{k}_rd"]) + float(x[f"{k}_wr"]))
                                / float(x["dt_s"]) / 1e9 for x in rows)
        for e in ENGINES:
            per[cell][e].append(f(e))
        per[cell]["total"].append(st.median(float(x["total_gbps"]) for x in rows))
        per[cell]["_rows"].append(len(rows))

    print(f"# per-engine DRAM bandwidth, {args.machine} "
          f"(GB/s, median within each cell's run window)\n")
    print(f"| cell | cpu | gpu | ane | other | total | samples in window |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for cell in sorted(per):
        v = per[cell]
        print(f"| {cell} | " + " | ".join(f"{st.median(v[e]):.2f}" for e in ENGINES)
              + f" | {st.median(v['total']):.2f} | {int(st.median(v['_rows']))} |")
    if skipped:
        print("\nskipped:")
        for cell, run, why in skipped:
            print(f"  {cell} r{run}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
