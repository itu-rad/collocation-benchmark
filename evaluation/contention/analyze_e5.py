#!/usr/bin/env python3
"""Section 5.2 analyzer: per-pipeline attribution under each collocation type.

Reads what collect_e5.sh writes:

    results/<machine>/e5_<cell>_<machine>_r<N>_<pipeline>.csv

One file per PIPELINE per run, because each pipeline is its own process with its
own run -- that separation is the point, and it is what lets a number be
attributed to the pipeline that caused it.

Per-query latency comes from the ``pipeline - <split>`` rows (run start/end, one
pair per query). The bare ``pipeline`` rows are the process bookends, not
per-query, and using them silently yields two "latencies" per run.

The run is the unit of replication: quantiles are pooled per cell across runs and
the CI is a hierarchical (run-then-query) bootstrap, matching the overhead
analyzers. Repetition 1 is dropped as system warm-up unless --keep-first.

    python evaluation/contention/analyze_e5.py --machine gb10
    python evaluation/contention/analyze_e5.py --machine m3pro --fig-dir paper_assets
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FG_HINT = "rag_serve"          # the foreground pipeline's slug
DROP_RUNS = 1                  # repetition 1 is warm-up (see first-run-system-warmup)
BOOT = 2000


def parse_name(path):
    """e5_<cell>_<machine>_r<N>_<pipeline>.csv -> dict or None."""
    m = re.match(r"^e5_(.+?)_(m3pro|gb10)_r(\d+)_(.+)\.csv$", os.path.basename(path))
    if not m:
        return None
    return {"cell": m.group(1), "machine": m.group(2), "run": int(m.group(3)),
            "pipeline": m.group(4), "path": path,
            "role": "fg" if FG_HINT in m.group(4) else "bg"}


def query_latencies(path):
    """Per-query end-to-end seconds, from the per-query pipeline rows."""
    starts, out = {}, []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            level, phase, edge, key = row[2].strip(), row[3].strip(), row[4].strip(), row[5].strip()
            # "pipeline - <split>" is per query; bare "pipeline" is the whole process
            if not level.startswith("pipeline - ") or phase != "run":
                continue
            if edge == "start":
                starts[key] = float(row[0])
            elif edge == "end" and key in starts:
                out.append(float(row[0]) - starts.pop(key))
    return out


def wall_seconds(path):
    """Span of the trace, for throughput."""
    ts = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 1:
                try:
                    ts.append(float(row[0]))
                except ValueError:
                    pass
    return (max(ts) - min(ts)) if len(ts) > 1 else 0.0


def q(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(int(p / 100 * len(sorted_vals)), len(sorted_vals) - 1)]


def hier_ci(per_run, p, n_boot=BOOT, seed=0):
    """Hierarchical bootstrap of a quantile: resample runs, then queries."""
    runs = [r for r in per_run if r]
    if len(runs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        pooled = []
        for _ in range(len(runs)):
            src = runs[rng.randrange(len(runs))]
            pooled.extend(src[rng.randrange(len(src))] for _ in range(len(src)))
        pooled.sort()
        stats.append(q(pooled, p))
    stats.sort()
    return (q(stats, 2.5), q(stats, 97.5))


def load(results_dir, machine, keep_first):
    cells = defaultdict(lambda: defaultdict(list))   # cell -> role -> [per-run lists]
    tput = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(results_dir, machine, "e5_*.csv"))):
        meta = parse_name(path)
        if not meta or (not keep_first and meta["run"] <= DROP_RUNS):
            continue
        lat = query_latencies(path)
        if not lat:
            continue
        cells[meta["cell"]][meta["role"]].append(lat)
        secs = wall_seconds(path)
        if secs > 0:
            tput[meta["cell"]][meta["role"]].append(len(lat) / secs)
    return cells, tput


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, choices=["m3pro", "gb10"])
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--baseline", default="baseline",
                    help="cell name of the uncontended reference")
    ap.add_argument("--keep-first", action="store_true",
                    help="keep repetition 1 (dropped as warm-up by default)")
    args = ap.parse_args()

    cells, tput = load(args.results_dir, args.machine, args.keep_first)
    if not cells:
        print(f"no e5 results under {args.results_dir}/{args.machine}")
        return 1

    print(f"# 5.2 — collocation types on {args.machine}\n")
    print(f"Run is the unit of replication; quantiles pooled across runs, 95% CI by "
          f"hierarchical (run-then-query) bootstrap."
          + ("" if args.keep_first else f" Repetition {DROP_RUNS} dropped as warm-up.") + "\n")

    base = cells.get(args.baseline, {}).get("fg", [])
    base_pooled = sorted(x for r in base for x in r)
    if base_pooled:
        print(f"uncontended baseline: p50 {q(base_pooled,50)*1000:.0f} ms, "
              f"p95 {q(base_pooled,95)*1000:.0f} ms  ({len(base)} runs, "
              f"{len(base_pooled)} queries)\n")
    else:
        print("**no baseline cell found — degradation columns omitted**\n")

    print("| cell | runs | fg p50 (ms) | 95% CI | fg p95 (ms) | vs baseline p50 | "
          "fg tput (q/s) | bg tput (q/s) |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|")
    for cell in sorted(cells):
        fg = cells[cell].get("fg", [])
        if not fg:
            continue
        pooled = sorted(x for r in fg for x in r)
        lo, hi = hier_ci(fg, 50)
        p50, p95 = q(pooled, 50) * 1000, q(pooled, 95) * 1000
        deg = ""
        if base_pooled and cell != args.baseline:
            deg = f"{(q(pooled,50)/q(base_pooled,50)-1)*100:+.0f}%"
        ft = st.median(tput[cell].get("fg", [0])) if tput[cell].get("fg") else float("nan")
        bt = st.median(tput[cell].get("bg", [0])) if tput[cell].get("bg") else float("nan")
        print(f"| {cell} | {len(fg)} | {p50:.0f} | [{lo*1000:.0f}, {hi*1000:.0f}] | "
              f"{p95:.0f} | {deg} | {ft:.2f} | "
              + (f"{bt:.2f} |" if bt == bt else "— |"))

    print("\nbg tput is the background pipeline's own delivered rate, from its own run —")
    print("the attribution this arrangement exists to provide. '—' means the cell has")
    print("no background (the baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
