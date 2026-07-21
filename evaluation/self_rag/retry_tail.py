#!/usr/bin/env python3
"""Measured open-loop per-request latency tail for multi-call critique-graph RAG.

Recovers per-request end-to-end latency (arrival -> completion) from an EXISTING
open-loop Poisson run — no new instrumentation needed:
  * <base>_arrivals.csv : per-query intended arrival ts (the open-loop schedule)
  * <base>.csv (trace)  : "End stage :: run" with perf-col "end" = per-query completion
                          "Query rewrite LLM :: run"/"start" within a query span = a retry
Serial pipeline => queries complete in arrival order (FIFO), so the i-th End-stage-end
is query i (epoch i). Correlates per-request latency with retry count and asks whether
the tail is retry-driven or service-variance-driven.

Usage: python retry_tail.py <results_dir>/<cell_base>   (base without extension)
"""
import sys, statistics as st


def analyze(base):
    intended, actual = [], []
    for l in list(open(base + "_arrivals.csv"))[1:]:
        f = l.split(",")
        intended.append(float(f[1])); actual.append(float(f[2]))
    comp, retr, cur = [], [], 0
    for line in open(base + ".csv", errors="ignore"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue
        ts, stage, ev, se = float(p[0]), p[2], p[3], p[4]
        if stage == "Query rewrite LLM" and ev == "run" and se == "start":
            cur += 1
        if stage == "End stage" and ev == "run" and se == "end":
            comp.append(ts); retr.append(cur); cur = 0
    n = min(len(comp), len(intended))
    lat = [comp[i] - intended[i] for i in range(n)]
    delay = [actual[i] - intended[i] for i in range(n)]
    ls = sorted(lat)
    P = lambda q: ls[min(n - 1, int(q * n))]
    print(f"cell: {base.split('/')[-1]}   n={n}")
    print(f"submission delay (actual-intended): mean={st.mean(delay):.2f}s max={max(delay):.2f}s"
          f"  [>0 => scheduler-side queueing/coordinated omission]")
    print(f"PER-REQUEST LATENCY (s): mean={st.mean(lat):.1f} p50={P(.5):.1f} p90={P(.9):.1f} "
          f"p95={P(.95):.1f} p99={P(.99):.1f} max={max(lat):.1f}  (tail p99/p50={P(.99)/P(.5):.1f}x)")
    print("retry dist: " + ", ".join(f"{k}:{retr.count(k)}" for k in sorted(set(retr)))
          + f"   retried={100*sum(1 for r in retr if r>0)/n:.0f}%")
    for k in sorted(set(retr)):
        g = [lat[i] for i in range(n) if retr[i] == k]
        gs = sorted(g)
        print(f"  retry={k} n={len(g):3d}  latency mean={st.mean(g):5.1f}s "
              f"p90={gs[min(len(g)-1,int(.9*len(g)))]:5.1f}s max={max(g):5.1f}s")
    thr = P(.9)
    tail = [i for i in range(n) if lat[i] >= thr]
    rtail = sum(1 for i in tail if retr[i] > 0)
    print(f"top-10% latency tail (n={len(tail)}): {rtail} retried ({100*rtail/len(tail):.0f}%) "
          f"vs {100*sum(1 for r in retr if r>0)/n:.0f}% retried overall "
          f"=> tail is {'retry-enriched' if rtail/len(tail) > sum(1 for r in retr if r>0)/n else 'NOT retry-dominated'}")


if __name__ == "__main__":
    analyze(sys.argv[1])
