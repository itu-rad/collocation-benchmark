#!/usr/bin/env python3
"""Did a run actually record hardware counters, and can they be joined to spans?

The profiling contribution rests on correlating application spans with
system-level metrics. Both halves fail silently: a listener whose binary is
missing never spawns, and a metric series that never arrives looks exactly like
a quiet machine. This checks a run end to end rather than trusting that
`listeners:` in the YAML did something.

    python scripts/check_listener_metrics.py <run-name-prefix> [--experiment 138]
"""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="run-name prefix, e.g. SMOKE_listeners_m3pro")
    ap.add_argument("--experiment", default="138")
    args = ap.parse_args()

    import mlflow
    c = mlflow.MlflowClient()

    runs, token = [], None
    while True:
        page = c.search_runs([args.experiment], max_results=1000, page_token=token)
        runs += [r for r in page
                 if r.data.tags.get("mlflow.runName", "").startswith(args.prefix)]
        token = page.token
        if not token:
            break
    if not runs:
        print(f"no run named {args.prefix}* in experiment {args.experiment}",
              file=sys.stderr)
        return 1

    r = max(runs, key=lambda x: x.info.start_time)
    name = r.data.tags.get("mlflow.runName")
    print(f"run: {name}  ({r.info.run_id})  status={r.info.status}")

    system = sorted(k for k in r.data.metrics if k.startswith("system/"))
    if not system:
        print("  NO system/* metrics -- the listeners did not record anything.",
              file=sys.stderr)
        print("  Check: the listener binary on PATH, and RADT_LISTENER_<NAME>=True.",
              file=sys.stderr)
        return 1

    print(f"  {len(system)} system metric series:")
    groups = {}
    for k in system:
        groups.setdefault(k.split(" - ")[0], []).append(k)
    for g, ks in sorted(groups.items()):
        print(f"    {g}: {len(ks)} series")
        for k in ks[:6]:
            print(f"       {k}")
        if len(ks) > 6:
            print(f"       ... and {len(ks) - 6} more")

    # Sample density and whether the window overlaps the run -- a series with one
    # point cannot be joined to anything.
    print("\n  sample density and span overlap:")
    ok = True
    for k in system[:4]:
        hist = c.get_metric_history(r.info.run_id, k)
        if len(hist) < 2:
            print(f"    {k}: {len(hist)} sample(s) -- too few to join")
            ok = False
            continue
        ts = sorted(h.timestamp for h in hist)
        span_s = (ts[-1] - ts[0]) / 1000.0
        rate = (len(ts) - 1) / span_s if span_s else float("nan")
        inside = r.info.start_time <= ts[0] and ts[-1] <= (r.info.end_time or ts[-1])
        print(f"    {k}: {len(ts)} samples over {span_s:.1f}s "
              f"(~{rate:.2f} Hz), within run window: {inside}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
