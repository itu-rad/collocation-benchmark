#!/usr/bin/env python3
"""Generate the E1 `nolog` control arm: identical NoOp chains with per-stage
trace logging switched OFF.

Why this arm exists. Every NoOp config ships `disable_logs: false`, so the
measured "core dispatch" (25-39 us/stage) includes the framework's OWN CSV trace
writes — four logging.info calls per stage per query. Without a logs-off arm we
are attributing to dispatch a cost that is partly our instrument.

This works because the metric of record, L_q, comes from the pipeline-level
`pipeline - <split>` row emitted in pipeline/pipeline.py, which does NOT consult
any stage's disable_logs. So the per-stage rows disappear while per-query latency
is still measured, and

    slope(L_q vs depth) with logs off   = dispatch alone
    slope(L_q vs depth) with logs on    = dispatch + our logging
    difference                          = the instrument's own per-stage cost

    python gen_nolog_configs.py [--out-dir DIR]
"""
import argparse
import glob
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=os.path.join(HERE, "configs"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs"))
    args = ap.parse_args()

    n = 0
    # Every NoOp config, not just the depth sweep's size-0/ref ones: the
    # payload sweep is now collected in the two instrument-free arms too, and
    # they need a logs-off variant of each size x mode cell.
    for src in sorted(glob.glob(os.path.join(args.configs,
                                             "noop_depth_*_size_*_mode_*.yml"))):
        if src.endswith("_nolog.yml"):
            continue
        cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))
        cfg["name"] = cfg.get("name", "") + " (logs off)"
        for pipe in cfg["pipelines"]:
            pipe["name"] = pipe["name"] + " nolog"
            for stage in pipe["stages"]:
                stage["disable_logs"] = True
        out = os.path.join(args.out_dir,
                           os.path.basename(src)[:-4] + "_nolog.yml")
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        n += 1
    print(f"wrote {n} nolog configs to {args.out_dir}")


if __name__ == "__main__":
    main()
