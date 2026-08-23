#!/usr/bin/env python3
"""Generate the E4 SERIALIZED configs — one request in flight, batch 1.

Why. The Poisson serving configs saturate: on mlx only 28/110 queries completed
on `factoid decomposed`, the `decomposed` arm ran with a median of 3.00
concurrently-executing LLM stages (four separate 4B models, independent locks),
and the two devices sat at different offered load (rho ~0.17-0.20 vs ~0.42-0.61).
Under those conditions a per-call prefill/decode time is not a phase measurement,
it is a queueing measurement — and the phase split is exactly what E4 claims.

With serialize_queries + queue_depth 1 there is one request in the pipeline at a
time, so: no co-runners, no queue wait inside the stage window, every query
completes, and both devices run the identical schedule. Keep the Poisson configs
for end-to-end serving latency; use these for the phase characterisation.

max_queries is reduced to 30 because the phase split is a per-CALL property: 30
queries still yields ~120 LLM calls per decomposed arm, far more than needed for
a median, and it keeps a full R sweep inside a few hours on both devices.

    python gen_serialized_configs.py [--queries 30]
"""
import argparse
import copy
import glob
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = ("ollama", "vllm", "_0.8b_", "_2b_", "_27b_", "serial", "quantest", "e4smoke")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs"))
    args = ap.parse_args()

    n = 0
    for src in sorted(glob.glob(os.path.join(HERE, "configs", "*_mlx.yml"))
                      + glob.glob(os.path.join(HERE, "configs", "*_cuda.yml"))):
        base = os.path.basename(src)
        if any(s in base for s in SKIP):
            continue
        cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))
        cfg = copy.deepcopy(cfg)
        cfg["listeners"] = []
        for pipe in cfg["pipelines"]:
            pipe["serialize_queries"] = True
            lg = pipe["loadgen"]
            lg["queue_depth"] = 1
            lg["max_queries"] = args.queries
            # closed loop: the next request is issued when the previous finishes,
            # so the arrival process cannot build a queue.
            lg["component"] = "loadgen.OfflineLoadScheduler"
            lg["config"] = {"rate": 0}
            lg["timeout"] = 3600000
        # Name as <task>_<arm>_serial_<device>.yml so that (a) collect.sh's
        # "<glob>_<device>.yml" pattern finds them with GLOB='*_serial' and
        # (b) analyze_e4's label regex still ends in _<device>_r<N>, giving the
        # arm name "<arm>_serial" — distinct from the Poisson serving arms.
        stem, dev = base[:-4].rsplit("_", 1)
        out = os.path.join(args.out_dir, f"{stem}_serial_{dev}.yml")
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        n += 1
    print(f"wrote {n} serialized configs ({args.queries} queries, queue_depth 1)")


if __name__ == "__main__":
    main()
