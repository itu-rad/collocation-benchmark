#!/usr/bin/env python3
"""Section 5.1 interplay table: quality x latency x throughput x memory x power.

Joins three sources, each of which has to come from the pass that can measure it:

  latency, throughput  the SERIAL pass (listeners off) -- the counter pass would
                       fold the observer's cost into the headline numbers
  power, energy, memory the OBS pass (listeners on), read from the tracking server
  quality              the LLM-judge verdicts under judge/

Power and memory come from different instruments per machine and are NOT directly
comparable across them -- macmon reports package power on Apple, DCGM reports GPU
power on gb10, which excludes the CPU side. The table says so rather than
implying one number.

    python evaluation/self_rag/build_interplay.py
    python evaluation/self_rag/build_interplay.py --machines m3pro gb10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ARMS = ["monolith", "monolith_4b", "decomposed", "decomposed_shared"]
TASKS = ["factoid", "multihop"]

# What each machine's listeners actually report. Keyed to the series names
# verified on the tracking server.
POWER = {"m3pro": "system/macmon - All Power",       # package, whole SoC
         "gb10":  "system/DCGMI - Power Usage"}      # GPU only
# Both are SYSTEM memory, which is the meaningful figure on two unified-memory
# machines -- but it is not GPU framebuffer, and on gb10 it includes whatever
# else the shared node is holding.
MEMORY = {"m3pro": "system/macmon - Memory:Ram Usage",
          "gb10":  "system/TOP - Memory Usage GB"}
ENERGY = {"gb10": "system/DCGMI - Total Energy Consumption"}
POWER_SCOPE = {"m3pro": "SoC package", "gb10": "GPU only"}
# macmon reports bytes; DCGM/TOP report GB.
MEM_IN_BYTES = {"m3pro": True, "gb10": False}


def quality(machine, task, arm):
    p = os.path.join(HERE, "judge", f"verdicts_{machine}_{task}_{arm}.json")
    if not os.path.exists(p):
        return None
    v = json.load(open(p))
    return sum(x["correct"] for x in v) / len(v) if v else None


def counters(machine, verbose=False):
    """Per-cell power/memory/energy from the obs runs on the tracking server.

    Three things this must not do, each of which produces a plausible wrong number:

    * `run.data.metrics[k]` is the LAST logged value of a series, not a summary.
      Reading it gives whatever the sampler happened to report as the run ended.
      The history has to be fetched and reduced over the run.
    * DCGM's "Total Energy Consumption" is CUMULATIVE since driver load -- tens of
      gigajoules. The run's energy is last minus first, not the level.
    * macmon reports memory in BYTES and DCGM/TOP in GB. Peak memory is also a
      max over samples, not a median: a median peak is not a peak.
    """
    try:
        import mlflow
    except ImportError:
        print("  (mlflow unavailable -- power/memory columns omitted)", file=sys.stderr)
        return {}
    c = mlflow.MlflowClient()
    runs, token = [], None
    while True:
        page = c.search_runs(["138"], max_results=1000, page_token=token)
        runs += [r for r in page
                 if f"_{machine}_obs_r" in r.data.tags.get("mlflow.runName", "")]
        token = page.token
        if not token:
            break
    out = {}
    for r in runs:
        name = r.data.tags["mlflow.runName"].split(" |")[0]
        body = name[len("e4_"):].rsplit(f"_{machine}_obs_r", 1)[0]
        task = body.split("_", 1)[0]
        arm = body.split("_", 1)[1] if "_" in body else ""
        key = (task, arm)
        rec = out.setdefault(key, {"power": [], "mem": [], "energy": []})
        hist = lambda k: [p.value for p in c.get_metric_history(r.info.run_id, k)]

        pw = hist(POWER[machine]) if POWER[machine] in r.data.metrics else []
        if pw:
            rec["power"].append(st.mean(pw))          # mean over the run

        mm = hist(MEMORY[machine]) if MEMORY[machine] in r.data.metrics else []
        if mm:
            peak = max(mm)
            rec["mem"].append(peak / 1e9 if MEM_IN_BYTES.get(machine) else peak)

        ek = ENERGY.get(machine)
        if ek and ek in r.data.metrics:
            ev = hist(ek)
            if len(ev) > 1:
                # DCGM reports this field in MILLIjoules. Cross-check: the raw
                # delta over a ~177 s run divided by that run's mean power gives
                # exactly the run length once scaled by 1000, and not otherwise.
                rec["energy"].append((ev[-1] - ev[0]) / 1000.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", nargs="+", default=["m3pro", "gb10"])
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    from analyze_e4 import load, agg_by_arm

    print("# 5.1 — the interplay table\n")
    print("Latency from the listener-off serial pass; power and memory from the "
          "listener-on counter pass; quality from the LLM judge over the same 30 "
          "questions on both machines.\n")

    for machine in args.machines:
        runs = load(args.results_dir, machine, "serial")
        if not runs:
            print(f"## {machine}: no serial runs\n")
            continue
        agg = agg_by_arm(runs)
        ctr = counters(machine)
        print(f"## {machine}  (power: {POWER_SCOPE[machine]})\n")
        has_energy = machine in ENERGY
        head = ("| task | strategy | quality | prefill (ms) | decode (ms) | "
                "decode tok/s | power (W) | peak mem (GB) |")
        rule = "|---|---|--:|--:|--:|--:|--:|--:|"
        if has_energy:
            head += " energy (J) |"
            rule += "--:|"
        print(head)
        print(rule)
        for task in TASKS:
            for arm in ARMS:
                a = agg.get((task, arm))
                if not a:
                    continue
                med = lambda x: st.median(x) if isinstance(x, (list, tuple)) else x
                q = quality(machine, task, arm)
                cr = ctr.get((task, arm), {})
                f = lambda vals: f"{st.median(vals):.1f}" if vals else "—"
                row = (f"| {task} | {arm} | "
                       f"{'—' if q is None else f'{q:.3f}'} | "
                       f"{med(a['prefill']):.0f} | {med(a['decode']):.0f} | "
                       f"{med(a['tok_s']):.1f} | {f(cr.get('power'))} | "
                       f"{f(cr.get('mem'))} |")
                if has_energy:
                    row += f" {f(cr.get('energy'))} |"
                print(row)
        print()

    print("Power is not comparable across machines: macmon reports SoC package "
          "power, DCGM reports GPU power only and so excludes the CPU side. "
          "Energy is a DCGM cumulative counter, reported here as the per-run "
          "delta in joules, and exists on gb10 only. Peak memory is system "
          "memory on both -- meaningful on unified-memory parts, but not GPU "
          "framebuffer, and on the shared gb10 it includes other tenants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
