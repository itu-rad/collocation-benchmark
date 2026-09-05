#!/usr/bin/env python3
"""Measure DRAM bytes per query for each co-runner, so the intensity axis can be
matched on BYTES MOVED rather than on each engine's own capacity.

Why this exists. The three co-runners are compared at "the same intensity", and
what that means is a real choice. Levels L25..L100 are fractions of each
co-runner's OWN saturating rate, so "L50" is 8.4 q/s of memory-stream against
9.2 q/s of GPU encode against 10.0 q/s of ANE encode -- equal effort relative to
each engine, but not equal pressure on the thing they actually share, which is
the memory system. Section 5.2's claim is about that shared memory system, so
the axis is matched on bytes/s.

Only MemoryStream can state its traffic analytically (`bytes_per_query()`:
passes x 3 x size_mb). The CLIP co-runners cannot, so they are measured: run
each one ALONE with the AMC per-engine sampler, take the steady-state bandwidth
over idle, and divide by the delivered query rate.

    python evaluation/contention/calibrate_bytes.py --device mlx

MACHINE NOTE: both Apple DUTs can now run this. The M2 Pro uses the exact AMC
per-requestor byte counters; the M3 Pro cannot subscribe to those and uses the
PMP bandwidth-histogram backend instead, which the sampler selects automatically
(see contention.md and docs/amc-m3-counters-plan.md). Historically this ran on
the M2 Pro only. Bytes per query is a property of the model, the input size and
the framework, so it carries across Apple machines; the delivered RATE is still
set per machine.

PREFER THE M2 PRO FOR SETTING RATES. The matched-bytes rates in
generate_stage_configs.py come from the M2 Pro's exact byte counters and should
stay there. The M3 backend derives bytes from bandwidth histograms and
time-averages a gated engine against the aggregate's tick count -- sound for a
smooth load, but a bursty one (a co-runner that runs flat out then sleeps, which
is what the CLIP encoders do) is where that estimate is weakest. Re-deriving a
bursty co-runner's bytes/query there risks setting its rate wrong in a way
nothing downstream would catch. Use the M3 backend to CONFIRM a dose and to
attribute traffic to an engine -- which is what exhibit 2 needs -- rather than to
set the rates.

If you re-run it on the M3 Pro: keep every co-runner under the backend's 32 GB/s
per-requestor ceiling (the ladder already is) and reject any sample whose CSV
`saturated` column is 1, since such a row is only a lower bound. An ANE co-runner
needs >=10 s of warm-up before the measurement window -- CoreML serves the first
~8 s on the CPU and the ANE bucket reads zero throughout it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CFG_DIR = HERE / "configs"
SAMPLER = ROOT / "scripts" / "amc_bandwidth_sampler.py"


def bg_only_config(kind: str, device: str, out_dir: Path) -> tuple[Path, float]:
    """Extract the background pipeline of stage_c_<kind>_L100 into its own config."""
    src = CFG_DIR / f"stage_c_{kind}_L100_{device}.yml"
    doc = yaml.safe_load(src.read_text())
    bg = [p for p in doc["pipelines"] if p.get("name", "").startswith("BG ")]
    if len(bg) != 1:
        raise SystemExit(f"{src.name}: expected 1 background pipeline, found {len(bg)}")
    bg = bg[0]
    interval = float((bg["loadgen"].get("config") or {}).get("interval") or 0)
    # Long enough to reach steady state, short enough to be a pilot.
    bg["loadgen"]["max_queries"] = int(120 / interval) if interval else 500
    bg["loadgen"]["timeout"] = 300
    bg["loadgen"]["queue_depth"] = 16
    out = out_dir / f"calib_{kind}_{device}.yml"
    out.write_text(yaml.safe_dump(
        {"name": f"calib_{kind}_{device}", "listeners": [], "pipelines": [bg]},
        sort_keys=False))
    return out, interval


def idle_bandwidth(seconds: float, out: Path) -> float:
    subprocess.run([sys.executable, str(SAMPLER), "--duration", str(seconds),
                    "--interval", "0.5", "--out", str(out)],
                   check=True, capture_output=True)
    return mean_gbps(out)


def mean_gbps(path: Path, skip_first: int = 2) -> float:
    vals = []
    with open(path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < skip_first:
                continue
            try:
                vals.append(float(row["total_gbps"]))
            except (KeyError, ValueError):
                pass
    return st.median(vals) if vals else 0.0


def run_corunner(cfg: Path, work: Path, label: str) -> tuple[float, int, float]:
    """Run the co-runner alone under the sampler. -> (median GB/s, queries, seconds)."""
    bw = work / f"{label}_bw.csv"
    env = dict(os.environ, SUITE_OUTPUT_DIR=str(work))
    env.pop("SUITE_PROC_TRACE", None)          # spans are not needed for a pilot
    env["SUITE_DISABLE_TRACING"] = "1"
    sampler = subprocess.Popen(
        [sys.executable, str(SAMPLER), "--interval", "0.5", "--out", str(bw)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / "main.py"), str(cfg), "-p", "0"],
                          env=env, capture_output=True, text=True)
    secs = time.time() - t0
    sampler.terminate(); sampler.wait(timeout=10)
    if proc.returncode != 0:
        print(proc.stdout[-1500:], file=sys.stderr)
        raise SystemExit(f"co-runner {label} failed rc={proc.returncode}")
    # queries completed = per-query pipeline run rows
    n = 0
    for p in work.glob("*.csv"):
        if p == bw:
            continue
        with open(p) as f:
            for row in csv.reader(f):
                if len(row) > 5 and row[2].strip().startswith("pipeline - ") \
                   and row[3].strip() == "run" and row[4].strip() == "end":
                    n += 1
    return mean_gbps(bw), n, secs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mlx", choices=["mlx", "cuda"])
    ap.add_argument("--kinds", nargs="+", default=["stream", "clipgpu", "clipane"])
    ap.add_argument("--out", default=str(HERE / "bytes_calibration.json"))
    args = ap.parse_args()

    if not SAMPLER.exists():
        raise SystemExit(f"missing {SAMPLER}")

    work = Path(tempfile.mkdtemp(prefix="calib_"))
    print(f"working dir: {work}\n")

    idle = idle_bandwidth(6, work / "idle_bw.csv")
    print(f"idle DRAM bandwidth: {idle:.2f} GB/s\n")

    results = {"device": args.device, "idle_gbps": idle, "corunners": {}}
    for kind in args.kinds:
        cfg, interval = bg_only_config(kind, args.device, work)
        print(f"-- {kind}: running alone at {1/interval:.1f} q/s (its L100) ...")
        for p in work.glob("*.csv"):
            if not p.name.endswith("idle_bw.csv"):
                p.unlink()
        gbps, n, secs = run_corunner(cfg, work, kind)
        if n == 0:
            print(f"   {kind}: no completed queries -- skipped\n")
            continue
        rate = n / secs
        over_idle = max(gbps - idle, 0.0)
        bpq = over_idle * 1e9 / rate if rate else 0.0
        results["corunners"][kind] = {
            "gbps_total": gbps, "gbps_over_idle": over_idle,
            "queries": n, "seconds": secs, "delivered_qps": rate,
            "bytes_per_query": bpq,
        }
        print(f"   {gbps:.2f} GB/s total, {over_idle:.2f} over idle; "
              f"{n} queries in {secs:.0f}s = {rate:.2f} q/s")
        print(f"   -> {bpq/1e9:.3f} GB/query\n")

    # MemoryStream states its traffic analytically; compare as a sanity check.
    try:
        d = yaml.safe_load((CFG_DIR / f"stage_c_stream_L100_{args.device}.yml").read_text())
        c = d["pipelines"][1]["stages"][0].get("config") or {}
        analytic = c.get("passes", 4) * 3 * c.get("size_mb", 256) * (1 << 20)
        results["stream_analytic_bytes_per_query"] = analytic
        if "stream" in results["corunners"]:
            m = results["corunners"]["stream"]["bytes_per_query"]
            print(f"stream cross-check: analytic {analytic/1e9:.3f} GB/query vs "
                  f"measured {m/1e9:.3f} ({m/analytic:.2f}x)")
    except Exception as e:  # noqa: BLE001
        print(f"stream cross-check skipped: {e}")

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
