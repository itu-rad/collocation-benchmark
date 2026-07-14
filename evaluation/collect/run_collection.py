#!/usr/bin/env python3
"""Main collection driver — the full paper matrix on one device.

Runs, in value order, on knob-locked configs (evaluation/pilots/knobs.yml):

  phase e4_factoid   4 arms x {pipelined, serial} x R=5 (timing)
                     + 1 serial quality run x 120 questions per arm
  phase e4_multihop  same shape (the slow phase: ~31 min/run on mlx)
  phase staged       Stage A-D contention family (evaluation/contention/configs)
                     B>0 cells use the radt multi-pipeline orchestrator path
  phase e7           size rungs {0.8b, 2b, 27b} quality+memory (4b/9b come from
                     e4_factoid); 27B OOM on the M2 is an EXPECTED datum
  phase e5           ResNet scenario reduction: Server (as-committed) +
                     SingleStream / Offline / MultiStream via loadgen override,
                     R=10

E2's cuda half runs separately via run_modularity.py (own driver).
Timing cells run the configs AS COMMITTED (tracing on — the instrumented
framework is the paper's measurement instrument); quality cells override the
loadgen to a serial closed loop of n_quality queries. Results + arrivals
sidecars + _outputs.jsonl are curated into evaluation/collect/results/<dev>/.
Writes collect_env_<dev>.txt (commit-pinned) and DONE_<dev> marker on success.

    python evaluation/collect/run_collection.py --device {mlx,cuda}
    python evaluation/collect/run_collection.py --device mlx --only 'e4_factoid*'
    python evaluation/collect/run_collection.py --list
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
GLOBAL_RESULTS = REPO_ROOT / "evaluation" / "results"
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "pilots"))
import pilot_lib as pl  # noqa: E402
from run_pilots import check_environment, _git_sha  # noqa: E402

SR = "evaluation/self_rag/configs"
CT = "evaluation/contention/configs"
DEVNAME = {"mlx": "m2pro", "cuda": "gb10"}


@dataclass
class Cell:
    label: str                  # unique cell label (device suffix appended)
    config: str                 # repo-relative config path (device-resolved)
    phase: str
    runs: int = 5
    serialize: bool = False     # --serialize true (serial schedule cell)
    quality_n: int | None = None  # override loadgen -> serial closed loop of N
    loadgen_override: dict | None = None  # e5 scenario swaps
    orchestrated: bool = False  # multi-pipeline config: run WITHOUT -p 0
    timeout_s: int = 7200
    tolerate_failure: bool = False  # e.g. the 27B OOM ceiling datum


def build_cells(device: str) -> list[Cell]:
    knobs = pl.load_knobs()
    dn = DEVNAME[device]

    def knob(exp, name, default=None):
        return pl.get_knob(knobs, exp, dn, name, default)

    cells: list[Cell] = []

    # ---- E4 timing + quality ------------------------------------------------
    arms = ["monolith", "decomposed", "monolith_4b", "decomposed_shared"]
    for task in ("factoid", "multihop"):
        phase = f"e4_{task}"
        # multihop runs last ~40/λ s; give 2x headroom over the knob timeout
        for arm in arms:
            cfg = f"{SR}/{task}_{arm}_{device}.yml"
            for sched, ser in (("pipe", False), ("serial", True)):
                cells.append(Cell(f"{phase}_{arm}_{sched}", cfg, phase,
                                  runs=5, serialize=ser,
                                  timeout_s=4 * 3600 if task == "multihop" else 7200))
            cells.append(Cell(f"{phase}_{arm}_quality", cfg, phase, runs=1,
                              quality_n=int(knob("e4", "n_quality", 120)),
                              timeout_s=4 * 3600 if task == "multihop" else 7200))

    # ---- Staged contention (Stages A-D) -------------------------------------
    ct_dir = REPO_ROOT / CT
    for p in sorted(ct_dir.glob(f"stage_*_{device}.yml")):
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        multi = len(doc.get("pipelines", [])) > 1
        cells.append(Cell(p.stem, str(p.relative_to(REPO_ROOT)), "staged",
                          runs=5, orchestrated=multi, timeout_s=3600))

    # ---- E7 size rungs (quality + resident memory; 4B/9B from e4_factoid) ---
    for rung in ("0.8b", "2b", "27b"):
        cfg = f"{SR}/factoid_monolith_{rung}_{device}.yml"
        if (REPO_ROOT / cfg).exists():
            cells.append(Cell(f"e7_rung_{rung}", cfg, "e7", runs=1,
                              quality_n=int(knob("e4", "n_quality", 120)),
                              timeout_s=7200,
                              tolerate_failure=(rung == "27b")))

    # ---- E5 scenario reduction ----------------------------------------------
    resnet = "pipeline_configs/mlperf/resnet_inference.yml"
    lam = knob("e5", "loadgen.config.rate")
    n_q = int(knob("e5", "loadgen.max_queries", 500))
    interval = knob("e5", "multistream_interval_s")
    scenarios = {
        "server": None,  # as committed (Poisson at the locked λ)
        "singlestream": {"component": "loadgen.OfflineLoadScheduler",
                         "queue_depth": 4, "max_queries": n_q,
                         "timeout": 1800, "config": {"rate": 0}},
        "offline": {"component": "loadgen.SaturatingOfflineScheduler",
                    "queue_depth": n_q, "max_queries": n_q,
                    "timeout": 1800, "config": {"rate": 0}},
    }
    if interval:
        scenarios["multistream"] = {"component": "loadgen.MultiStreamScheduler",
                                    "queue_depth": n_q, "max_queries": n_q,
                                    "timeout": 1800,
                                    "config": {"interval": interval}}
    for name, ov in scenarios.items():
        cells.append(Cell(f"e5_{name}", resnet, "e5", runs=10,
                          loadgen_override=ov, timeout_s=1800))

    return cells


def make_config(base_rel: str, device: str, quality_n: int | None,
                loadgen_override: dict | None) -> str:
    doc = yaml.safe_load((REPO_ROOT / base_rel).read_text(encoding="utf-8"))
    doc["listeners"] = ["macmon"] if device == "mlx" else ["top"]
    for pipe in doc["pipelines"]:
        if quality_n:
            pipe["loadgen"] = {"component": "loadgen.OfflineLoadScheduler",
                               "queue_depth": 4, "max_queries": quality_n,
                               "timeout": 6 * 3600, "config": {"rate": 0}}
        elif loadgen_override:
            pipe["loadgen"] = dict(loadgen_override)
        if device == "mlx":
            for stage in pipe.get("stages", []):
                cfg = stage.get("config") or {}
                if cfg.get("device") == "cuda":
                    cfg["device"] = "mps"
    fd, tmp = tempfile.mkstemp(suffix=".yml", prefix="collect_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return tmp


def run_cell(cell: Cell, device: str, results_dir: Path, force: bool) -> bool:
    ok = True
    for r in range(1, cell.runs + 1):
        label = f"{cell.label}_{device}_r{r}"
        target = results_dir / f"{label}.csv"
        if target.exists() and not force:
            print(f"[skip] {label}", flush=True)
            continue
        tmp = make_config(cell.config, device, cell.quality_n,
                          cell.loadgen_override)
        cmd = [sys.executable, "main.py", tmp]
        if not cell.orchestrated:
            cmd += ["-p", "0"]
        cmd += ["--label", label]
        if cell.serialize:
            cmd += ["--serialize", "true"]
        print(f"[run ] {label} ({cell.config})", flush=True)
        t0 = time.time()
        log_path = results_dir / f"{label}.log"
        results_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w") as logf:
                proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=logf,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True)
                rc = proc.wait(timeout=cell.timeout_s)
        except subprocess.TimeoutExpired:
            # kill the whole process group (grandchildren included) — a wedged
            # cell must never wedge the pass
            try:
                os.killpg(os.getpgid(proc.pid), 15)
                time.sleep(10)
                os.killpg(os.getpgid(proc.pid), 9)
            except ProcessLookupError:
                pass
            print(f"[FAIL] {label}: timeout {cell.timeout_s}s", flush=True)
            ok = ok and cell.tolerate_failure
            os.unlink(tmp)
            continue
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        src = GLOBAL_RESULTS / f"{label}.csv"
        if rc != 0 or not src.exists():
            print(f"[{'tolerated' if cell.tolerate_failure else 'FAIL'}] "
                  f"{label}: rc={rc} (log: {log_path.name})", flush=True)
            ok = ok and cell.tolerate_failure
            continue
        shutil.move(str(src), target)
        for suffix in ("_arrivals.csv", "_outputs.jsonl"):
            p = GLOBAL_RESULTS / f"{label}{suffix}"
            if p.exists():
                shutil.move(str(p), results_dir / f"{label}{suffix}")
        print(f"[done] {label} in {time.time() - t0:.0f}s", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["mlx", "cuda"],
                    default="mlx" if platform.system() == "Darwin" else "cuda")
    ap.add_argument("--only", default=None, help="glob over cell labels")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cells = build_cells(args.device)
    if args.only:
        cells = [c for c in cells if fnmatch.fnmatch(c.label, args.only)]
    if args.list:
        total = sum(c.runs for c in cells)
        for c in cells:
            print(f"  {c.phase:12s} {c.label:36s} R={c.runs} "
                  f"{'serial' if c.serialize else 'quality' if c.quality_n else 'orchestrated' if c.orchestrated else ''}")
        print(f"  -- {len(cells)} cells, {total} runs")
        return 0
    if not cells:
        sys.exit("no cells match")

    env_lines = check_environment(args.device)
    sha = _git_sha()
    if sha.endswith("-dirty"):
        sys.exit("[collect FATAL] dirty tree — commit before collection")
    results_dir = HERE / "results" / args.device
    results_dir.mkdir(parents=True, exist_ok=True)
    (HERE / f"collect_env_{args.device}.txt").write_text(
        "\n".join([f"date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
                   f"git_commit: {sha}",
                   f"device: {args.device}",
                   f"platform: {platform.platform()}"] + env_lines) + "\n",
        encoding="utf-8")

    failures = []
    for c in cells:
        if not run_cell(c, args.device, results_dir, args.force):
            failures.append(c.label)
    if failures:
        print(f"\nFAILED cells: {failures}", flush=True)
        return 1
    (HERE / f"DONE_{args.device}").write_text(time.strftime("%F %T"))
    print("\nCollection pass complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
