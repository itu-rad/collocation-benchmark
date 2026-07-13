#!/usr/bin/env python3
"""Pilot driver for the hyperparameter protocol (PAPER_TODO §2.6).

Runs short SERIAL pilots (closed-loop OfflineLoadScheduler, one query in
flight) per (workload, device) and curates their trace CSVs into
``evaluation/pilots/results/``. The pilots supply the two rule inputs —
serial service time and warm-up horizon — that derive_knobs.py turns into
every experiment knob. Pilots are excluded from reported data and are
commit-pinned (pilot_env.txt records the git SHA; derive_knobs.py refuses to
mix commits).

    python evaluation/pilots/run_pilots.py --device mlx          # M2 Pro
    python evaluation/pilots/run_pilots.py --device cuda         # GB10 (Ties)
    python evaluation/pilots/run_pilots.py --list                # show cells
    python evaluation/pilots/run_pilots.py --only e4_factoid_mono4b --runs-override 1 --n-override 5   # smoke

Follows the run_matrix.py driver idiom: subprocess `python main.py <tmpcfg>
-p 0 --label pilot_<cell>_r<r>`, loadgen-block override via yaml→tempfile,
idempotent (skips existing results unless --force).
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
RESULTS_DIR = HERE / "results"
GLOBAL_RESULTS = REPO_ROOT / "evaluation" / "results"
ENV_FILE = HERE / "pilot_env.txt"


# ---------------------------------------------------------------------------
# Cell table
# ---------------------------------------------------------------------------

@dataclass
class PilotCell:
    id: str                      # cell id (device-independent)
    config: dict                 # device -> repo-relative base config path
    n_queries: int
    runs: int = 1
    timeout_s: int = 3600        # per-subprocess hard cap
    serves: list = field(default_factory=list)
    blocked_on: str | None = None  # unbuilt apparatus; listed, never run

    def config_for(self, device: str) -> str | None:
        return self.config.get(device)


SR = "evaluation/self_rag/configs"
PC = "pipeline_configs"

CELLS = [
    # --- E4 Self-RAG (λ-determining cell first: slowest-arm candidate, R=2) --
    PilotCell("e4_factoid_mono9b",
              {"mlx": f"{SR}/factoid_monolith_mlx.yml",
               "cuda": f"{SR}/factoid_monolith_cuda.yml"},
              n_queries=20, runs=2, timeout_s=3600,
              serves=["E4 λ (slowest-arm candidate)", "E4 warm-up k"]),
    PilotCell("e4_factoid_decomp",
              {"mlx": f"{SR}/factoid_decomposed_mlx.yml",
               "cuda": f"{SR}/factoid_decomposed_cuda.yml"},
              n_queries=20, serves=["E4 slowest-arm race"]),
    PilotCell("e4_factoid_mono4b",
              {"mlx": f"{SR}/factoid_monolith_4b_mlx.yml",
               "cuda": f"{SR}/factoid_monolith_4b_cuda.yml"},
              n_queries=20, serves=["E4 size control"]),
    PilotCell("e4_factoid_shared",
              {"mlx": f"{SR}/factoid_decomposed_shared_mlx.yml",
               "cuda": f"{SR}/factoid_decomposed_shared_cuda.yml"},
              n_queries=20, serves=["E4 logical control"]),
    PilotCell("e4_multihop_mono9b",
              {"mlx": f"{SR}/multihop_monolith_mlx.yml",
               "cuda": f"{SR}/multihop_monolith_cuda.yml"},
              n_queries=12, timeout_s=5400,
              serves=["E4 multi-hop λ", "retry-loop tail"]),
    PilotCell("e4_multihop_decomp",
              {"mlx": f"{SR}/multihop_decomposed_mlx.yml",
               "cuda": f"{SR}/multihop_decomposed_cuda.yml"},
              n_queries=12, timeout_s=5400, serves=["E4 multi-hop"]),
    # --- E3 (committed VQA; M2-only by design) ------------------------------
    PilotCell("e3_vqa_a", {"mlx": f"{PC}/multimodal_vqa_mapping_a.yml"},
              n_queries=15, serves=["E3 mapping-A service time", "E3 warm-up k"]),
    PilotCell("e3_vqa_b", {"mlx": f"{PC}/multimodal_vqa_mapping_b.yml"},
              n_queries=15,
              serves=["E3 mapping-B service time", "ANE first-call outlier"]),
    # --- E3' components (also AMC calibration loads) ------------------------
    PilotCell("e3p_fg_decode",
              {"mlx": f"{PC}/pilots/decode_9b_mlx.yml",
               "cuda": f"{PC}/pilots/decode_9b_cuda.yml"},
              n_queries=15, serves=["E3' foreground baseline", "AMC GPU calibration"]),
    PilotCell("e3p_c1_rmax", {"mlx": f"{PC}/pilots/clip_gpu_encode.yml"},
              n_queries=100, serves=["E3' co-runner C1 R_max"]),
    PilotCell("e3p_c2_rmax", {"mlx": f"{PC}/pilots/clip_ane_encode.yml"},
              n_queries=100,
              serves=["E3' co-runner C2 R_max", "AMC ANE calibration"]),
    PilotCell("e3p_c3_rmax", {}, n_queries=100,
              serves=["E3' co-runner C3 R_max"],
              blocked_on="CPU memory-streaming stage (CONTENTION_EXPERIMENTS_REDESIGN.md §1)"),
    # --- E5 ------------------------------------------------------------------
    PilotCell("e5_resnet_serial",
              {"mlx": f"{PC}/mlperf/resnet_inference.yml",
               "cuda": f"{PC}/mlperf/resnet_inference.yml"},
              n_queries=200, serves=["E5 Server λ", "MultiStream interval"]),
    # --- E6 / E6' foregrounds -------------------------------------------------
    PilotCell("e6_fg_effnet",
              {"mlx": f"{PC}/torchvision_inference.yml",
               "cuda": f"{PC}/torchvision_inference.yml"},
              n_queries=100, serves=["committed-E6 foreground λ"]),
    PilotCell("e6p_fg_ragserve",
              {"mlx": f"{PC}/rag_serve_plain.yml",
               "cuda": f"{PC}/rag_serve_plain_cuda.yml"},
              n_queries=20, serves=["E6' foreground λ (R-E6-HEADROOM)"]),
    PilotCell("e6p_bg_index_rmax", {}, n_queries=0,
              serves=["E6' background intensity axis"],
              blocked_on="EmbedStage + ChromaIndexer (CONTENTION_EXPERIMENTS_REDESIGN.md §2)"),
]


# ---------------------------------------------------------------------------

def capture_env(device: str) -> None:
    lines = [
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"git_commit: {_git_sha()}",
        f"device_arg: {device}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"perf_counter: {time.get_clock_info('perf_counter')}",
    ]
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize",
                    "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"):
            try:
                val = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                     text=True, timeout=5).stdout.strip()
                lines.append(f"{key}: {val}")
            except Exception:
                pass
    else:
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            lines.append(f"nvidia-smi: {gpu}")
        except Exception:
            pass
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[env] wrote {ENV_FILE}")


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain",
                                "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_pilot_config(base_cfg_path: Path, n_queries: int, timeout_s: int,
                      device: str) -> str:
    """Copy the base config with the loadgen block overridden to serial pilot
    mode (closed-loop, one in flight). On the mlx device, device-agnostic
    torch configs (E5 resnet, E6 torchvision) get their stage `device: cuda`
    remapped to `mps` (same device patching run_modularity.py does).
    Returns the temp file path."""
    doc = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    for pipe in doc["pipelines"]:
        pipe["loadgen"] = {
            "component": "loadgen.OfflineLoadScheduler",
            "queue_depth": 4,
            "max_queries": n_queries,
            "timeout": timeout_s,
            "config": {"rate": 0},
        }
        if device == "mlx":
            for stage in pipe.get("stages", []):
                cfg = stage.get("config") or {}
                if cfg.get("device") == "cuda":
                    cfg["device"] = "mps"
    fd, tmp = tempfile.mkstemp(suffix=".yml", prefix="pilot_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return tmp


def run_cell(cell: PilotCell, device: str, force: bool,
             runs_override: int | None, n_override: int | None) -> bool:
    cfg_rel = cell.config_for(device)
    if cfg_rel is None:
        print(f"[skip] {cell.id}: no {device} config")
        return True
    base = REPO_ROOT / cfg_rel
    if not base.exists():
        print(f"[FAIL] {cell.id}: missing config {cfg_rel}")
        return False
    runs = runs_override or cell.runs
    n_q = n_override or cell.n_queries
    ok = True
    for r in range(1, runs + 1):
        label = f"pilot_{cell.id}_{device}_r{r}"
        target = RESULTS_DIR / f"{label}.csv"
        if target.exists() and not force:
            print(f"[skip] {label} (exists)")
            continue
        tmp = make_pilot_config(base, n_q, cell.timeout_s, device)
        env = dict(os.environ, CHOREO_DISABLE_TRACING="1")
        print(f"[run ] {label}  (config {cfg_rel}, N={n_q})")
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "main.py", tmp, "-p", "0", "--label", label],
                cwd=REPO_ROOT, env=env, timeout=cell.timeout_s,
                capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {label}: timeout after {cell.timeout_s}s")
            ok = False
            os.unlink(tmp)
            continue
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        src = GLOBAL_RESULTS / f"{label}.csv"
        if proc.returncode != 0 or not src.exists():
            print(f"[FAIL] {label}: rc={proc.returncode}")
            print("  stderr tail:", "\n  ".join(proc.stderr.splitlines()[-5:]))
            ok = False
            continue
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), target)
        for extra in (f"{label}_arrivals.csv", f"{label}_outputs.jsonl"):
            p = GLOBAL_RESULTS / extra
            if p.exists():
                shutil.move(str(p), RESULTS_DIR / extra)
        print(f"[done] {label} in {time.time() - t0:.0f}s -> {target.name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["mlx", "cuda"],
                    default="mlx" if platform.system() == "Darwin" else "cuda")
    ap.add_argument("--only", default=None, help="glob over cell ids")
    ap.add_argument("--runs-override", type=int, default=None)
    ap.add_argument("--n-override", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cells = [c for c in CELLS
             if args.only is None or fnmatch.fnmatch(c.id, args.only + "*")]
    if args.list:
        for c in CELLS:
            state = (f"BLOCKED on {c.blocked_on}" if c.blocked_on
                     else ", ".join(sorted(c.config)))
            print(f"  {c.id:24s} N={c.n_queries:<4d} R={c.runs}  [{state}]  "
                  f"serves: {'; '.join(c.serves)}")
        return 0
    if not cells:
        sys.exit(f"no cells match --only {args.only}")

    capture_env(args.device)
    failures = []
    for c in cells:
        if c.blocked_on:
            print(f"[blocked] {c.id}: {c.blocked_on}")
            continue
        if not run_cell(c, args.device, args.force,
                        args.runs_override, args.n_override):
            failures.append(c.id)
    if failures:
        print(f"\nFAILED cells: {failures}")
        return 1
    print("\nAll requested pilot cells complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
