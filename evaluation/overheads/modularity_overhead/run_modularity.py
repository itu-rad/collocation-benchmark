#!/usr/bin/env python3
"""Driver for the modularity-overhead experiment.

For one device, runs the hand-written PyTorch baseline (R repetitions) and the
equivalent Choreo pipeline (R repetitions x {tracing off, on}), so the analyzers
can show the framework wrapper adds negligible per-step overhead vs the monolith.

  * baseline: `python baseline_finetune.py --no-radt ...` -> a true zero-framework
    control. Writes straight into this experiment's results/ (via --label).
  * choreo t0 (tracing off): `CHOREO_DISABLE_TRACING=1` -> core framework overhead.
  * choreo t1 (tracing on): the MLflow span layer; uses run_matrix's mlflow-safe
    local async store to dodge the sync-export-with-no-backend pathology on the
    `-p 0` path (numbers are an async proxy; see modularity_overhead.md).

The Choreo config's `device` (hardcoded mps) and `max_queries` are patched into a
temp config per device/run. main.py writes `evaluation/results/<label>.csv`; we
curate-move it into results/. Labels: `mod_baseline_d{dev}_r{R}`,
`mod_choreo_t{0|1}_d{dev}_r{R}`. Idempotent/resumable (skip existing unless --force).

Run inside the project's torch+cuda env (e.g. benchmark_engines):

    python evaluation/overheads/modularity_overhead/run_modularity.py --device cuda --runs 5
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_RESULTS_DIR = os.path.join(_HERE, "results")
_SHARED_RESULTS = os.path.join(_REPO_ROOT, "evaluation", "results")
_CONFIG = os.path.join(_HERE, "configs", "torchvision_training.yml")
_BASELINE = os.path.join(_HERE, "baseline_finetune.py")

# The canonical anchor cell (the single published E2 point). Used when --cells is
# not given, so no-arg behavior reproduces the original workload and filenames.
_CANONICAL_CELL = {
    "model": "efficientnet_v2_s",
    "weights": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    "batch": 8,
    "tag": None,  # None -> legacy label with no cell suffix
}


def load_cells(path, default_max_batches, default_runs):
    """Load the scale-sweep manifest (a YAML list of cells) or fall back to the
    single canonical anchor cell. Each cell: {model, weights, batch, tag?,
    max_batches?, runs?}; missing max_batches/runs inherit the CLI defaults."""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cells = raw["cells"] if isinstance(raw, dict) else raw
    else:
        cells = [dict(_CANONICAL_CELL)]
    for c in cells:
        for k in ("model", "weights", "batch"):
            if k not in c:
                raise SystemExit(f"[cells] every cell needs '{k}': {c}")
        c.setdefault("tag", None)
        c.setdefault("max_batches", default_max_batches)
        c.setdefault("runs", default_runs)
    return cells


RADT_PIN = "0.2.29"  # async_tracing fork @9dda7b8 (itu-rad/radt; environments/*.yaml pin)


def check_environment(device):
    """Hard gate: the tracing-ON arm is only meaningful on the pinned
    async_tracing radt (0.2.29 @9dda7b8), and a CPU-only torch wheel on a
    cuda device silently measures nothing. Both have burned collection time
    (2026-07-13). Refuse to run on a mismatched env."""
    fatal = []
    try:
        import radt
        if radt.__version__ != RADT_PIN:
            fatal.append(f"radt {radt.__version__} != pinned {RADT_PIN} "
                         "(async_tracing @9dda7b8) — create the env from "
                         "environments/macos.yaml (benchmark_macos) / nvidia.yaml")
    except ImportError:
        fatal.append("radt not importable")
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            fatal.append(f"torch {torch.__version__} has no CUDA — CPU-only "
                         "wheel; rebuild from environments/nvidia.yaml")
    except ImportError:
        fatal.append("torch not importable")
    if fatal:
        for f in fatal:
            print(f"[env FATAL] {f}", file=sys.stderr)
        sys.exit(3)


def capture_env(device, runs, max_batches, num_workers=0, cooldown=0):
    check_environment(device)
    import radt
    lines = ["# modularity-overhead run environment",
             f"radt = {radt.__version__}",
             f"timestamp_wall = {time.time():.6f}",
             f"device = {device}", f"runs_per_cell = {runs}",
             f"max_batches = {max_batches}", f"num_workers = {num_workers}",
             f"cooldown_s = {cooldown}", "interleaved = true", ""]
    ci = time.get_clock_info("perf_counter")
    lines += ["[perf_counter]", f"resolution = {ci.resolution}",
              f"monotonic = {ci.monotonic}", f"implementation = {ci.implementation}", ""]
    lines += ["[platform]", f"python = {platform.python_version()}",
              f"system = {platform.system()} {platform.release()}",
              f"machine = {platform.machine()}"]
    try:
        import torch
        lines += [f"torch = {torch.__version__}",
                  f"torch_cuda = {torch.version.cuda}"]
        if device == "cuda" and torch.cuda.is_available():
            lines.append(f"cuda_device = {torch.cuda.get_device_name(0)}")
    except Exception as e:  # pylint: disable=broad-except
        lines.append(f"torch = unavailable ({e})")
    if device == "cuda":
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"], text=True).strip()
            lines.append(f"nvidia_smi = {smi}")
        except Exception:  # pylint: disable=broad-except
            pass
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            try:
                lines.append(f"{key} = " + subprocess.check_output(
                    ["sysctl", "-n", key], text=True).strip())
            except Exception:  # pylint: disable=broad-except
                pass
    text = "\n".join(lines) + "\n"
    with open(os.path.join(_HERE, "run_modularity_env.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def cell_suffix(cell):
    """Filename suffix encoding the scale cell. A cell with no `tag` (the default
    canonical EfficientNetV2-S @ batch 8) keeps the LEGACY label (no suffix), so
    existing published CSVs / analyses reproduce byte-for-byte. Sweep cells carry
    an explicit tag -> `_m{tag}_b{batch}`."""
    tag = cell.get("tag")
    if not tag:
        return ""
    return f"_m{tag}_b{cell['batch']}"


def make_choreo_config(device, max_queries, num_workers, cell):
    """Write a temp Choreo config with device + max_queries + num_workers AND the
    cell's model / weights / batch patched in. Model weights are left unset
    (random init) exactly as the canonical training config, so the wrapped arm
    matches the bare baseline (which builds with weights=None). The Choreo stage
    `name:` fields are deliberately NOT changed, so modularity_lib's stage-name
    constants and the existing 47 CSVs keep parsing."""
    with open(_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipe = cfg["pipelines"][0]
    pipe["loadgen"]["max_queries"] = max_queries
    for stage in pipe["stages"]:
        if "TorchVisionClassification" in stage.get("component", ""):
            stage["config"]["device"] = device
            stage["config"]["model"]["component"] = f"torchvision.models.{cell['model']}"
        if "TorchVisionDataLoader" in stage.get("component", ""):
            stage["config"]["num_workers"] = num_workers
            stage["config"]["batch_size"] = cell["batch"]
            # dataset.weights drives the preprocessing transform (input resolution),
            # so both arms feed the model the same shape — match the cell's weights.
            stage["config"]["dataset"]["weights"] = cell["weights"]
    fd, path = tempfile.mkstemp(prefix="mod_choreo_", suffix=".yml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def exec_baseline(device, max_batches, num_workers, timeout, r, force, cell):
    lab = f"mod_baseline{cell_suffix(cell)}_d{device}_r{r}"
    target = os.path.join(_RESULTS_DIR, lab + ".csv")
    if os.path.exists(target) and not force:
        print(f"[skip] {lab}")
        return
    if os.path.exists(target):
        os.remove(target)  # belt: never append onto a stale baseline CSV
    print(f"[run ] {lab}")
    try:
        rc = subprocess.run(
            [sys.executable, _BASELINE, "--device", device,
             "--model", cell["model"], "--weights", cell["weights"],
             "--batch-size", str(cell["batch"]),
             "--num-workers", str(num_workers),
             "--max-batches", str(max_batches), "--label", lab,
             "--no-radt", "--run", str(r)],
            cwd=_REPO_ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {lab}: timeout")
        return
    ok = os.path.exists(target) and os.path.getsize(target) > 0
    print(f"[{'ok ' if ok else 'FAIL'}] {lab} (rc={rc})")


def exec_choreo(cfg, device, mode, timeout, r, force, online, experiment, cell):
    # mode: "off" (t0, tracing off) | "on" (t1, current in-process mlflow) |
    #       "proc" (t2, radt-owned out-of-process span export)
    _t = {"off": 0, "on": 1, "proc": 2}[mode]
    lab = f"mod_choreo_t{_t}{cell_suffix(cell)}_d{device}_r{r}"
    target = os.path.join(_RESULTS_DIR, lab + ".csv")
    if os.path.exists(target) and not force:
        print(f"[skip] {lab}")
        return
    env = os.environ.copy()
    tmpdir = None
    if online:
        # ONLINE: inherit the ambient MLflow server (res17, from the conda env
        # credential vars). The tracing-ON arm uses ASYNC span export, so the
        # measured overhead is the REAL production tracing cost, not a local
        # file-store cost. Do NOT override MLFLOW_TRACKING_URI. RADT_PRESENT
        # makes main.py end_run() + drain the async span queue on exit; NO
        # RADT_LISTENER_* vars are set, so NO listener processes spawn (zero
        # measurement perturbation).
        env["RADT_PRESENT"] = "True"
        if mode == "off":
            env["CHOREO_DISABLE_TRACING"] = "1"
        elif mode == "proc":
            env["CHOREO_PROC_TRACE"] = "1"
            env["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
        else:
            env["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
    else:
        # Offline default: throwaway per-run local file store for BOTH arms. This
        # overrides any ambient MLFLOW_TRACKING_URI (e.g. a shared sqlite DB
        # another user may migrate out from under us -> schema-mismatch crashes)
        # and touches no one else's MLflow state. The non-existent "store" subdir
        # lets MLflow's FileStore initialize the default experiment.
        tmpdir = tempfile.mkdtemp(prefix="mod_mlruns_")
        env["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(tmpdir, "store")
        if mode == "off":
            env["CHOREO_DISABLE_TRACING"] = "1"
        elif mode == "proc":
            env["CHOREO_PROC_TRACE"] = "1"
            env["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
        else:
            env["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
    cmd = [sys.executable, "main.py", cfg, "-p", "0", "--label", lab]
    if online or mode == "proc":
        cmd += ["-e", str(experiment)]
    print(f"[run ] {lab}")
    try:
        rc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {lab}: timeout")
        rc = None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    produced = os.path.join(_SHARED_RESULTS, lab + ".csv")
    if os.path.exists(produced) and os.path.getsize(produced) > 0:
        shutil.move(produced, target)
        print(f"[ ok ] {lab} -> results/ (rc={rc})")
    else:
        print(f"[FAIL] {lab}: no/empty CSV (rc={rc})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cuda", "mps"], required=True,
                    help="cpu is refused (no GPU sync -> meaningless step timing)")
    ap.add_argument("--cells", default=None,
                    help="path to a scale-sweep manifest (YAML list of cells: "
                         "model/weights/batch [+tag/max_batches/runs]). Omit to run "
                         "the single canonical EfficientNetV2-S @ batch-8 anchor. "
                         "See configs/scale_sweep.yml.")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=1100,
                    help="steps per run - one continuous epoch (200 dropped as "
                         "warm-up by the analyzers; canonical value, see "
                         "evaluation/pilots/knobs.yml R-WARMUP); baseline and "
                         "Choreo use the same count for an apples-to-apples N")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers for BOTH arms. Default 0 removes the "
                         "concurrent-prefetch confound so the per-step metric "
                         "isolates the framework wrapper, not data-path contention.")
    ap.add_argument("--cooldown", type=int, default=0,
                    help="seconds to idle the GPU before each run. Default 0: each "
                         "run is a single continuous epoch whose per-step time is "
                         "flat at steady state (verified), so no cooldown is needed. "
                         "Set >0 only to probe cold-start effects.")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--arms", choices=["off", "on", "both"], default="both")
    ap.add_argument("--proc-trace", action="store_true",
                    help="add a 4th choreo arm (t2): radt-owned OUT-OF-PROCESS span "
                         "export (CHOREO_PROC_TRACE=1). The prototype under test.")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-choreo", action="store_true")
    ap.add_argument("--online", action="store_true",
                    help="log Choreo runs to the ambient MLflow server (res17, from "
                         "the conda env credential vars) with ASYNC span export — the "
                         "representative production tracing cost. Requires the server "
                         "credentials to be set as conda env vars.")
    ap.add_argument("--experiment", type=int, default=138,
                    help="MLflow experiment id for --online runs (default 138)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    capture_env(args.device, args.runs, args.max_batches,
                num_workers=args.num_workers, cooldown=args.cooldown)

    cells = load_cells(args.cells, args.max_batches, args.runs)
    arms = {"off": ["off"], "on": ["on"], "both": ["off", "on"]}[args.arms]
    if args.proc_trace:
        arms = arms + ["proc"]
    print(f"[cells] {len(cells)} scale cell(s): "
          + ", ".join(f"{c['model']}@b{c['batch']}(r{c['runs']},mb{c['max_batches']})"
                      for c in cells))

    # Each cell is an independent (model, batch) workload; within a cell the arms
    # are interleaved run-by-run so every arm sees the same thermal trajectory
    # (the paired difference relies on shared per-run conditions). Cells run to
    # completion in sequence.
    for cell in cells:
        mb, runs = cell["max_batches"], cell["runs"]
        print(f"\n=== cell {cell['model']} @ batch {cell['batch']} "
              f"(tag={cell['tag']}, max_batches={mb}, runs={runs}) ===")
        cfg = make_choreo_config(args.device, mb, args.num_workers, cell) \
            if not args.no_choreo else None
        try:
            for r in range(1, runs + 1):
                if not args.no_baseline:
                    if args.cooldown:
                        print(f"[cool] {args.cooldown}s"); time.sleep(args.cooldown)
                    exec_baseline(args.device, mb, args.num_workers,
                                  args.timeout, r, args.force, cell)
                if not args.no_choreo:
                    for mode in arms:
                        if args.cooldown:
                            print(f"[cool] {args.cooldown}s"); time.sleep(args.cooldown)
                        exec_choreo(cfg, args.device, mode, args.timeout, r,
                                    args.force, args.online, args.experiment, cell)
        finally:
            if cfg:
                os.unlink(cfg)

    print("\nDone. CSVs in:", _RESULTS_DIR)


if __name__ == "__main__":
    main()
