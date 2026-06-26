#!/usr/bin/env python3
"""Driver for the framework-overhead (NoOp) microbenchmark.

Runs the full depth + payload sweep, R times each, in two arms:

  * ``t0`` (tracing OFF)  -- sets ``CHOREO_DISABLE_TRACING=1`` so MLflow spans
    become no-ops. This isolates the framework's CORE dispatch cost
    (thread wake + queue hand-off + CSV log) -- tens of microseconds, flat in
    depth, and backend-independent (reproducible on any machine).
  * ``t1`` (tracing ON)   -- the profiling layer the case studies actually pay
    (3 MLflow spans per stage per query). Characterised so the write-up can
    frame it via overhead-in-context (negligible vs. seconds-long real ML
    stages). NOTE: invoked through ``main.py -p 0`` there is no RadT MLflow
    server, so by default this driver points MLflow at a local file store with
    ASYNC export to avoid the sync-export-with-no-backend pathology (~seconds
    per span). Those are async-export numbers; the canonical "as-used" sync
    numbers come from the orchestrated path on a machine with RadT's backend.

Each run is one ``python main.py <config> -p 0 --label <label>`` subprocess.
``main.py`` hard-codes its output to ``evaluation/results/<label>.csv`` (relative
to CWD); we curate-then-move each CSV into this experiment's ``results/`` dir so
the shared dir stays clean (same pattern the self_rag study uses).

Labels encode the full cell so files never collide and the analyzer can parse
them: ``noop_d{D}_s{S}_m{ref|copy}_t{0|1}_r{R}``.

The driver is idempotent/resumable: a (cell, arm, run) whose target CSV already
exists is skipped unless ``--force``.

Run it from inside the project's conda env on the M2 Pro, e.g.::

    python evaluation/overheads/framework_overhead/run_matrix.py --runs 5
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

# --- paths -------------------------------------------------------------------
_HERE = os.path.abspath(os.path.dirname(__file__))
# framework_overhead -> overheads -> evaluation -> repo root
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_CONFIGS_DIR = os.path.join(_HERE, "configs")
_RESULTS_DIR = os.path.join(_HERE, "results")
_SHARED_RESULTS = os.path.join(_REPO_ROOT, "evaluation", "results")
_GENERATOR = os.path.join(_HERE, "noop_chain_generator.py")

# --- the matrix --------------------------------------------------------------
DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 50, 64, 100]
PAYLOAD_SIZES = [0, 1024, 1048576, 10485760]  # 0, 1 KiB, 1 MiB, 10 MiB
PAYLOAD_MODES = ["ref", "copy"]
PAYLOAD_DEPTH = 10


def config_path(depth: int, size: int, mode: str) -> str:
    return os.path.join(_CONFIGS_DIR, f"noop_depth_{depth}_size_{size}_mode_{mode}.yml")


def label(depth: int, size: int, mode: str, trace_on: bool, run: int) -> str:
    return f"noop_d{depth}_s{size}_m{mode}_t{1 if trace_on else 0}_r{run}"


def ensure_config(depth: int, size: int, mode: str, max_queries: int, force: bool) -> str:
    """(Re)generate the config via noop_chain_generator.py.

    Always regenerates: configs are deterministic generated artifacts and the
    filename does not encode ``max_queries``, so a stale config would silently
    override the requested query count.
    """
    path = config_path(depth, size, mode)
    subprocess.run(
        [sys.executable, _GENERATOR,
         "--depths", str(depth), "--sizes", str(size), "--modes", mode,
         "--max-queries", str(max_queries), "--out-dir", _CONFIGS_DIR],
        cwd=_REPO_ROOT, check=True,
    )
    return path


def capture_env(max_queries: int, runs: int) -> str:
    """Write perf_counter resolution + machine SKU to run_matrix_env.txt."""
    lines = []
    ci = time.get_clock_info("perf_counter")
    lines.append("# framework-overhead run environment")
    lines.append(f"timestamp_wall = {time.time():.6f}")
    lines.append(f"runs_per_cell = {runs}")
    lines.append(f"max_queries = {max_queries}")
    lines.append("")
    lines.append("[perf_counter]")
    lines.append(f"resolution = {ci.resolution}")
    lines.append(f"monotonic = {ci.monotonic}")
    lines.append(f"adjustable = {ci.adjustable}")
    lines.append(f"implementation = {ci.implementation}")
    lines.append("")
    lines.append("[platform]")
    lines.append(f"python = {platform.python_version()}")
    lines.append(f"system = {platform.system()} {platform.release()}")
    lines.append(f"machine = {platform.machine()}")
    lines.append(f"processor = {platform.processor()}")
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize",
                    "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"):
            try:
                val = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
                lines.append(f"{key} = {val}")
            except Exception:  # pylint: disable=broad-except
                pass
    text = "\n".join(lines) + "\n"
    out = os.path.join(_HERE, "run_matrix_env.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return out


def run_cell(depth, size, mode, trace_on, runs, max_queries, timeout,
             force, mlflow_safe):
    """Run R repetitions of one (cell, arm); curate CSVs into results/."""
    cfg = ensure_config(depth, size, mode, max_queries, force)
    for r in range(1, runs + 1):
        lab = label(depth, size, mode, trace_on, r)
        target = os.path.join(_RESULTS_DIR, lab + ".csv")
        if os.path.exists(target) and not force:
            print(f"[skip] {lab} (exists)")
            continue

        env = os.environ.copy()
        tmpdir = None
        if not trace_on:
            env["CHOREO_DISABLE_TRACING"] = "1"
        elif mlflow_safe:
            # Avoid the sync-export-with-no-backend pathology on the -p 0 path:
            # async export to a throwaway local file store. Point MLflow at a
            # NON-existent subdir so its FileStore initializes the store
            # (incl. the default experiment); an empty pre-made dir would raise
            # "Could not find experiment with ID 0".
            env["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
            tmpdir = tempfile.mkdtemp(prefix="noop_mlruns_")
            env["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(tmpdir, "store")

        print(f"[run ] {lab}")
        rc = None
        try:
            proc = subprocess.run(
                [sys.executable, "main.py", cfg, "-p", "0", "--label", lab],
                cwd=_REPO_ROOT, env=env, timeout=timeout,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {lab}: timed out after {timeout}s")
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

        produced = os.path.join(_SHARED_RESULTS, lab + ".csv")
        if os.path.exists(produced):
            os.makedirs(_RESULTS_DIR, exist_ok=True)
            shutil.move(produced, target)
            print(f"[ ok ] {lab} -> results/{lab}.csv (rc={rc})")
        else:
            print(f"[FAIL] {lab}: no CSV produced at {produced} (rc={rc})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5, help="repetitions per cell (R)")
    ap.add_argument("--max-queries", type=int, default=101,
                    help="queries per run (1 discarded as warm-up)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-run subprocess timeout (s)")
    ap.add_argument("--arms", choices=["off", "on", "both"], default="both",
                    help="tracing arms to run")
    ap.add_argument("--depths", type=int, nargs="+", default=DEPTHS,
                    help="depth sweep values")
    ap.add_argument("--no-payload", action="store_true",
                    help="skip the payload (zero-copy) sweep")
    ap.add_argument("--no-mlflow-safe", action="store_true",
                    help="for the tracing-ON arm, do NOT redirect MLflow to a "
                         "local async store (use the ambient MLflow config)")
    ap.add_argument("--force", action="store_true",
                    help="re-run and regenerate configs even if outputs exist")
    args = ap.parse_args()

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    capture_env(args.max_queries, args.runs)

    arms = {"off": [False], "on": [True], "both": [False, True]}[args.arms]
    mlflow_safe = not args.no_mlflow_safe

    # depth sweep (size 0, ref) -- both arms
    depth_cells = [(d, 0, "ref") for d in args.depths]
    # payload sweep (depth 10, sizes x modes) -- OFF arm only (zero-copy is an
    # architecture property; tracing cost is payload-independent)
    payload_cells = [] if args.no_payload else [
        (PAYLOAD_DEPTH, s, m) for s in PAYLOAD_SIZES for m in PAYLOAD_MODES
    ]

    for trace_on in arms:
        cells = list(depth_cells)
        if not trace_on:
            # add payload cells, de-duplicating the shared (10, 0, ref) point
            for c in payload_cells:
                if c not in cells:
                    cells.append(c)
        for depth, size, mode in cells:
            run_cell(depth, size, mode, trace_on, args.runs, args.max_queries,
                     args.timeout, args.force, mlflow_safe)

    print("\nDone. CSVs in:", _RESULTS_DIR)


if __name__ == "__main__":
    main()
