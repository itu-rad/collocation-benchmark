"""DRAM-bandwidth trace for benchmark runs (Apple Silicon, no root).

Wraps the compiled AMC sampler (`amc_bandwidth_sampler.c`, auto-built on first
use) that reads the Apple Memory Controller's per-requestor DRAM byte counters
via IOReport and streams a CSV time series: per-interval bytes moved by
CPU / GPU / ANE / other, plus total GB/s. See
CONTENTION_EXPERIMENTS_REDESIGN.md (E3'/E6' "Counters") for provenance and the
calibration notes.

CLI (standalone, e.g. alongside a manual run):

    python scripts/amc_bandwidth_sampler.py --label my_run --interval 0.5
    python scripts/amc_bandwidth_sampler.py --label my_run --duration 60 --summary

writes ``evaluation/results/<label>_bandwidth.csv`` (or use --out PATH).

From an experiment driver:

    from amc_bandwidth_sampler import AMCBandwidthSampler

    with AMCBandwidthSampler(label="noop_d10_r1", interval=0.5):
        run_benchmark()                       # CSV rows appear as it runs
    # -> evaluation/results/noop_d10_r1_bandwidth.csv

Timestamps are wall-clock seconds (same base as the framework CSV's
``%(created)f`` column), so bandwidth rows join the timing trace directly.
"""

from __future__ import annotations

import argparse
import csv
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC = SCRIPTS_DIR / "amc_bandwidth_sampler.c"
BIN = SCRIPTS_DIR / ".build" / "amc_bandwidth_sampler"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "results"


def ensure_built() -> Path:
    """Compile the sampler if the binary is missing or older than the source."""
    if platform.system() != "Darwin":
        raise RuntimeError("AMC sampler is Apple-Silicon-only; on the NVIDIA DUT "
                           "use dcgmi/nvidia-smi (see preflight_bandwidth_counters.sh)")
    if BIN.exists() and BIN.stat().st_mtime >= SRC.stat().st_mtime:
        return BIN
    BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["clang", "-O2", "-o", str(BIN), str(SRC), "-framework", "CoreFoundation"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sampler build failed:\n{proc.stderr}")
    return BIN


class AMCBandwidthSampler:
    """Run the sampler as a sidecar process for the duration of a with-block."""

    def __init__(self, label: str | None = None, out: str | Path | None = None,
                 interval: float = 0.5, raw: bool = False):
        if out is None:
            if label is None:
                raise ValueError("pass label= or out=")
            out = DEFAULT_RESULTS_DIR / f"{label}_bandwidth.csv"
        self.out = Path(out)
        self.interval = interval
        self.raw = raw
        self._proc: subprocess.Popen | None = None

    def start(self) -> "AMCBandwidthSampler":
        binary = ensure_built()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(binary), "-i", str(int(self.interval * 1000)), "-o", str(self.out)]
        if self.raw:
            cmd.append("--raw")
        self._proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
        # Fail fast if the machine has no AMC channels (exit code 3).
        time.sleep(0.2)
        if self._proc.poll() is not None:
            rc = self._proc.returncode
            self._proc = None
            raise RuntimeError(f"sampler exited immediately (rc={rc}); "
                               "run scripts/preflight_bandwidth_counters.sh")
        return self

    def stop(self) -> Path:
        if self._proc is not None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        return self.out

    def __enter__(self) -> "AMCBandwidthSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def summarize(csv_path: Path) -> dict:
    """Mean/max GB/s overall and mean GB/s per engine bucket."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (TypeError, ValueError):
                continue
    if not rows:
        return {}
    dt_total = sum(r["dt_s"] for r in rows)
    per_engine = {}
    for eng in ("cpu", "gpu", "ane", "other"):
        traffic = sum(r[f"{eng}_rd"] + r[f"{eng}_wr"] for r in rows)
        per_engine[eng] = traffic / dt_total / 1e9
    return {
        "samples": len(rows),
        "duration_s": dt_total,
        "mean_total_gbps": sum(r["total_gbps"] * r["dt_s"] for r in rows) / dt_total,
        "max_total_gbps": max(r["total_gbps"] for r in rows),
        **{f"mean_{k}_gbps": v for k, v in per_engine.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", help="output name -> evaluation/results/<label>_bandwidth.csv")
    ap.add_argument("--out", help="explicit output CSV path (overrides --label)")
    ap.add_argument("--interval", type=float, default=0.5, help="sampling period in seconds")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after N seconds (default: run until Ctrl-C)")
    ap.add_argument("--raw", action="store_true", help="also dump per-channel deltas")
    ap.add_argument("--summary", action="store_true", help="print a summary when done")
    args = ap.parse_args()

    if not args.label and not args.out:
        ap.error("pass --label or --out")
    sampler = AMCBandwidthSampler(label=args.label, out=args.out,
                                  interval=args.interval, raw=args.raw)
    sampler.start()
    print(f"sampling -> {sampler.out} (Ctrl-C to stop)", file=sys.stderr)
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()

    if args.summary:
        for k, v in summarize(sampler.out).items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
