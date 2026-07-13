"""Shared parsing + statistics for the hyperparameter pilot protocol (PAPER_TODO §2.6).

Pilots are short SERIAL runs (closed-loop OfflineLoadScheduler, one query in
flight) per (workload, device). This module turns their trace CSVs into the two
rule inputs everything else derives from:

  * serial service time  — per-query pipeline latency (median/mean/p95/cv)
  * warm-up horizon      — rolling-median flatness point k*, fixed knob k = 2*k*

plus the knobs.yml reader used by drivers/analyzers and the parser for the
intended-arrivals sidecar (``<label>_arrivals.csv``) that verifies queue_depth
never blocked and the realized rate matched the intended one.

Per-query latency comes from the trace CSV's pipeline rows keyed by
(epoch, batch) — the unique per-query key across ALL workloads (NoOp runs vary
epoch at batch 0; Self-RAG/case-study runs keep epoch constant and vary batch;
training pipelines vary both). Same convention as
modularity_lib.parse_pipeline_latency.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent

KNOBS_PATH = _HERE / "knobs.yml"
RESULTS_DIR = _HERE / "results"

NS = 1e9


# ---------------------------------------------------------------------------
# Per-query latency (seconds) from a trace CSV
# ---------------------------------------------------------------------------

def per_query_latencies(csv_path) -> list[float]:
    """Per-query pipeline latencies (seconds), in submission order.

    Pairs pipeline-row start/end perf_counter timestamps keyed by
    (epoch, batch) — unique per query for every workload (NoOp varies epoch,
    case studies vary batch, training varies both).
    """
    starts, ends = {}, {}
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if (len(parts) < 10 or not parts[2].startswith("pipeline -")
                    or parts[3] != "run"):
                continue
            try:
                key = (int(parts[7]), int(parts[8]))
                perf = int(parts[-1])
            except (IndexError, ValueError):
                continue
            (starts if parts[4] == "start" else ends)[key] = perf
    return [(ends[k] - starts[k]) / NS for k in sorted(starts) if k in ends]


# ---------------------------------------------------------------------------
# Warm-up detection (rolling-median flatness)
# ---------------------------------------------------------------------------

@dataclass
class WarmupResult:
    k_star: int              # detected flatness onset (0 = flat from the start)
    k_fixed: int             # pre-registered knob: 2 * k_star (>=1), capped N/3
    window: int
    epsilon: float
    converged: bool          # tail itself flat AND k_star found within N/3
    outlier_idxs: list[int] = field(default_factory=list)  # first-call class
    tail_median: float = float("nan")
    n: int = 0
    note: str = ""


def _rolling_median(x: list[float], w: int) -> list[float]:
    return [statistics.median(x[i:i + w]) for i in range(len(x) - w + 1)]


def detect_warmup(x: list[float], window: int = 5, epsilon: float = 0.10,
                  outlier_factor: float = 5.0) -> WarmupResult:
    """Rolling-median flatness warm-up detector (one implementation for all
    experiments; window=5/eps=0.10 for query-scale series, window=51/eps=0.05
    for step-scale series like E2).

    Tail reference M = median of the second half (asserted flat itself);
    k* = first index whose rolling median stays within eps*M through the end;
    fixed knob k = max(1, 2*k*), capped at N//3 (else inconclusive).
    First-call outliers (> outlier_factor * M, the ANE-compile class) are
    reported separately — they are excluded AND reported, never folded into k.
    """
    n = len(x)
    if n < max(2 * window, 8):
        return WarmupResult(0, 1, window, epsilon, False, [], float("nan"), n,
                            note=f"series too short (n={n})")
    tail = x[n // 2:]
    m_ref = statistics.median(tail)
    if m_ref <= 0:
        return WarmupResult(0, 1, window, epsilon, False, [], m_ref, n,
                            note="non-positive tail median")
    outliers = [i for i, v in enumerate(x) if v > outlier_factor * m_ref]

    # Tail flatness check (on the outlier-free tail rolling medians).
    tail_clean = [v for i, v in enumerate(tail) if (n // 2 + i) not in set(outliers)]
    if len(tail_clean) >= window:
        tail_rm = _rolling_median(tail_clean, window)
        if any(abs(m - m_ref) > epsilon * m_ref for m in tail_rm):
            return WarmupResult(0, 1, window, epsilon, False, outliers, m_ref, n,
                                note="tail not flat — pilot inconclusive, extend N")

    rm = _rolling_median(x, window)
    k_star = None
    for i in range(len(rm)):
        if all(abs(m - m_ref) <= epsilon * m_ref for m in rm[i:]):
            k_star = i
            break
    if k_star is None:
        return WarmupResult(0, 1, window, epsilon, False, outliers, m_ref, n,
                            note="no flatness onset found")
    k_fixed = max(1, 2 * k_star)
    if k_fixed > n // 3:
        return WarmupResult(k_star, k_fixed, window, epsilon, False, outliers,
                            m_ref, n, note=f"k_fixed={k_fixed} > N/3 — extend N")
    return WarmupResult(k_star, k_fixed, window, epsilon, True, outliers, m_ref, n)


# ---------------------------------------------------------------------------
# Service-time statistics
# ---------------------------------------------------------------------------

def service_stats(x: list[float], warmup_k: int,
                  outlier_idxs: list[float] | None = None) -> dict:
    """median/mean/p95/cv of the post-warm-up, outlier-excluded series (s)."""
    drop = set(range(warmup_k)) | set(outlier_idxs or [])
    xs = [v for i, v in enumerate(x) if i not in drop]
    if not xs:
        return {"n": 0}
    mean = sum(xs) / len(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    out = {
        "n": len(xs),
        "median": statistics.median(xs),
        "mean": mean,
        "cv": sd / mean if mean else float("nan"),
        "min": min(xs),
        "max": max(xs),
    }
    if len(xs) >= 20:
        s = sorted(xs)
        out["p95"] = s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]
    return out


# ---------------------------------------------------------------------------
# knobs.yml reader (drivers and analyzers consume knobs through this)
# ---------------------------------------------------------------------------

def load_knobs(path=KNOBS_PATH) -> dict:
    """Parsed knobs.yml, or {} if not yet generated."""
    import yaml
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def get_knob(knobs: dict, experiment: str, device: str, name: str, default=None):
    """Look up a knob value; returns default when knobs.yml lacks it."""
    for entry in (knobs.get("experiments", {}).get(experiment, {}).get(device) or []):
        if entry.get("knob") == name:
            return entry.get("value")
    return default


# ---------------------------------------------------------------------------
# Intended-arrivals sidecar
# ---------------------------------------------------------------------------

@dataclass
class ArrivalTrace:
    epochs: list[int]
    intended_ts: list[float]
    actual_ts: list[float]
    block_s: list[float]

    @property
    def n(self) -> int:
        return len(self.epochs)

    @property
    def max_block_s(self) -> float:
        return max(self.block_s) if self.block_s else 0.0

    @property
    def blocked_puts(self) -> int:
        return sum(1 for b in self.block_s if b > 0.005)

    def realized_rate(self) -> float:
        if self.n < 2:
            return float("nan")
        span = self.actual_ts[-1] - self.actual_ts[0]
        return (self.n - 1) / span if span > 0 else float("nan")

    def intended_rate(self) -> float:
        if self.n < 2:
            return float("nan")
        span = self.intended_ts[-1] - self.intended_ts[0]
        return (self.n - 1) / span if span > 0 else float("nan")


def parse_arrivals(sidecar_csv) -> ArrivalTrace:
    epochs, it, at, bl = [], [], [], []
    with open(sidecar_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                epochs.append(int(row["epoch"]))
                it.append(float(row["intended_ts"]))
                at.append(float(row["actual_ts"]))
                bl.append(float(row["block_s"]))
            except (KeyError, ValueError):
                continue
    return ArrivalTrace(epochs, it, at, bl)


def default_results_dir():
    return str(RESULTS_DIR)


def repo_git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True
                              ).stdout.strip()
    except Exception:
        return "unknown"
