"""Intended-vs-actual arrival sidecar for open-loop load schedulers.

Open-loop generators promise an arrival process (e.g. Poisson at rate λ), but
the bounded entry queue can silently throttle it (coordinated omission). This
helper records, per query, the *intended* arrival time (from the schedule's
offsets), the *actual* enqueue time, and how long the blocking put() stalled —
and writes them next to the run's trace CSV as ``<label>_arrivals.csv``.

Downstream uses (evaluation/pilots/):
  * R-QDEPTH verification — queue_depth "never blocks" ⇔ max block_s ≈ 0;
  * realized-vs-intended rate check (±5%);
  * E5's "arrival process matches the MLPerf scenario" trace confirmation.

Every open-loop scheduler (Poisson, saturating-Offline, fixed-interval
MultiStream) should adopt this from birth; the closed-loop OfflineLoadScheduler
has no arrival process to verify.
"""

from __future__ import annotations

import os


class ArrivalLog:
    """Accumulates (epoch, intended_ts, actual_ts, block_s) rows; writes on close."""

    def __init__(self):
        self.rows: list[tuple[int, float, float, float]] = []

    def record(self, epoch: int, intended_ts: float, actual_ts: float,
               block_s: float) -> None:
        self.rows.append((epoch, intended_ts, actual_ts, block_s))

    def write(self, results_dir: str = os.path.join("evaluation", "results"),
              label: str | None = None) -> str | None:
        """Write the sidecar CSV; label defaults to SUITE_OUTPUT_LABEL (the
        same env var main.py uses to name the trace CSV). Returns the path."""
        if not self.rows:
            return None
        label = label or os.environ.get("SUITE_OUTPUT_LABEL")
        if not label:
            return None
        try:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{label}_arrivals.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("epoch,intended_ts,actual_ts,block_s\n")
                for epoch, its, ats, bs in self.rows:
                    f.write(f"{epoch},{its:.6f},{ats:.6f},{bs:.6f}\n")
            return path
        except OSError:
            return None  # never let telemetry kill a run
