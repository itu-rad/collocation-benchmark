"""Orchestrator watchdog: bounded cleanup for the RadT post-loop hang.

Why this exists
---------------
``radt.schedule_external`` runs the workload, then in its post-loop
section uploads its captured stdout/stderr via ``client.log_text`` HTTPS
calls. When the tracking server's idle-connection timeout has fired
mid-pipeline, those calls hang on stale pooled connections; with MLflow's
default 7-retry policy a single ``log_text`` can take ~16 minutes to
fail, and RadT does two such calls per pipeline — so the orchestrator
stays alive for ~30 minutes after the actual work is done.

While it's hung:
  * The parent run sits in ``RUNNING`` state.
  * The orchestrator's RadT listener subprocesses (``macmon pipe``,
    ``smi``, ``top``, …) are orphaned and keep running.
  * If the user CTRL-C's, the listener subprocesses leak forever and
    eventually degrade the host.

This watchdog detects "the workload is genuinely done" by polling MLflow
for the state of our pipeline child runs, gives ``schedule_external`` a
short grace period to drain on its own, and otherwise force-cleans up.

It runs in a daemon thread so a normal completion (``schedule_external``
returns before the workload runs are all FINISHED — unusual but possible)
just kills the watchdog when ``main()`` exits.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Optional

import psutil


_LOGGER = logging.getLogger(__name__)


# Tool names spawned by RadT listeners. We only reap processes whose
# command line contains one of these AND that were started after our
# orchestrator began (so we never kill the user's pre-existing processes).
_LISTENER_TOKENS = (
    "macmon",
    "nvidia-smi",
    "dcgmi",
    "iostat",
    " top ",   # `top -l 0 -s 1` etc. — leading/trailing spaces avoid matching "Python.app"
)


def _call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a worker thread and wait up to ``timeout_s``.

    Returns the result on success, or raises ``TimeoutError`` if the
    worker doesn't finish in time. Used to keep the watchdog itself from
    being subject to the same MLflow HTTPS hangs it's trying to work
    around.
    """
    holder: dict[str, object] = {}

    def _worker():
        try:
            holder["result"] = fn(*args, **kwargs)
        except BaseException as exc:  # pylint: disable=broad-except
            holder["exc"] = exc

    t = threading.Thread(target=_worker, name=f"watchdog-call-{fn.__name__}", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError(f"{fn.__name__} exceeded {timeout_s}s")
    if "exc" in holder:
        raise holder["exc"]  # type: ignore[misc]
    return holder.get("result")


class OrchestratorWatchdog:
    """Watches for workload completion, cleans up if schedule_external hangs.

    Args:
        experiment_id: RadT experiment id (used to scope MLflow queries).
        parent_name: ``BenchmarkModel.name`` — also the parent run's name.
        pipeline_names: list of pipeline names; RadT names their child
            runs as ``"(0 0) <pipeline name>"``.
        process_start_time: ``time.time()`` at orchestrator start. Used
            to filter MLflow runs and process scans so we don't touch
            anything that predates this invocation.
        poll_interval_s: how often to poll MLflow for child run state.
        grace_period_s: after every child is terminal, wait this long
            for ``schedule_external`` to return naturally before forcing.
        absolute_deadline_s: never run longer than this even if children
            never reach terminal state (e.g. they never started).
        mlflow_call_timeout_s: per-MLflow-call timeout INSIDE the
            watchdog — we don't want the watchdog itself to hang on the
            very problem it's trying to work around.
    """

    def __init__(
        self,
        experiment_id: int,
        parent_name: str,
        pipeline_names: list[str],
        process_start_time: float,
        poll_interval_s: float = 30.0,
        grace_period_s: float = 30.0,
        absolute_deadline_s: float = 24 * 3600,
        mlflow_call_timeout_s: float = 30.0,
    ) -> None:
        self.experiment_id = experiment_id
        self.parent_name = parent_name
        self.expected_child_names = set(pipeline_names)
        self.process_start_time = process_start_time
        self.process_start_time_ms = int(process_start_time * 1000)
        self.poll_interval_s = poll_interval_s
        self.grace_period_s = grace_period_s
        self.absolute_deadline_s = absolute_deadline_s
        self.mlflow_call_timeout_s = mlflow_call_timeout_s

        self._schedule_returned = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watchdog thread. Call before ``radt.schedule_external``."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="orchestrator-watchdog", daemon=True
        )
        self._thread.start()

    def notify_schedule_returned(self) -> None:
        """Call this immediately after ``radt.schedule_external`` returns.

        Signals the watchdog that schedule_external completed cleanly, so
        no force-cleanup is needed.
        """
        self._schedule_returned.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        deadline = time.time() + self.absolute_deadline_s

        # Phase 1: poll until all pipeline child runs are terminal
        while time.time() < deadline:
            if self._schedule_returned.is_set():
                return
            if self._all_children_terminal():
                _LOGGER.info("[watchdog] all pipeline child runs are terminal")
                break
            time.sleep(self.poll_interval_s)
        else:
            # Hit the absolute deadline without ever seeing all children
            # terminal. Force cleanup anyway — something is seriously wrong
            # (e.g. pipelines never started) and we don't want to leak.
            print(
                f"[watchdog] absolute deadline ({self.absolute_deadline_s}s) reached "
                f"without all children terminal — forcing cleanup",
                file=sys.stderr, flush=True,
            )
            self._force_cleanup()
            return

        # Phase 2: grace period for schedule_external to return naturally
        print(
            f"[watchdog] pipelines done; allowing schedule_external "
            f"{self.grace_period_s:.0f}s grace to return on its own",
            file=sys.stderr, flush=True,
        )
        if self._schedule_returned.wait(timeout=self.grace_period_s):
            return

        # Phase 3: schedule_external didn't return in time — force cleanup
        print(
            "[watchdog] schedule_external still running after grace period — "
            "force-finalizing parent run, reaping listeners, and exiting",
            file=sys.stderr, flush=True,
        )
        self._force_cleanup()

    # ------------------------------------------------------------------
    # MLflow polling
    # ------------------------------------------------------------------

    def _all_children_terminal(self) -> bool:
        """Return True when every expected pipeline child run is FINISHED/FAILED."""
        try:
            runs = _call_with_timeout(
                self._search_runs, self.mlflow_call_timeout_s,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[watchdog] MLflow poll failed: {exc}",
                  file=sys.stderr, flush=True)
            return False

        # Find each expected pipeline name as either "(0 0) <name>" (RadT
        # child run) or "<name>" (in case the prefix format ever changes).
        found: dict[str, str] = {}
        for r in runs or []:
            name = r.info.run_name or ""
            for pname in self.expected_child_names:
                if name == f"(0 0) {pname}" or name == pname:
                    found[pname] = r.info.status

        if len(found) < len(self.expected_child_names):
            return False
        return all(s in ("FINISHED", "FAILED", "KILLED") for s in found.values())

    def _search_runs(self):
        # Local import: avoid taking the mlflow dependency just to load
        # this module (helps unit testing too).
        import mlflow  # pylint: disable=import-outside-toplevel

        client = mlflow.MlflowClient()
        return client.search_runs(
            experiment_ids=[str(self.experiment_id)],
            filter_string=f"attributes.start_time >= {self.process_start_time_ms}",
            max_results=200,
        )

    # ------------------------------------------------------------------
    # Force cleanup
    # ------------------------------------------------------------------

    def _force_cleanup(self) -> None:
        self._finalize_parent_run()
        self._reap_listener_processes()
        # Hard-exit so we don't get stuck in Python interpreter shutdown
        # on the same pooled HTTPS sockets that hung schedule_external.
        os._exit(0)

    def _finalize_parent_run(self) -> None:
        try:
            runs = _call_with_timeout(
                self._search_runs, self.mlflow_call_timeout_s,
            ) or []
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[watchdog] cleanup search failed: {exc}",
                  file=sys.stderr, flush=True)
            return

        for r in runs:
            if r.info.status != "RUNNING":
                continue
            name = r.info.run_name or ""
            if name != self.parent_name:
                continue
            try:
                _call_with_timeout(
                    self._set_terminated, self.mlflow_call_timeout_s,
                    r.info.run_id,
                )
                print(
                    f"[watchdog] finalized parent run {r.info.run_id[:12]} ({name})",
                    file=sys.stderr, flush=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(
                    f"[watchdog] failed to finalize {r.info.run_id[:12]}: {exc}",
                    file=sys.stderr, flush=True,
                )

    def _set_terminated(self, run_id: str) -> None:
        import mlflow  # pylint: disable=import-outside-toplevel

        mlflow.MlflowClient().set_terminated(run_id, status="FINISHED")

    def _reap_listener_processes(self) -> int:
        """Kill orphaned RadT listener subprocesses started during this run.

        Conservative match: command line contains a listener tool name
        AND ``create_time`` is at or after our orchestrator start. We
        never touch a process that predates this invocation.
        """
        killed = 0
        try:
            for proc in psutil.process_iter(["cmdline", "create_time", "name"]):
                try:
                    info = proc.info
                    cmdline_list = info.get("cmdline") or []
                    cmdline = " ".join(cmdline_list)
                    name_lc = (info.get("name") or "").lower()
                    if not any(t in cmdline or t in name_lc for t in _LISTENER_TOKENS):
                        continue
                    if (info.get("create_time") or 0) < self.process_start_time:
                        continue
                    if proc.pid == os.getpid():
                        continue
                    proc.terminate()
                    killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[watchdog] process scan failed: {exc}",
                  file=sys.stderr, flush=True)
            return killed

        if killed:
            # SIGTERM may not be enough for tools like macmon that ignore it.
            # Give a brief moment, then SIGKILL anything still alive.
            time.sleep(2)
            for proc in psutil.process_iter(["cmdline", "create_time"]):
                try:
                    info = proc.info
                    cmdline = " ".join(info.get("cmdline") or [])
                    if not any(t in cmdline for t in _LISTENER_TOKENS):
                        continue
                    if (info.get("create_time") or 0) < self.process_start_time:
                        continue
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            print(f"[watchdog] reaped {killed} listener process(es)",
                  file=sys.stderr, flush=True)
        return killed
