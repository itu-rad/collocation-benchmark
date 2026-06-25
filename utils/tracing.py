"""Helpers for MLflow tracing flush handling.

Why this module exists: ``main.py:radt_entrypoint`` and ``main.py:main``
both call ``os._exit(0)`` on completion to skip Python interpreter
shutdown (which hangs for minutes on the MLflow HTTPS socket in
CLOSE_WAIT). ``os._exit`` also skips atexit handlers and MLflow's
background async-logging threads — so any spans / metrics still in flight
when we hit ``os._exit`` are lost.

Two functions:

* ``configure_sync_export`` — call ONCE per process before the first
  ``mlflow.start_span``. Switches trace export from the default async
  queue to synchronous, so spans hit the backend at span-close time
  rather than via a background worker.

* ``flush_traces`` — call right before ``os._exit`` (and from the
  SIGTERM handler). Drains both the run-level async logger (params,
  metrics, artifacts) and the trace async logger (spans).
"""

from __future__ import annotations

import os
import sys


# Env var read by MLflow's tracing exporter at initialization. Set BEFORE
# any mlflow.start_span call — once the exporter is constructed the flag
# is frozen for the process lifetime.
_ASYNC_TRACE_ENV_VAR = "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"


def configure_sync_export() -> None:
    """Force MLflow tracing to export spans synchronously.

    Idempotent. Call once per process, as early as possible (before any
    ``mlflow.start_span`` call).
    """
    # setdefault: an explicit user override in the shell wins.
    os.environ.setdefault(_ASYNC_TRACE_ENV_VAR, "true")
    print("Blocked from configuring trace export. Set to ASYNC TRUE")


def flush_traces() -> None:
    """Drain pending MLflow trace spans and run-level async logs.

    Safe to call multiple times. Never raises — flush errors are printed
    to stderr so the caller (typically ``os._exit`` right after) can keep
    going.
    """
    try:
        import mlflow  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        print(f"[tracing] mlflow unavailable, cannot flush: {exc}",
              file=sys.stderr, flush=True)
        return

    # 1) Drain MLflow run-level async logger (params, metrics, artifacts).
    try:
        mlflow.flush_async_logging()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[tracing] flush_async_logging failed: {exc}",
              file=sys.stderr, flush=True)

    # 2) Drain MLflow trace async logger (spans). Only meaningful when async
    # export is on — in sync mode (our default, see configure_sync_export)
    # spans are already exported at span-close and there's no queue.
    # Calling it anyway in sync mode causes MLflow to log a noisy
    # "'MlflowV3SpanExporter' object has no attribute '_async_queue'" error.
    if os.environ.get(_ASYNC_TRACE_ENV_VAR, "true").lower() != "false":
        try:
            mlflow.flush_trace_async_logging(terminate=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[tracing] flush_trace_async_logging failed: {exc}",
                  file=sys.stderr, flush=True)
