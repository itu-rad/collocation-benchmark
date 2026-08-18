"""Dispatch span creation across three backends (prototype A/B switch).

Every ``with mlflow.start_span(name, attributes)`` site in the pipeline goes
through :func:`trace_span` so the tracing backend is a single env switch:

  * ``proc``   (``CHOREO_PROC_TRACE`` truthy)   -> ``radt.trace.span`` — radt owns
    mlflow in a separate process; the workload only emits a lightweight event.
  * ``off``    (``CHOREO_DISABLE_TRACING``)     -> ``nullcontext`` — no spans (the
    tracing-off "core" arm).
  * ``mlflow`` (default)                        -> ``mlflow.start_span`` — the
    current in-process path (kept for A/B comparison).

The CSV timing markers emitted by ``log_phase`` are independent of spans and are
produced identically in all three modes.
"""
import contextlib
import os

_TRUTHY = ("1", "true", "yes")


def _resolve_mode():
    if os.environ.get("CHOREO_PROC_TRACE", "").lower() in _TRUTHY:
        return "proc"
    if os.environ.get("CHOREO_DISABLE_TRACING", "").lower() in _TRUTHY:
        return "off"
    return "mlflow"


MODE = _resolve_mode()

if MODE == "proc":
    import radt

    def trace_span(name, attributes=None):
        return radt.trace.span(name, attributes)

elif MODE == "off":

    def trace_span(name, attributes=None):
        return contextlib.nullcontext()

else:
    import mlflow

    def trace_span(name, attributes=None):
        return mlflow.start_span(name=name, attributes=attributes)
