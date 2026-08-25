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

Monotonic timestamps on spans
-----------------------------
Spans are stamped by the backend with ``time.time_ns()`` (wall clock), because
mlflow needs wall clock to place spans from different processes on one timeline.
Wall clock is the wrong instrument for per-stage latency: it is not monotonic
(NTP slew), and its granularity is device-dependent — measured 2026-08-25,
``min`` successive delta over 200k reads:

    device                  time.time_ns()    time.perf_counter_ns()
    M2 Pro (macOS)              1000 ns                 41 ns
    GB10 / Grace (Linux)          16 ns                 16 ns

so on macOS the backend clock quantises to 1 µs — a real fraction of a ~23 µs
stage transition.

Rather than change the backend, we attach the monotonic clock **from here, as a
span attribute** (:data:`PERF_ATTR`). radt stays generic, no version pin moves
with our measurement needs, and the exported span carries BOTH clocks: the
backend's wall clock for cross-process alignment, ours for within-process
precision. Measured cost is ~270 ns/span on top of the backend's ~1.0 µs.

``perf_counter_ns`` is PROCESS-LOCAL. The attribute is only comparable between
spans of the same process — which is every quantity a single pipeline measures
(stages are threads). For cross-process work (collocated pipelines) the
analyzer must fall back to the span's own wall-clock start/end.

The attribute is read when ``trace_span()`` is called, a few hundred ns before
the backend reads its wall clock inside ``__enter__``. That offset is constant
and cancels in any interval computed from two ``PERF_ATTR`` values, which is how
the analyzers use it.
"""
import contextlib
import os
import time

_TRUTHY = ("1", "true", "yes")

#: Span-attribute key carrying ``time.perf_counter_ns()`` at span start.
PERF_ATTR = "perf_start_ns"


def _resolve_mode():
    if os.environ.get("CHOREO_PROC_TRACE", "").lower() in _TRUTHY:
        return "proc"
    if os.environ.get("CHOREO_DISABLE_TRACING", "").lower() in _TRUTHY:
        return "off"
    return "mlflow"


MODE = _resolve_mode()


def _with_perf(attributes):
    """Copy `attributes` and stamp the monotonic clock. Copies rather than
    mutating: call sites build the dict inline, but a caller passing a shared
    dict must not have it grow a timestamp key."""
    attrs = dict(attributes) if attributes else {}
    attrs[PERF_ATTR] = time.perf_counter_ns()
    return attrs


if MODE == "proc":
    import radt

    def trace_span(name, attributes=None):
        return radt.trace.span(name, _with_perf(attributes))

elif MODE == "off":

    def trace_span(name, attributes=None):
        return contextlib.nullcontext()

else:
    import mlflow

    def trace_span(name, attributes=None):
        return mlflow.start_span(name=name, attributes=_with_perf(attributes))
