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
import itertools
import logging
import os
import time

_TRUTHY = ("1", "true", "yes")

#: Span-attribute key carrying ``time.perf_counter_ns()`` at span start.
PERF_ATTR = "perf_start_ns"

# The project has no settled name -- "choreo" here is a working title, and the
# on-disk spelling is an ARBITRARY NAMESPACE, not a product name. It is defined
# once, here, so a rename is a one-line change rather than a grep.
#
# It cannot be changed casually: this string is already written into every
# collected trace (the CSV marker row) and onto every mlflow run on res17 (the
# tag), for E1-E4. utils.span_reader accepts both this value and the historical
# spelling when reading, so old data stays readable across a rename.
NAMESPACE = "choreo"

#: Tag / CSV marker under which the emitted-span count is reported.
COUNT_KEY = f"{NAMESPACE}.spans_emitted"

# The backend drops span events on queue overflow (put_nowait) and only WARNS
# at shutdown. We do not change that -- radt stays as it is -- so completeness
# has to be checkable from our side instead: count what we handed over, publish
# it, and let the reader compare it against the manifest's event_count. A
# mismatch means events were dropped between us and the artifact.
#
# itertools.count().__next__ is a single C call, so it is atomic under the GIL
# (a plain `n += 1` is not -- load/add/store can interleave between the stage
# threads). Measured at ~50 ns, against ~1.3 us for the span emit it counts.
_span_seq = itertools.count()
_next_seq = _span_seq.__next__


_final_count = None


def emitted_count():
    """How many spans this process handed to the backend.

    `itertools.count` has no peek, so reading it means consuming an index: the
    value returned is exactly the number of indices issued before it. That
    makes the read single-use, so it is cached -- calling this twice must not
    report two different totals, and the second call would otherwise be one
    too high.

    Call it at shutdown, after the stage threads have joined. Spans emitted
    after the first call are not counted (they would shift the answer under the
    reader's feet), which is why this is a shutdown-time report and not a live
    gauge.
    """
    global _final_count
    if _final_count is None:
        _final_count = _next_seq()
    return _final_count


_reported = False


def report_span_count():
    """Publish the emitted-span count so a reader can verify completeness.

    Written two ways because they fail differently: an mlflow tag travels with
    the run and its artifacts (but needs a live run), and a `benchmark` CSV row
    survives even when the tracking server does not. Neither is gated by
    `disable_logs` -- it is one row per run, and it is the row that says whether
    the rest of the data is complete.

    MUST be called while the mlflow run is still active and the benchmark
    logger's file handler is still open -- i.e. right after the pipeline
    finishes, not at interpreter teardown, where both are already gone and only
    the stdout line survives. The shutdown sites call it again as a backstop;
    reporting happens once and later calls are silent.

    This is a LOWER BOUND, not the final total: once the pipeline finishes, the
    stage threads park in a blocking get_input wait -- inside a span -- until
    os._exit reaps them, so a few more spans start after this runs. Those are
    exactly the spans that never close. utils.span_reader treats a surplus of
    starts as expected and only flags a SHORTFALL, which is what dropped events
    would look like.
    """
    global _reported
    if MODE != "proc" or _reported:
        return _final_count
    _reported = True
    n = emitted_count()
    logging.getLogger("benchmark").info("%s, spans, emitted, %d", NAMESPACE, n)
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.set_tag(COUNT_KEY, str(n))
    except Exception:
        pass          # tagging is the redundant path; the CSV row is the record
    print(f"[{NAMESPACE}] spans emitted: {n}", flush=True)
    return n


def _resolve_mode():
    if os.environ.get("CHOREO_PROC_TRACE", "").lower() in _TRUTHY:
        return "proc"
    if os.environ.get("CHOREO_DISABLE_TRACING", "").lower() in _TRUTHY:
        return "off"
    return "mlflow"


MODE = _resolve_mode()

#: False in the tracing-off arm, so callers can skip building attribute dicts
#: for spans that would be thrown away.
SPANS_ENABLED = MODE != "off"


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
        _next_seq()
        return radt.trace.span(name, _with_perf(attributes))

elif MODE == "off":

    def trace_span(name, attributes=None):
        return contextlib.nullcontext()

else:
    import mlflow

    def trace_span(name, attributes=None):
        return mlflow.start_span(name=name, attributes=_with_perf(attributes))
