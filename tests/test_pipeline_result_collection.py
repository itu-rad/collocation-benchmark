"""Pipeline result collection: no polling, correct shutdown, no lost wakeups.

None of this was covered before. It matters because the collector and the drain
loop both used to wake every 100 ms, which cost measurable time inside E2's
dataloader stage, and because the last shutdown bug of this class was found only
by noticing 80 s of dead time on a live run rather than by a test.

CHOREO_DISABLE_TRACING is set before importing anything that pulls in
utils.trace_span -- its MODE is resolved at import.
"""

import os
import sys
import threading
import time
import unittest
from queue import Queue

os.environ.setdefault("CHOREO_DISABLE_TRACING", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.pipeline import Pipeline                       # noqa: E402
from stages import Stage                                     # noqa: E402
from utils.queues import PeekableQueue                       # noqa: E402
from utils.schemas import PipelineModel, Query               # noqa: E402

IDENTITY = "stages.Stage"
SLOW = "tests.test_pipeline_result_collection.SlowStage"
SLOW_S = 0.03


class SlowStage(Stage):
    """Slow enough that _wait_for_drain actually blocks. With instant stages the
    counter has already reached its target before the drain loop ever waits, so
    the shared-Event race window never opens and the test proves nothing."""

    def run(self, query):
        time.sleep(SLOW_S)
        return {i: query for i in self.output_queues}



def _config(n_outputs=1, serialize=False, component=IDENTITY):
    """A pipeline of one input stage fanning out to `n_outputs` sinks."""
    sinks = list(range(1, n_outputs + 1))
    stages = [{"id": 0, "name": "src", "component": component,
               "outputs": sinks, "disable_logs": True}]
    stages += [{"id": i, "name": f"sink{i}", "component": component,
                "outputs": [], "disable_logs": True} for i in sinks]
    return PipelineModel(
        inputs=[0], outputs=sinks, dataset_stage_id=0,
        loadgen={"component": "loadgen.schedulers.OfflineLoadScheduler",
                 "max_queries": 1000, "timeout": 600},
        stages=stages, name="test pipeline",
        serialize_queries=serialize,
    )


def _query(i):
    return Query(query_id=i, query_submitted_timestamp=0.0, epoch=0,
                 split="train", batch=i, data={})


class TestResultCollection(unittest.TestCase):

    def _run_pipeline(self, n_queries, n_outputs=1, serialize=False,
                      decoy_event_consumer=False, component=IDENTITY):
        p = Pipeline(_config(n_outputs, serialize, component))
        p.prepare()
        q, event = Queue(), threading.Event()
        for i in range(n_queries):
            q.put(_query(i))
        q.put(None)

        stop_decoy = threading.Event()
        if decoy_event_consumer:
            # Mimics OfflineLoadScheduler: a second waiter on the SAME Event
            # that also clears it, so every set() can be stolen from the drain
            # loop. The Event-based drain survived this only via its 100 ms cap.
            def decoy():
                while not stop_decoy.is_set():
                    if event.wait(0.05):
                        event.clear()
            threading.Thread(target=decoy, daemon=True).start()

        t = threading.Thread(target=p.run, args=[q, event])
        started = time.monotonic()
        t.start()
        t.join(timeout=30)
        elapsed = time.monotonic() - started
        stop_decoy.set()
        self.assertFalse(t.is_alive(), "pipeline.run did not return -- shutdown hung")
        return p, elapsed

    def test_all_queries_retrieved_and_clean_shutdown(self):
        p, _ = self._run_pipeline(25)
        self.assertEqual(p.queries_processed, 25)
        self.assertTrue(all(p._outputs_drained))
        for stage in p.stages.values():
            self.assertFalse(stage._thread.is_alive(), f"{stage.name} thread alive")

    def test_collector_wakes_on_notify_not_on_timeout(self):
        """Discriminating: a broken notify path falls back to the 1 s safety
        net, and any reintroduced poll longer than 100 ms also fails this."""
        p = Pipeline(_config(1))
        out = PeekableQueue(notify_condition=p._output_cond)
        p._output_queues = [out]
        p._outputs_drained = [False]

        t = threading.Thread(target=p.retrieve_results, args=[threading.Event()],
                             daemon=True)
        t.start()
        time.sleep(0.25)                       # let it park in cond.wait()
        self.assertEqual(p.queries_processed, 0)

        started = time.monotonic()
        out.put(_query(1))
        while p.queries_processed < 1 and time.monotonic() - started < 2:
            time.sleep(0.001)
        latency = time.monotonic() - started
        self.assertEqual(p.queries_processed, 1)
        self.assertLess(latency, 0.1, f"woke in {latency:.3f}s -- not on the notify")

        out.put(None)
        t.join(timeout=2)
        self.assertFalse(t.is_alive(), "collector did not exit on the terminator")

    def test_two_outputs_terminator_does_not_discard_the_other(self):
        p = Pipeline(_config(2))
        a = PeekableQueue(notify_condition=p._output_cond)
        b = PeekableQueue(notify_condition=p._output_cond)
        p._output_queues = [a, b]
        p._outputs_drained = [False, False]

        t = threading.Thread(target=p.retrieve_results, args=[threading.Event()],
                             daemon=True)
        t.start()
        a.put(None)                            # first output terminates early
        b.put(_query(7))                       # second still has work
        deadline = time.monotonic() + 2
        while p.queries_processed < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(p.queries_processed, 1,
                         "a terminator on one output discarded the other's work")
        self.assertTrue(t.is_alive(), "collector exited before all outputs drained")

        b.put(None)
        t.join(timeout=2)
        self.assertFalse(t.is_alive())

    def test_drain_waits_on_condition_and_honours_its_timeout(self):
        p = Pipeline(_config(1))

        def producer():
            time.sleep(0.02)
            with p._drain_cond:
                p.queries_processed += 1
                p._drain_cond.notify_all()
        threading.Thread(target=producer, daemon=True).start()
        started = time.monotonic()
        self.assertTrue(p._wait_for_drain(1, timeout=5.0))
        self.assertLess(time.monotonic() - started, 0.5)

        started = time.monotonic()
        self.assertFalse(p._wait_for_drain(99, timeout=0.05))
        elapsed = time.monotonic() - started
        # The old Event-capped loop could only resolve on 100 ms boundaries.
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 0.5)

    def test_serialize_survives_a_greedy_event_consumer(self):
        """Serialized mode stays correct and prompt while a second thread
        competes for the loadgen Event.

        Scope note, so this test is not read as more than it is: the old
        Event-based drain shared one Event with OfflineLoadScheduler and both
        cleared it, so a set() meant for the drain loop could be stolen. I could
        not get that race to reproduce here -- the old 100 ms cap bounds the
        damage, and this test passes against the old implementation too. Waiting
        on _drain_cond removes the race by construction (no shared clear, and
        the counter is incremented under the lock it is tested under); what this
        test actually guards is that serialized mode still completes every query
        promptly with a competing Event consumer present."""
        n = 12
        p, elapsed = self._run_pipeline(n, serialize=True,
                                        decoy_event_consumer=True, component=SLOW)
        self.assertEqual(p.queries_processed, n)
        # Two slow stages per query is the floor. The Event-based drain adds up
        # to a 100 ms stall per stolen wakeup, so allow the floor plus a small
        # margin -- well under floor + n * 0.1.
        floor = n * SLOW_S * 2
        self.assertLess(elapsed, floor + 0.4,
                        f"serialized run took {elapsed:.2f}s against a {floor:.2f}s "
                        f"floor -- drain is stalling on stolen Event wakeups")


if __name__ == "__main__":
    unittest.main(verbosity=2)
