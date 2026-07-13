import threading
from utils.queues import PeekableQueue
from queue import Empty

from utils.schemas import Query
from .polling_policy import PollingPolicy


class FirstSubmittedPolicy(PollingPolicy):
    """
    Polls the first query submitted to the pipeline.

    Termination semantics: each upstream may send its own None terminator.
    A queue is considered "drained" only after we've consumed its None.
    The policy returns None only once ALL upstream queues have drained,
    so a None on one input never causes us to drop in-flight queries
    queued on a sibling input (important for stages with back-edges,
    e.g. a retriever fed by both a dataloader and a rewriter loop).

    When no input is ready the policy blocks on the shared condition
    (signalled by ``PeekableQueue.put``) rather than busy-waiting, so a
    fan-in stage adds no per-hop latency floor.
    """

    def __init__(
        self,
        input_queues: dict[int, PeekableQueue],
        notify_condition: threading.Condition | None = None,
    ):
        super().__init__(input_queues, notify_condition)
        # Track which upstreams have sent their None.
        self._drained = {key: False for key in self.input_queues}

    def get_input_from_queues(self) -> Query | None:
        """
        Peeks at all of the queues and returns the first query submitted to the pipeline.
        Drains None markers from terminating upstreams without exiting until
        every upstream has terminated.

        Returns:
            Query | None: The first query submitted to the pipeline or None
            once every upstream has sent its terminator.
        """

        with self._cond:
            while True:
                # If every upstream has terminated, we're done.
                if all(self._drained.values()):
                    return None

                queries: dict[int, Query] = {}
                for key, queue in self.input_queues.items():
                    if self._drained[key]:
                        continue
                    try:
                        query = queue.peek()
                    except Empty:
                        continue
                    if not query:
                        # Consume the None terminator and mark this upstream
                        # as drained, but keep polling the others.
                        queue.get()
                        self._drained[key] = True
                        continue
                    queries[key] = query

                if queries:
                    # order the queries by submitted timestamp
                    first_submitted = sorted(
                        queries.items(), key=lambda x: x[1].query_submitted_timestamp
                    )[0]

                    # poll the first submitted query (already peeked, but need
                    # to remove it from its queue). Item is present and this is
                    # the sole consumer, so get() returns without blocking.
                    return self.input_queues[first_submitted[0]].get()

                # A terminator consumed above may have drained the last upstream.
                if all(self._drained.values()):
                    return None

                # Nothing ready and not all upstreams drained: block until a
                # producer notifies via put(), or the safety-net timeout elapses.
                # Correctness: readiness is re-checked at the top of the loop
                # while holding self._cond, so a notify issued after put() (which
                # also acquires self._cond) cannot be lost.
                self._cond.wait(timeout=1.0)
