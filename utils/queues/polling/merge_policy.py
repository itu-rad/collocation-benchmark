from time import sleep
from typing import Any
from queue import Empty

from utils.queues import PeekableQueue
from utils.schemas import Query
from .polling_policy import PollingPolicy


class MergePolicy(PollingPolicy):
    """
    Merges the data from all input queues into a single query.

    Termination semantics: returns None only once every upstream has
    sent its terminator. Until then, a None on one queue is consumed
    and recorded, but the policy keeps waiting for the remaining
    upstreams to produce a matching query before emitting the merged
    result.
    """

    def __init__(self, input_queues: dict[int, PeekableQueue]):
        super().__init__(input_queues)
        self._drained = {key: False for key in self.input_queues}

    def get_input_from_queues(self) -> Query | None:
        """
        Retrieve and merge queries from all input queues into a single Query object.

        Waits until every non-drained upstream has a query available, then
        merges them. Drains None terminators as they arrive without
        exiting until every upstream has terminated.

        Returns:
            Query | None: A Query object with merged data, or None once
            every upstream has terminated.
        """
        while True:
            if all(self._drained.values()):
                return None

            queries: dict[int, Query] = {}
            ready = True
            for key, queue in self.input_queues.items():
                if self._drained[key]:
                    continue
                try:
                    query = queue.peek()
                except Empty:
                    ready = False
                    continue
                if not query:
                    queue.get()
                    self._drained[key] = True
                    continue
                queries[key] = query

            if ready and queries:
                # Pop the queries we peeked.
                for key in queries:
                    self.input_queues[key].get()

                first_query = next(iter(queries.values()))
                merged_query = Query(
                    query_id=first_query.query_id,
                    split=first_query.split,
                    batch=first_query.batch,
                    epoch=first_query.epoch,
                    query_submitted_timestamp=first_query.query_submitted_timestamp,
                )
                merged_data: dict[int, Any] = {
                    key: q.data for key, q in queries.items()
                }
                merged_query.data = merged_data
                return merged_query

            sleep(0.1)
