from utils.queues import PeekableQueue

from utils.schemas import Query
from .polling_policy import PollingPolicy


class FirstSubmittedPolicy(PollingPolicy):
    """
    Polls the first query submitted to the pipeline.
    """

    def __init__(self, input_queues: dict[int, PeekableQueue]):
        super().__init__(input_queues)

    def get_input_from_queues(self) -> Query | None:
        """
        Peeks at all of the queues and returns the first query submitted to the pipeline.

        Returns:
            Query | None: The first query submitted to the pipeline or None as a terminating character.
        """
        queries: dict[int, Query] = {}
        for key, queue in self.input_queues.items():
            # if a queue is empty, just ignore it
            try:
                query = queue.peek()
            except queue.Empty:
                continue
            if not query:
                return None
            queries[key] = query

        # order the queries by submitted timestamp
        first_submitted = sorted(
            queries.items(), key=lambda x: x[1].query_submitted_timestamp
        )[0]

        # poll the first submitted query (already have the query, but need to remove it from its queue)
        first_submitted_query = self.input_queues[first_submitted[0]].get()
        return first_submitted_query
