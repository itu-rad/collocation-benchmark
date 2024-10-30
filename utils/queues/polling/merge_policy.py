from typing import Any

from utils.schemas import Query
from .polling_policy import PollingPolicy


class MergePolicy(PollingPolicy):
    """
    Merges the data from all input queues into a single query.
    """

    def get_input_from_queues(self) -> Query | None:
        """
        Retrieve and merge queries from all input queues into a single Query object.

        This method retrieves the first query from each input queue, merges their data
        into a dictionary, and returns a new Query object containing this merged data.
        The metadata for the returned Query is taken from the first query retrieved.

        Returns:
            Query | None: A Query object with merged data from all queues, or None if terminating character is received.
        """

        # retrieve the first query from each queue
        queries: dict[int, Query] = {}
        for key, queue in self.input_queues.items():
            query = queue.get()
            if not query:
                return None
            queries[key] = query

        # populate data of the query to return using one of the queries retrieved from the queues
        first_query = next(iter(queries.values()))
        merged_query = Query(
            query_id=first_query.query_id,
            split=first_query.split,
            batch=first_query.batch,
            epoch=first_query.epoch,
            query_submitted_timestamp=first_query.query_submitted_timestamp,
        )

        # merge the data of the retrieved queries into a dictionary
        merged_data: dict[int, Any] = {}
        for key, query in queries.items():
            merged_data[key] = query.data

        # pass the dictionary to the output query
        merged_query.data = merged_data
        return merged_query
