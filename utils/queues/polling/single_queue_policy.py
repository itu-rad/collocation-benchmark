from queue import Queue

from utils.schemas import Query
from .polling_policy import PollingPolicy


class SingleQueuePolicy(PollingPolicy):
    """
    Polls the first input queue. Only applicable when a single input queue is used.
    """

    def get_input_from_queues(self) -> Query | None:
        """
        Poll the first input queue. Only applicable when a single input queue is used.

        Returns:
            Query | None: The next query from the first input queue or None as a terminating character.
        """

        return next(iter(self.input_queues.values())).get()
