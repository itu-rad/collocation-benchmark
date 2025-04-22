from utils.queues.peekable_queue import PeekableQueue
from utils.schemas import Query
from abc import ABC, abstractmethod


class PollingPolicy(ABC):
    """
    Base class for all queue polling policies.
    """

    def __init__(self, input_queues: dict[int, PeekableQueue]):
        self.input_queues = input_queues

    @abstractmethod
    def get_input_from_queues(self) -> Query | None:
        """
        Poll the input queues and return query based on the policy

        Returns:
            Query | None: Thequery or None if no query is available.
        """
        pass
