import threading

from utils.queues.peekable_queue import PeekableQueue
from utils.schemas import Query
from abc import ABC, abstractmethod


class PollingPolicy(ABC):
    """
    Base class for all queue polling policies.
    """

    def __init__(
        self,
        input_queues: dict[int, PeekableQueue],
        notify_condition: threading.Condition | None = None,
    ):
        self.input_queues = input_queues
        # Fan-in policies wait on this condition (signalled by PeekableQueue.put)
        # instead of busy-waiting. If none is supplied we fall back to a private
        # condition so ``wait(timeout=...)`` degrades to slow polling rather than
        # crashing; single-queue consumers ignore it entirely.
        self._cond = notify_condition if notify_condition is not None else threading.Condition()

    @abstractmethod
    def get_input_from_queues(self) -> Query | None:
        """
        Poll the input queues and return query based on the policy

        Returns:
            Query | None: Thequery or None if no query is available.
        """
        pass
