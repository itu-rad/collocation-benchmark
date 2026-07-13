import queue


class PeekableQueue(queue.Queue):
    """
    Wrapper around a queue.Queue that allows peeking at the first element without removing it.

    Optionally notifies a shared ``threading.Condition`` on every ``put``, so a
    consumer that polls several queues at once (the fan-in polling policies) can
    block on that condition instead of busy-waiting with a fixed sleep.
    """

    def __init__(self, maxsize: int = 0, notify_condition=None):
        super().__init__(maxsize=maxsize)
        # Shared across all input queues of one consuming stage; None disables
        # notification (single-queue consumers just block on ``.get()``).
        self._notify_condition = notify_condition

    def peek(self):
        """
        Peek at the first element in the queue without removing it.

        Raises:
            queue.Empty: If the queue is empty.

        Returns:
            Any: The first element in the queue.
        """
        with self.mutex:
            if not self.queue:
                raise queue.Empty()
            return self.queue[0]

    def put(self, item, block=True, timeout=None):
        """Enqueue ``item`` and wake any consumer blocked on ``notify_condition``.

        ``queue.Queue.put_nowait`` delegates to ``put(block=False)``, so
        overriding ``put`` alone covers both call paths. The notify happens
        *after* the item is enqueued and after the queue's own ``mutex`` is
        released, so a consumer holding the condition observes the item.
        """
        super().put(item, block=block, timeout=timeout)
        if self._notify_condition is not None:
            with self._notify_condition:
                self._notify_condition.notify_all()
