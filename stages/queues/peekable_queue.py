import queue


class PeekableQueue(queue.Queue):
    """
    Wrapper around a queue.Queue that allows peeking at the first element without removing it.
    """

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
