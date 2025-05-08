import uuid
from time import time
from queue import Queue
import threading

from utils.schemas import Query
from .scheduler import LoadScheduler


class OfflineLoadScheduler(LoadScheduler):
    """Load generation scheduler, which waits for processing of the previous query"""

    def generate(self, queue: Queue, event: threading.Event) -> None:
        """Generate load, which is synchronized with the pipeline execution.
        The generation is stopped when the max_queries is reached or timer elapses.

        Args:
            queue (queue.Queue): Pipeline's input queue.
            event (threading.Event): Conditional variable for synchronizing between the load generation and the pipeline's execution.
        """
        counter = 0
        # start the timeout timer
        self.timer.start()

        print("Starting load generation")
        print("Splits: ", self.dataset_splits)

        # terminate when max_queries is reached
        while counter < self.max_queries:
            for split_name, split_batches in self.dataset_splits.items():
                for batch_idx in range(split_batches):
                    # look for a timeout
                    if self.stop:
                        break

                    # wait for the pipeline execution to finish
                    # wait only when we have already pushed something (counter > 0)
                    if counter > 0:
                        event.wait()
                        event.clear()

                    # push the query onto queue
                    queue.put_nowait(
                        Query(
                            split=split_name,
                            batch=batch_idx,
                            query_submitted_timestamp=time(),
                        )
                    )

                    # increment the counter and check if exceeds the max_queries
                    counter += 1
                    if counter >= self.max_queries:
                        self.stop = True
                        break
                # propagate the stop signal
                if self.stop:
                    break
            # propagate the stop signal
            if self.stop:
                break
        # push termination element onto the queue
        queue.put(None)
        self.timer.cancel()
