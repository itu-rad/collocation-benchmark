import uuid
from time import time

from .scheduler import LoadScheduler


class OfflineLoadScheduler(LoadScheduler):
    """Load generation scheduler, which waits for processing of the previous query"""

    def __init__(self, loadgen_config, dataset_length):
        super().__init__(loadgen_config, dataset_length)

    def generate(self, queue, event):
        """Generate load, which is synchronized with the pipeline execution.
        The generation is stopped when the max_queries is reached or timer elapses.

        Args:
            queue (queue.Queue): Pipeline's input queue.
            event (threading.Event): Conditional variable for synchronizing between the load generation and the pipeline's execution.
        """
        counter = 0
        # start the timeout timer
        self.timer.start()

        # terminate when max_queries is reached
        while counter < self.max_queries - 1:
            # optionally use the training split
            if self.is_training:
                # loop through entire train set
                for batch_idx in range(self.dataset_length["train"]):
                    # look for a timeout
                    if self.stop:
                        break

                    # wait for the pipeline execution to finish
                    if counter > 0:
                        event.wait()
                        event.clear()

                    # push the query onto queue
                    queue.put_nowait(
                        {
                            "id": uuid.uuid4(),
                            "split": "train",
                            "batch": batch_idx,
                            "query_submitted": time(),
                        }
                    )

                    # trigger termination (gracefully)
                    counter += 1
                    if counter > self.max_queries:
                        self.stop = True

            # loop through entire validation set
            for batch_idx in range(self.dataset_length["val"]):
                # look for a timeot
                if self.stop:
                    break

                # wait for the pipeline execution to finish
                if counter > 0:
                    event.wait()
                    event.clear()

                # push the query onto queue
                queue.put_nowait(
                    {
                        "id": uuid.uuid4(),
                        "split": "val",
                        "batch": batch_idx,
                        "query_submitted": time(),
                    }
                )

                # trigger termination (gracefully)
                counter += 1
                if counter > self.max_queries:
                    self.stop = True
            if self.stop:
                break

        # push termination element onto the queue
        queue.put(None)
        self.timer.cancel()
