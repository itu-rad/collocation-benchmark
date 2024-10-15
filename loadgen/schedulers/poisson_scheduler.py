from random import expovariate
from time import sleep, time

from utils.schemas import Query
from .scheduler import LoadScheduler


class PoissonLoadScheduler(LoadScheduler):
    """Load generation scheduler based on the poisson distribution"""

    def __init__(self, max_queries, timeout, load_scheduler_config, dataset_splits):
        super().__init__(max_queries, timeout, load_scheduler_config, dataset_splits)

        self.rate = self.extra_config.get("rate", 3)
        self.offsets = []

    def prepare(self):
        """Generate time offsets based on the poisson distribution."""
        self.offsets = [0]
        for _ in range(self.max_queries - 1):
            self.offsets.append(expovariate(self.rate))

    def generate(self, queue, event):
        """Generate load based on the poisson distribution.
        The generation is stopped when the max_queries is reached or timer elapses.

        Args:
            queue (queue.Queue): Pipeline's input queue.
            event (threading.Event): Conditional variable for synchronizing between the load generation and the pipeline's execution.
        """
        # start the timeout timer
        self.timer.start()
        # release the lock, so the pipeline thread can execute
        event.set()

        counter = 0

        while counter < self.max_queries:
            for split_name, split_batches in self.dataset_splits.items():
                for batch_idx in range(split_batches):
                    # look for a timeout
                    if self.stop:
                        break

                    # sleep until it's time to generate next query
                    sleep(self.offsets[counter])

                    # push the query onto queue
                    queue.put_nowait(
                        Query(
                            split=split_name,
                            batch=batch_idx,
                            query_submitted_timestamp=time(),
                        )
                    )

                    # increament the counter and check that it does not exceed max_queries
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
