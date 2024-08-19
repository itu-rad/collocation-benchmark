from random import expovariate
import uuid
from time import sleep, time

from .scheduler import LoadScheduler


class PoissonLoadScheduler(LoadScheduler):
    """Load generation scheduler based on the poisson distribution"""

    def __init__(self, loadgen_config, dataset_length):
        super().__init__(loadgen_config, dataset_length)

        poisson_config = loadgen_config.get("config", {})
        self.rate = poisson_config.get("rate", 3)
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

        total_length = sum(self.dataset_length.values())
        train_length = self.dataset_length.get("train", 0)

        split = "val"

        # Iterate over all the offsets
        for i, offset in enumerate(self.offsets):
            # look for a timeout
            if self.stop:
                break

            # wait for the offset time before generating next query
            sleep(offset)

            # choose split based on the total iteration count
            iters_this_epoch = i % total_length
            if self.is_training:
                split = "train" if iters_this_epoch < train_length else "val"

            # push the query onto queue
            queue.put_nowait(
                {
                    "id": uuid.uuid4(),
                    "split": split,
                    "query_submitted": time(),
                    "batch": (
                        iters_this_epoch
                        if split == "train"
                        else iters_this_epoch - train_length
                    ),
                }
            )

        # push termination element onto the queue
        queue.put(None)
        self.timer.cancel()
