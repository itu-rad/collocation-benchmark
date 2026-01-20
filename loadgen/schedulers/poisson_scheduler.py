from random import expovariate
from time import sleep, time
import threading
import mlflow
import uuid

from utils.schemas import Query
from .scheduler import LoadScheduler


class PoissonLoadScheduler(LoadScheduler):
    """
    PoissonLoadScheduler is a load scheduler that generates load based on a Poisson distribution.
    """

    def __init__(self, max_queries, timeout, load_scheduler_config, dataset_splits):
        """
        Initializes the PoissonScheduler with the given parameters.

        Args:
            max_queries (int): The maximum number of queries to handle.
            timeout (int): The timeout value for the scheduler.
            load_scheduler_config (dict): Configuration settings for the load scheduler.
            dataset_splits (list): List of dataset splits to be used.
        """
        super().__init__(max_queries, timeout, load_scheduler_config, dataset_splits)

        self.rate = self.extra_config.get("rate", 3)
        self.offsets = []

    def prepare(self) -> None:
        """
        Generate time offsets based on the Poisson distribution.

        This method initializes the `offsets` list with a starting value of 0 and
        then appends additional time offsets generated using the exponential
        distribution with the specified rate. The number of offsets generated
        is determined by `self.max_queries`.

        Returns:
            None
        """
        self.offsets = [0]
        for _ in range(self.max_queries - 1):
            self.offsets.append(expovariate(self.rate))

    def generate(self, queue, event):
        """
        Generates queries and pushes them onto the provided queue at intervals defined by offsets.
        Args:
            queue (Queue): The queue to push generated queries onto.
            event (Event): An event to signal the start of query generation.
        The method will generate queries until the maximum number of queries (`self.max_queries`) is reached or a stop signal is received.
        It uses the dataset splits defined in `self.dataset_splits` to generate queries for each split and batch.
        The method also handles timeout and stop signals to ensure proper termination of the query generation process.
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

                    flow_id = uuid.uuid4()
                    with mlflow.start_span(
                        name="generate query",
                        attributes={
                            "out_flow_id": flow_id,
                            "thread_id": threading.get_ident(),
                            "epoch": counter,
                            "batch": batch_idx,
                            "split": split_name,
                        },
                    ):
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
