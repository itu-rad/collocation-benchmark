import threading
import uuid
from time import time
from queue import Queue

import mlflow

from utils.schemas import Query
from .scheduler import LoadScheduler


class SaturatingOfflineScheduler(LoadScheduler):
    """Open-loop *saturating* load generator.

    Enqueues every query up front with no inter-arrival pacing and without
    waiting for the pipeline to finish the previous one. This reproduces MLPerf
    Inference's **Offline** scenario, whose reported metric is throughput
    (completed queries / makespan), not per-query latency.

    Naming note: the repo's ``OfflineLoadScheduler`` is actually a *closed-loop,
    one-in-flight* generator, which corresponds to MLPerf **SingleStream** — not
    Offline. This class is the all-at-once saturating generator that MLPerf
    calls **Offline**. (See the E5 scenario-reduction table.)
    """

    def generate(self, queue: Queue, event: threading.Event) -> None:
        """Enqueue every query as fast as the entry queue accepts it.

        Args:
            queue (queue.Queue): the pipeline's input queue.
            event (threading.Event): pipeline completion signal. Unused here —
                a saturating generator does not pace on completion; we only
                ``set()`` it once so the pipeline thread starts draining.
        """
        # start the timeout timer
        self.timer.start()
        # release the pipeline thread so it begins consuming immediately
        event.set()

        counter = 0
        try:
            while counter < self.max_queries:
                for split_name, split_batches in self.dataset_splits.items():
                    for batch_idx in range(split_batches):
                        # look for a timeout
                        if self.stop:
                            break

                        flow_id = uuid.uuid4()
                        with mlflow.start_span(
                            name="generate query",
                            attributes={
                                "out_flow_id": str(flow_id),
                                "thread_id": threading.get_ident(),
                                "epoch": counter,
                                "batch": batch_idx,
                                "split": split_name,
                            },
                        ):
                            # Blocking put: with an Offline-sized queue_depth
                            # (>= total samples) this never blocks; if the queue
                            # is smaller it back-pressures rather than raising
                            # queue.Full. Throughput (not per-query latency) is
                            # the metric for this scenario, so back-pressure is
                            # acceptable here.
                            queue.put(
                                Query(
                                    split=split_name,
                                    batch=batch_idx,
                                    query_submitted_timestamp=time(),
                                    out_flow_id=flow_id,
                                )
                            )

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
        finally:
            # Always send terminator and cancel the timer, even on exception,
            # so downstream stages don't hang and we don't leak a Timer thread.
            queue.put(None)
            self.timer.cancel()
