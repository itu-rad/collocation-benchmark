from random import expovariate
import uuid
from time import sleep, time
from threading import Timer

from loadgen.schedulers.scheduler import LoadScheduler


class PoissionLoadScheduler(LoadScheduler):
    offsets = []
    max_queries = 100
    rate = 10
    timer = None
    stop = False

    def __init__(self, loadgen_config, dataset_length):
        super().__init__(loadgen_config, dataset_length)

        poisson_config = loadgen_config.get("config", {})
        self.rate = poisson_config.get("rate", 3)

    def prepare(self):
        self.offsets = [0]
        for _ in range(self.max_queries - 1):
            self.offsets.append(expovariate(self.rate))

    def generate(self, queue, event):
        self.timer.start()
        event.set()
        total_length = sum(self.dataset_length.values())
        train_length = self.dataset_length.get("train", 0)
        split = "val"
        for i, offset in enumerate(self.offsets):
            if self.stop:
                break
            sleep(offset)
            iters_this_epoch = i % total_length
            if self.is_training:
                split = "train" if iters_this_epoch < train_length else "val"
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

        queue.put(None)
        self.timer.cancel()
