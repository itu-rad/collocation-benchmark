import uuid
from threading import Timer
from time import time

from loadgen.schedulers.scheduler import LoadScheduler


class OfflineLoadScheduler(LoadScheduler):
    max_queries = 100
    timer = None
    stop = False

    def __init__(self, loadgen_config, dataset_length):
        super().__init__(loadgen_config, dataset_length)

    def generate(self, queue, event):
        counter = 0
        self.timer.start()
        while counter < self.max_queries - 1:
            if self.is_training:
                for _ in range(self.dataset_length["train"]):
                    print("train")
                    if self.stop:
                        break
                    if counter > 0:
                        event.wait()
                        event.clear()
                    queue.put_nowait(
                        {
                            "id": uuid.uuid4(),
                            "split": "train",
                            "query_submitted": time(),
                        }
                    )
                    counter += 1
                    if counter > self.max_queries:
                        self.stop = True
            for _ in range(self.dataset_length["val"]):
                print("eval")
                if self.stop:
                    break
                if counter > 0:
                    event.wait()
                    event.clear()
                queue.put_nowait(
                    {"id": uuid.uuid4(), "split": "val", "query_submitted": time()}
                )
                counter += 1
                if counter > self.max_queries:
                    self.stop = True
            if self.stop:
                break

        queue.put(None)
        self.timer.cancel()
