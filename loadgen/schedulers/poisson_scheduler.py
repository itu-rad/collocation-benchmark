from random import expovariate
import uuid
from time import sleep
from threading import Timer

from loadgen.schedulers.scheduler import LoadScheduler


class PoissionLoadScheduler(LoadScheduler):
    offsets = []
    max_queries = 100
    rate = 10
    timer = None
    stop = False

    def __init__(self, loadgen_config):
        super().__init__(loadgen_config)

        poisson_config = loadgen_config.get("config", {})
        self.rate = poisson_config.get("rate", 3)

    def prepare(self):
        self.offsets = [expovariate(self.rate) for _ in range(self.max_queries - 1)]

    def generate(self, queue, event):
        self.timer.start()
        queue.put_nowait(uuid.uuid4())
        for offset in self.offsets:
            if self.stop:
                break
            sleep(offset)
            queue.put_nowait(uuid.uuid4())
        queue.put(None)
        self.timer.cancel()
