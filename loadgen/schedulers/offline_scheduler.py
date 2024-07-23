import uuid
from threading import Timer

from loadgen.schedulers.scheduler import LoadScheduler


class OfflineLoadScheduler(LoadScheduler):
    max_queries = 100
    timer = None
    stop = False

    def __init__(self, loadgen_config):
        super().__init__(loadgen_config)

    def generate(self, queue, event):
        self.timer.start()
        queue.put_nowait(uuid.uuid4())
        for _ in range(self.max_queries - 1):
            if self.stop:
                break
            event.wait()
            event.clear()
            queue.put_nowait(uuid.uuid4())
        queue.put(None)
        self.timer.cancel()
