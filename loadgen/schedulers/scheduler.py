from threading import Timer


class LoadScheduler:
    max_queries = 100
    timer = None
    stop = False

    def __init__(self, loadgen_config):
        self.max_queries = loadgen_config.get("max_queries", 100)
        timeout = loadgen_config.get("timeout", 15)
        self.timer = Timer(timeout, self.stop_generator)

    def prepare(self):
        pass

    def stop_generator(self):
        print("TIMEOUT!")
        self.stop = True

    def generate(self, queue, event):
        pass
