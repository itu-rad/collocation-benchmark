from threading import Timer

# other scheduler options include: gamma distribution, paretto distribution, weibull distribution


class LoadScheduler:
    max_queries = 100
    timer = None
    stop = False
    dataset_length = dict()
    is_training = False

    def __init__(self, loadgen_config, dataset_length):
        self.max_queries = loadgen_config.get("max_queries", 100)
        timeout = loadgen_config.get("timeout", 15)
        self.timer = Timer(timeout, self.stop_generator)
        self.dataset_length = dataset_length
        self.is_training = loadgen_config.get("is_training", False)

    def prepare(self):
        pass

    def stop_generator(self):
        print("TIMEOUT!")
        self.stop = True

    def generate(self, queue, event):
        pass
