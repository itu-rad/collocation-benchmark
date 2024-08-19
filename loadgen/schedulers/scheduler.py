from threading import Timer

# other scheduler options include: gamma distribution, paretto distribution, weibull distribution


class LoadScheduler:
    """Base load scheduler class, responsible for generating load for a single pipeline."""

    def __init__(self, loadgen_config, dataset_length):
        self.max_queries = loadgen_config.get("max_queries", 100)
        self.stop = False
        timeout = loadgen_config.get("timeout", 15)
        self.timer = Timer(timeout, self.stop_generator)
        self.dataset_length = dataset_length
        self.is_training = loadgen_config.get("is_training", False)

    def prepare(self):
        pass

    def stop_generator(self):
        """Set a flag to gracefully end load generation."""
        print("TIMEOUT!")
        self.stop = True

    def generate(self, queue, event):
        pass
