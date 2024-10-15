from threading import Timer
from typing import Any

# other scheduler options include: gamma distribution, paretto distribution, weibull distribution


class LoadScheduler:
    """Base load scheduler class, responsible for generating load for a single pipeline."""

    def __init__(
        self,
        max_queries: int,
        timeout: int,
        load_scheduler_config: dict[str, Any],
        dataset_splits: dict[str, int],
    ):
        self.max_queries = max_queries
        self.stop = False
        self.timer = Timer(timeout, self.stop_generator)
        self.dataset_splits = dataset_splits
        self.extra_config = load_scheduler_config

    def prepare(self):
        pass

    def stop_generator(self):
        """Set a flag to gracefully end load generation."""
        print("TIMEOUT!")
        self.stop = True

    def generate(self, queue, event):
        pass
