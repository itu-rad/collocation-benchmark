import logging
from logging.handlers import QueueListener
import os
import re


class BenchmarkFilter(logging.Filter):
    """Only allow logs from the 'benchmark' logger"""

    def filter(self, record):
        return record.name == "benchmark"


class Logger:
    """
    Multiprocessing-safe logger, listening on events in a queue.

    Attributes:
        queue (multiprocessing.Queue): The queue to listen on for logging events.
        queue_listener (logging.handlers.QueueListener): The listener that processes log records from the queue.

    Methods:
        __init__(queue, benchmark_name):
            Initializes the Logger with a queue and a benchmark name.

            Args:
                queue (multiprocessing.Queue): The queue to listen on for logging events.
                benchmark_name (str): The name of the benchmark, used to create a filename-safe log file.

        stop_queue_listener():
            Stops the queue listener.
    """

    queue = None
    queue_listener = None

    def __init__(self, queue, benchmark_name):
        self.queue = queue

        formatter = logging.Formatter("%(created)f, %(message)s")
        # formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")

        # Create filter to only allow logs from benchmark logger
        benchmark_filter = BenchmarkFilter()

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # stream_handler.addFilter(benchmark_filter)

        # make benchmark name filename-safe
        benchmark_name = benchmark_name.lower()
        benchmark_name = re.sub(r"\s", "_", benchmark_name)
        benchmark_name = re.sub(r"[^\w_]+", "", benchmark_name)
        file_handler = logging.FileHandler(
            filename=os.path.join(os.getcwd(), "tmp", f"{benchmark_name}.csv")
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(benchmark_filter)

        # queue_listener gets records from the queue and sends them to the handler
        self.queue_listener = QueueListener(self.queue, stream_handler, file_handler)
        self.queue_listener.start()

        logger = logging.getLogger("benchmark")
        logger.setLevel(logging.DEBUG)
        # add the handler to the logger so records from this process are handled
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    def stop_queue_listener(self):
        """Stop the queue listener"""
        self.queue_listener.stop()
