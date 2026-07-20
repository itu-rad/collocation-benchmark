import logging
from logging.handlers import QueueListener
import os
import re
import time


# The trace's first column stays wall-clock (``%(created)`` = ``time.time()``)
# because RadT aligns host-listener samples and cross-process/cross-pipeline
# spans on wall-clock. Wall-clock is, however, unsuitable for microsecond-scale
# per-stage latency: it is not monotonic (NTP slew) and its resolution varies.
# We therefore stamp every LogRecord with a monotonic ``perf_counter_ns`` at the
# moment the record is created -- the same instant ``created`` is set, inside the
# record factory -- and emit it as the trailing column. Stages of one pipeline
# run as threads in one process, so this clock is monotonic and directly
# comparable across stages within a pipeline (exactly the quantity per-stage
# latency needs); it is process-local and never used for cross-process work.
PERF_FORMAT = "%(created)f, %(message)s, %(perf)d"

_perf_clock_installed = False


def install_perf_clock():
    """Install a LogRecord factory that adds ``record.perf`` (perf_counter_ns).

    Idempotent and process-global: must be called once per process (the pipeline
    subprocess and the orchestrator) before any record is formatted with
    ``PERF_FORMAT``. Stamping in the factory (vs. a handler filter) means ``perf``
    is set at record-creation time with no handler/queue lag and survives pickling
    through the multiprocessing log queue.
    """
    global _perf_clock_installed
    if _perf_clock_installed:
        return
    _orig_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = _orig_factory(*args, **kwargs)
        record.perf = time.perf_counter_ns()
        return record

    logging.setLogRecordFactory(_factory)
    _perf_clock_installed = True


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

        install_perf_clock()
        formatter = logging.Formatter(PERF_FORMAT)
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
        log_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            filename=os.path.join(log_dir, f"{benchmark_name}.csv")
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
