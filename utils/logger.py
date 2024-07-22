import logging
from logging.handlers import QueueListener
import os
import re


class Logger:
    queue = None
    queue_listener = None

    def __init__(self, queue, benchmark_name):
        self.queue = queue

        # make benchmark name filename-safe
        benchmark_name = benchmark_name.lower()
        benchmark_name = re.sub("\s", "_", benchmark_name)
        benchmark_name = re.sub("[^\w_]+", "", benchmark_name)
        logging.basicConfig(
            filename=os.path.join(os.getcwd(), "tmp", f"{benchmark_name}.csv"),
            format="%(created)f, %(message)s",
            level=logging.INFO,
        )

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(created)f, %(message)s"))

        # ql gets records from the queue and sends them to the handler
        self.queue_listener = QueueListener(self.queue, handler)
        self.queue_listener.start()

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # add the handler to the logger so records from this process are handled
        logger.addHandler(handler)

    def stop_queue_listener(self):
        self.queue_listener.stop()
