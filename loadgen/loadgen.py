import logging
from logging.handlers import QueueHandler
import uuid

from pipeline import pipeline
from pipeline.pipeline import Pipeline


class LoadGen:
    logger = None
    pipeline = None

    def __init__(self, pipeline_config, logger_queue):

        # initialize process-safe logging
        qh = QueueHandler(logger_queue)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(qh)

        self.pipeline = Pipeline(pipeline_config)

    def run(self):
        self.pipeline.prepare()

        for _ in range(100):
            result = self.pipeline.run(uuid.uuid4())
            print(result)
