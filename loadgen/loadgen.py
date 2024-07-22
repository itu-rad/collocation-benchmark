import logging
from logging.handlers import QueueHandler
import uuid
import torch

from pipeline import pipeline
from pipeline.pipeline import Pipeline


def runLoadGen(pipeline_config, logger_queue):
    """Initialization and invokation of LoadGen. Initialization is necessary after process
    creation, because fork is not the default start method on all platforms (spawn does
    not copy all resources).

    Args:
        pipeline_config (_type_): Pipeline configuration (parsed from YAML file)
        logger_queue (_type_): Queue to facilitate process-safe logging
    """
    loadgen = LoadGen(pipeline_config, logger_queue)
    loadgen.run()


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

        # print("Num threads: ", torch.get_num_thread())
        # torch.set_num_threads(1)
        # print(torch.__config__.parallel_info())

    def run(self):
        self.pipeline.prepare()

        for _ in range(1000):
            result = self.pipeline.run(uuid.uuid4())
            print(result)
            # Test on 1000 samples:
            # mixed_benchmark 36.681s
            # extra preprocessing 32.108s
            # preprocessing in dataloader 33.903s
