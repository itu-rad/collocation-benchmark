import logging
from logging.handlers import QueueHandler
import uuid
from threading import Thread, Event
from queue import Queue
from time import sleep

from loadgen.schedulers.offline_scheduler import OfflineLoadScheduler
from loadgen.schedulers.poisson_scheduler import PoissionLoadScheduler
from loadgen.schedulers.scheduler_registry import SCHEDULER_REGISTRY
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


# responsibility of a scheduler
def genload(sample_queue):
    for _ in range(30):
        sample_queue.put_nowait(uuid.uuid4())
        sleep(0.2)
    sample_queue.put(None)


class LoadGen:
    logger = None
    pipeline_thread = None

    def __init__(self, pipeline_config, logger_queue):

        # initialize process-safe logging
        qh = QueueHandler(logger_queue)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(qh)

        pipeline = Pipeline(pipeline_config)
        pipeline.prepare()
        dataset_length = list(pipeline.get_dataset_length().values())[0]

        loadgen_config = pipeline_config.get("loadgen", {})
        scheduler_type = loadgen_config.get("type", "offline")
        scheduler = SCHEDULER_REGISTRY[scheduler_type](loadgen_config, dataset_length)
        scheduler.prepare()

        queue_depth = loadgen_config.get("queue_depth", 10)
        sample_queue = Queue(maxsize=queue_depth)
        event = Event()

        self.pipeline_thread = Thread(target=pipeline.run, args=[sample_queue, event])
        self.scheduler_thread = Thread(
            target=scheduler.generate, args=[sample_queue, event]
        )

    def run(self):
        self.pipeline_thread.start()
        self.scheduler_thread.start()
        self.scheduler_thread.join()
        self.pipeline_thread.join()
