import logging
from logging.handlers import QueueHandler
from threading import Thread, Event
from queue import Queue
from typing import Union

from pipeline import Pipeline
from .schedulers import LoadScheduler
from .schedulers import SCHEDULER_REGISTRY


def run_loadgen(pipeline_config, logger_queue):
    """Initialization and invokation of LoadGen. Initialization is necessary after process
    creation, because fork is not the default start method on all platforms (spawn does
    not copy all resources).

    Args:
        pipeline_config (dict[str, any]): Pipeline configuration (parsed from YAML file)
        logger_queue (multiprocessing.Queue): Queue to facilitate process-safe logging
    """
    loadgen = LoadGen(pipeline_config, logger_queue)
    loadgen.run()


class LoadGen:
    """Wrapper around execution of a single pipeline. This includes the pipeline itself,
    the load generation, logging and communication between all of these components.
    """

    def __init__(self, pipeline_config, logger_queue):

        # initialize process-safe logging
        qh = QueueHandler(logger_queue)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(qh)

        # initialize the pipeline and all pipeline stages
        pipeline = Pipeline(pipeline_config)
        pipeline.prepare()

        # TODO: Fix this so the loadgen can use multiple datasets
        dataset_length = list(pipeline.get_dataset_length().values())[0]

        # parse the loadgen scheduler config and initialize the appropriate scheduler
        loadgen_config = pipeline_config.get("loadgen", {})
        scheduler_type = loadgen_config.get("type", "offline")
        scheduler_class: Union[LoadScheduler, None] = SCHEDULER_REGISTRY.get(
            scheduler_type, None
        )
        if scheduler_class is None:
            raise ValueError("Scheduler class not found")

        scheduler = scheduler_class(loadgen_config, dataset_length)
        scheduler.prepare()

        # set up queue for passing queries from the loadgen to the pipeline
        queue_depth = loadgen_config.get("queue_depth", 10)
        sample_queue = Queue(maxsize=queue_depth)

        # set up a conditional variable for synchronization between the pipeline
        # execution and load generation (where necessary, i.e. offline scheduler)
        event = Event()

        # set up the pipeline and scheduler threads
        self.pipeline_thread = Thread(target=pipeline.run, args=[sample_queue, event])
        self.scheduler_thread = Thread(
            target=scheduler.generate, args=[sample_queue, event]
        )

    def run(self):
        """Start the pipeline and load generation threads and join them
        after execution is finished."""
        self.pipeline_thread.start()
        self.scheduler_thread.start()
        self.scheduler_thread.join()
        self.pipeline_thread.join()
