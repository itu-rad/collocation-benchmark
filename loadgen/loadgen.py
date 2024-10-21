import logging
from logging.handlers import QueueHandler
from threading import Thread, Event
from queue import Queue
import multiprocessing

from pipeline import Pipeline
from utils.component import get_component
from utils.schemas import PipelineModel


def run_loadgen(
    pipeline_config: PipelineModel, logger_queue: multiprocessing.Queue
) -> None:
    """
    Initialization and invokation of LoadGen. Initialization is necessary after process
    creation, because fork is not the default start method on all platforms (spawn does
    not copy all resources).

    Args:
        pipeline_config (PipelineModel): Pipeline configuration (parsed from YAML file)
        logger_queue (multiprocessing.Queue): Queue to facilitate process-safe logging
    """
    loadgen = LoadGen(pipeline_config, logger_queue)
    # TODO: Write this to a .md file
    print(loadgen)
    loadgen.run()


class LoadGen:
    """
    Wrapper around execution of a single pipeline. This includes the pipeline itself,
    the load generation, logging and communication between all of these components.
    """

    def __init__(
        self, pipeline_config: PipelineModel, logger_queue: multiprocessing.Queue
    ):
        self._pipeline_config = pipeline_config

        # initialize process-safe logging
        qh = QueueHandler(logger_queue)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(qh)

        # initialize the pipeline and all pipeline stages
        self._pipeline = Pipeline(pipeline_config)
        self._pipeline.prepare()
        dataset_splits = self._pipeline.get_dataset_splits()

        # parse the loadgen scheduler config and initialize the appropriate scheduler
        loadgen_config = pipeline_config.loadgen
        load_scheduler_config = loadgen_config.config
        self._load_scheduler = get_component(loadgen_config.component)(
            loadgen_config.max_queries,
            loadgen_config.timeout,
            load_scheduler_config,
            dataset_splits,
        )
        self._load_scheduler.prepare()

        # set up queue for passing queries from the loadgen to the pipeline
        sample_queue = Queue(maxsize=loadgen_config.queue_depth)

        # set up a conditional variable for synchronization between the pipeline
        # execution and load generation (where necessary, i.e. offline scheduler)
        event = Event()

        # set up the pipeline and scheduler threads
        self.pipeline_thread = Thread(
            target=self._pipeline.run, args=[sample_queue, event]
        )
        self.scheduler_thread = Thread(
            target=self._load_scheduler.generate, args=[sample_queue, event]
        )

    def __str__(self) -> str:
        s = f"---\ntitle: 'Pipeline: {self._pipeline_config.name}'\n---\nflowchart LR\n"
        s += str(self._load_scheduler)
        s += str(self._pipeline)
        return s

    def run(self) -> None:
        """
        Start the pipeline and load generation threads and join them
        after execution is finished.

        The execution is stopped when the max_queries is reached or timer elapses.
        """
        self.pipeline_thread.start()
        self.scheduler_thread.start()
        self.scheduler_thread.join()
        self.pipeline_thread.join()
