import logging
import threading
import uuid
from queue import Queue, Empty
from threading import Event, Thread

import mlflow

from stages import Stage
from utils.component import get_stage_component
from utils.schemas import PipelineModel, StageModel, Query


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    def __init__(self, pipeline_config: PipelineModel):
        """Initialize the pipeline by parsing the pipeline configuration.

        Args:
            pipeline_config (PipelineModel): The configuration of the pipeline.
        """
        self._logger = logging.getLogger("benchmark")
        self._pipeline_config = pipeline_config
        self.name = self._pipeline_config.name
        stage_config = pipeline_config.stages
        stage_config_dict = {stage.id: stage for stage in stage_config}

        self._input_queues: list[Queue] = []
        self._output_queues: list[Queue] = []

        self.stages = self._create_stages(stage_config_dict)

        self._populate_stages_with_stage_dict()
        self._populate_queues()

    def __str__(self) -> str:
        s = "subgraph Stages\n"
        for stage in self.stages.values():
            s += str(stage)
        s += "end\n"
        for input_stage_id in self._pipeline_config.inputs:
            s += f"load_sched -->|queue depth:{self._pipeline_config.loadgen.queue_depth}|{input_stage_id}\n"
        for output_stage_id in self._pipeline_config.outputs:
            s += f"{output_stage_id} --> load_sched\n"
        return s

    def _create_stages(
        self, stage_config_dict: dict[int, StageModel]
    ) -> dict[int, Stage]:
        """Create a dictionary of stage objects from the given configuration.

        Args:
            stage_config_dict (dict[int, StageModel]): A dictionary where the keys are stage IDs and the values are the configurations of the stages.

        Returns:
            dict: A dictionary where the keys are stage IDs and the values are the corresponding stage objects.
        """
        stages = {}
        for stage_idx, stage_config in stage_config_dict.items():
            stage_obj = get_stage_component(stage_config, self._pipeline_config)
            stages[stage_idx] = stage_obj
        return stages

    def _populate_stages_with_stage_dict(self) -> None:
        """
        Populate all of the stages in the pipeline with the stage dictionary.
        """
        for stage in self.stages.values():
            stage.set_stage_dict(self.stages)

    def _populate_queues(self) -> None:
        """
        Populate all of the stages in the pipeline with their output queues.
        """

        for stage in self.stages.values():
            stage.set_output_queues()

    def get_dataset_splits(self) -> dict[str, int]:
        """
        Get the number of batches in each dataset split of the pipeline.

        Returns:
            dict[str, int]: A dictionary where the keys are the dataset split names and the values are the number of batches in each split.
        """
        stage_id = self._pipeline_config.dataset_stage_id
        return self.stages[stage_id].get_dataset_splits()

    def prepare(self) -> None:
        """
        Prepare all the stages in the pipeline.

        This includes creating input queues, output queues, as well as calling the prepare method of each stage.
        """
        self._logger.info("%s, pipeline, prepare, start", self.name)

        # get all the input queues of the pipeline
        for input_stage_idx in self._pipeline_config.inputs:
            self._input_queues.append(self.stages[input_stage_idx].get_input_queue(-1))

        # create output queue for each output stage of the pipeline and pass it to the output stage
        for output_stage_idx in self._pipeline_config.outputs:
            queue = Queue()
            self._output_queues.append(queue)
            self.stages[output_stage_idx].set_output_queue(queue)

        # start the stage threads
        for stage in self.stages.values():
            stage.prepare()

        self._logger.info("%s, pipeline, prepare, end", self.name)

    def join_threads(self, timeout: float | None = None) -> None:
        """
        Wait for all the stage threads to finish, with optional per-stage timeout.

        Bounded join prevents a stuck stage from holding the whole pipeline
        in shutdown forever. The caller (typically `main.py`) is expected
        to force-exit the process after this returns so any straggler
        threads are reclaimed.
        """
        for stage in self.stages.values():
            stage._thread.join(timeout=timeout)

    def retrieve_results(self, event: Event) -> None:
        """
        This method continuously checks all output queues for new queries. It performs
        non-blocking retrieval from the queues and processes the queries if available.
        If a terminating character (None) is received, the method returns and stops
        further processing. Additionally, it logs the end of the pipeline execution
        and sets the provided event to signal completion.

        Args:
            event (Event): An event object used to signal the completion of the pipeline.
        """

        _EMPTY = object()

        while True:
            for output_queue in self._output_queues:
                # Span every poll so the polling overhead shows up in the
                # trace. Catch Empty INSIDE the span — an empty queue is
                # the expected polling case, not an error, and letting it
                # propagate makes MLflow tag the span as failed.
                new_query = _EMPTY
                with mlflow.start_span(
                    name="pipeline retrieve_results",
                    attributes={"thread_id": threading.get_ident()},
                ):
                    try:
                        new_query = output_queue.get(timeout=0.1)
                    except Empty:
                        pass

                if new_query is _EMPTY:
                    continue

                # if terminating character is received, return
                if new_query is None:
                    return

                # log the end of pipeline execution
                self._logger.info(
                    "%s, pipeline - %s, run, end, %d, %.6f, %d, %d",
                    self.name,
                    new_query.split,
                    new_query.query_id,
                    new_query.query_submitted_timestamp,
                    new_query.epoch,
                    new_query.batch + 1,
                )

                with mlflow.start_span(
                    name="pipeline query processed",
                    attributes={
                        "in_flow_id": str(new_query.out_flow_id) if new_query.out_flow_id else None,
                        "thread_id": threading.get_ident(),
                        "pipeline": self.name,
                        "epoch": new_query.epoch,
                        "split": new_query.split,
                        "batch": new_query.batch,
                        "query_id": new_query.query_id,
                    },
                ):
                    self.queries_processed += 1
                event.set()

    def run(self, query_queue: Queue, event: Event) -> None:
        """
        Executes the pipeline by reading queries from the query queue and processing them until a termination signal is received.

        This method starts a separate thread for result retrieval and processes incoming queries from the sample queue.
        It continues to process queries until a `None` value is retrieved from the queue, indicating termination.
        Upon termination, it sends a termination signal to subsequent stages and joins all threads.

        If `serialize_queries` is True on the pipeline config, the loop
        waits for the previous query to exit (queries_processed catches up
        to queries_sent) before admitting the next one. There is also a
        bounded wait at shutdown — see DRAIN_TIMEOUT below — so a dropped
        query never causes an indefinite hang.

            query_queue (Queue): The queue containing queries from the load generator.
            event (Event): An event used for synchronization between the pipeline and the load generator.
        """

        queries_sent = 0
        self.queries_processed = 0

        result_retrieval_thread = Thread(
            target=self.retrieve_results, args=[event]
        )
        result_retrieval_thread.start()

        dataset_splits = self.get_dataset_splits()
        epoch_dict = {split: 0 for split in dataset_splits}

        serialize = bool(getattr(self._pipeline_config, "serialize_queries", False))

        # Per-query drain budget (s) for serialized mode and for shutdown.
        # If a query gets dropped mid-pipeline we don't want to deadlock
        # waiting for queries_processed to catch up forever.
        DRAIN_TIMEOUT = 600.0

        def _wait_for_drain(target_processed: int, timeout: float) -> bool:
            """Block until queries_processed >= target_processed or timeout."""
            from time import monotonic
            deadline = monotonic() + timeout
            while self.queries_processed < target_processed:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self._logger.warning(
                        "%s, pipeline, drain, timeout, sent=%d, processed=%d",
                        self.name, queries_sent, self.queries_processed,
                    )
                    return False
                event.wait(min(0.1, remaining))
                event.clear()
            return True

        while True:
            query: Query | None = query_queue.get()

            # check if done
            if query is None:
                # Wait for all in-flight queries to finish, but with a bound
                # so a silently-dropped query (e.g. from a buggy polling
                # policy or a stage exception) doesn't cause indefinite hang.
                _wait_for_drain(queries_sent, DRAIN_TIMEOUT)

                # send the termination element to the following stages and join the threads
                for input_queue in self._input_queues:
                    input_queue.put(None)
                # Bounded join — a stuck stage shouldn't hold the whole
                # pipeline in shutdown. main.py force-exits afterwards,
                # so leaked threads are reclaimed by process exit.
                self.join_threads(timeout=10.0)
                break

            # log the start of pipeline execution for the query
            if query.batch == 0:
                epoch_dict[query.split] += 1

            query.epoch = epoch_dict[query.split]

            self._logger.info(
                "%s, pipeline - %s, run, start, %d, %.6f, %d, %d",
                self.name,
                query.split,
                query.query_id,
                query.query_submitted_timestamp,
                query.epoch,
                query.batch + 1,
            )

            out_flow_id = uuid.uuid4()
            with mlflow.start_span(
                name="pipeline query",
                attributes={
                    "in_flow_id": str(query.out_flow_id) if query.out_flow_id else None,
                    "out_flow_id": str(out_flow_id),
                    "thread_id": threading.get_ident(),
                    "pipeline": self.name,
                    "epoch": query.epoch,
                    "split": query.split,
                    "batch": query.batch,
                    "query_id": query.query_id,
                },
            ):
                query.out_flow_id = out_flow_id
                # populate the pipeline input queues / start pipeline execution
                for input_queue in self._input_queues:
                    input_queue.put(query)

            queries_sent += 1

            # In serialized mode, wait for THIS query to exit before
            # admitting the next one. Bounded so a dropped query
            # surfaces as a warning rather than a hang.
            if serialize:
                _wait_for_drain(queries_sent, DRAIN_TIMEOUT)

        # Bounded join on the result-retrieval thread too. If the End
        # stage somehow never delivered its None terminator, we'd otherwise
        # hang here forever. main.py force-exits to clean up.
        result_retrieval_thread.join(timeout=10.0)
