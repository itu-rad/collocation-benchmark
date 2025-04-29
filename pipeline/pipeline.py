import logging
from queue import Queue, Empty
from threading import Event, Thread

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

    def join_threads(self) -> None:
        """
        Wait for all the stage threads to finish.

        This method is useful for shutting down the pipeline, since it waits for all the stages to finish.
        """
        for stage in self.stages.values():
            stage.join_thread()

    def retrieve_results(self, event: Event, queries_processed: int) -> None:
        """
        This method continuously checks all output queues for new queries. It performs
        non-blocking retrieval from the queues and processes the queries if available.
        If a terminating character (None) is received, the method returns and stops
        further processing. Additionally, it logs the end of the pipeline execution
        and sets the provided event to signal completion.

        Args:
            event (Event): An event object used to signal the completion of the pipeline.
        """

        while True:
            for output_queue in self._output_queues:
                # non-blocking retrieval from the queues
                # if the queue is empty, simply move on
                try:
                    new_query: Query | None = output_queue.get(timeout=0.1)
                except Empty:
                    continue

                # if terminating character is received, return
                if not new_query:
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

                queries_processed += 1
                event.set()

    def run(self, query_queue: Queue, event: Event) -> None:
        """
        Executes the pipeline by reading queries from the query queue and processing them until a termination signal is received.

        This method starts a separate thread for result retrieval and processes incoming queries from the sample queue.
        It continues to process queries until a `None` value is retrieved from the queue, indicating termination.
        Upon termination, it sends a termination signal to subsequent stages and joins all threads.

            query_queue (Queue): The queue containing queries from the load generator.
            event (Event): An event used for synchronization between the pipeline and the load generator.
        """

        queries_sent = 0
        queries_procesed = 0

        result_retrieval_thread = Thread(
            target=self.retrieve_results, args=[event, queries_procesed]
        )
        result_retrieval_thread.start()

        dataset_splits = self.get_dataset_splits()
        epoch_dict = {split: 0 for split in dataset_splits}

        while True:
            query: Query | None = query_queue.get()

            # check if done
            if query is None:
                # only terminate once all of the queries have been processed in order to protect the graph from circular dependencies during termination
                while queries_sent > queries_procesed:
                    # print(
                    #     "Waiting for all queries to be processed, sent: %d, processed: %d",
                    #     queries_sent,
                    #     queries_procesed,
                    # )
                    # wait for a query to be processed
                    event.wait(0.1)

                # if so, send the termination element to the following stages and join the threads
                for input_queue in self._input_queues:
                    input_queue.put(None)
                self.join_threads()
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

            # populate the pipeline input queues / start pipeline execution
            for input_queue in self._input_queues:
                input_queue.put(query)

            queries_sent += 1

        result_retrieval_thread.join()
