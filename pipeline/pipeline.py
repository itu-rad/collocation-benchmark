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
        self._pipeline_config = pipeline_config
        self.name = self._pipeline_config.name
        stage_config = pipeline_config.stages
        stage_config_dict = {stage.id: stage for stage in stage_config}

        self._input_queues: list[Queue] = []
        self._output_queues: list[Queue] = []

        self.stages = self._create_stages(stage_config_dict)

        self._populate_stages_with_stage_dict()
        self._populate_queues()

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
        logging.info("%s, pipeline, prepare, start", self.name)

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

        logging.info("%s, pipeline, prepare, end", self.name)

    def join_threads(self) -> None:
        """
        Wait for all the stage threads to finish.

        This method is useful for shutting down the pipeline, since it waits for all the stages to finish.
        """
        for stage in self.stages.values():
            stage.join_thread()

    def retrieve_results(self, event: Event) -> None:
        """
        Retrieve the results from all the output queues of the pipeline.
        """
        while True:
            for output_queue in self._output_queues:
                # non-blocking retrieval from the queues
                # if the queue is empty, simply move on
                try:
                    new_query: Query | None = output_queue.get_nowait()
                except Empty:
                    continue

                # if terminating character is received, return
                if not new_query:
                    return

                # log the end of pipeline execution
                logging.info(
                    "%s, pipeline - %s, run, end, %d, %.6f, %d, %d",
                    self.name,
                    new_query.split,
                    new_query.query_id,
                    new_query.query_submitted_timestamp,
                    new_query.epoch,
                    new_query.batch + 1,
                )

                event.set()

    def run(self, sample_queue: Queue, event: Event) -> None:
        """
        Read the queries from the loadgen queue and execute the pipeline until a terminating element is retrieved.

        Args:
            sample_queue (Queue): The queue with the queries from the load generator.
            event (Event): An event that is used for synchronization between the pipeline and the load generator.
        """
        result_retrieval_thread = Thread(target=self.retrieve_results, args=[event])
        result_retrieval_thread.start()

        dataset_splits = self.get_dataset_splits()
        epoch_dict = {split: 0 for split in dataset_splits}

        while True:
            query: Query | None = sample_queue.get()

            # check if done
            if query is None:
                # if so, send the termination element to the following stages and join the threads
                for input_queue in self._input_queues:
                    input_queue.put(None)
                self.join_threads()
                break

            # log the start of pipeline execution for the query
            if query.batch == 0:
                epoch_dict[query.split] += 1

            query.epoch = epoch_dict[query.split]

            logging.info(
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

        result_retrieval_thread.join()
