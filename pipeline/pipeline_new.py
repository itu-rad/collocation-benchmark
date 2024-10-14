import logging
from queue import Queue
from typing import Union, Any
from threading import Event

from stages import Stage
from utils.component import get_stage_component


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    def __init__(self, pipeline_config):
        """Initialize the pipeline by parsing the pipeline configuration.

        Args:
            pipeline_config (dict): The configuration of the pipeline.
        """
        print("Config:", pipeline_config)

        self._pipeline_config: dict = pipeline_config
        self.name: str = self._pipeline_config.get("name", "Unknown pipeline name")
        stage_config: list = pipeline_config.get("stages", [])
        stage_config_dict: dict[int, Any] = {
            stage.get("id", None): stage for stage in stage_config
        }

        self._input_queues: list[Queue] = []
        self._output_queues: list[Queue] = []

        self.stages: list[Stage] = self._create_stages(stage_config_dict)

        self._populate_stages_with_stage_dict()
        self._populate_queues()

    def _create_stages(self, stage_config_dict) -> dict[int, Stage]:
        """Create a dictionary of stage objects from the given configuration.

        Args:
            stage_config_dict (dict): A dictionary where the keys are stage IDs and the values are the configurations of the stages.

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

    def prepare(self) -> None:
        """
        Prepare all the stages in the pipeline.

        This includes creating input queues, output queues, as well as calling the prepare method of each stage.
        """
        logging.info("%s, pipeline, prepare, start", self.name)

        # get all the input queues of the pipeline
        for input_stage_idx in self._pipeline_config.get("inputs", []):
            self._input_queues.append(self.stages[input_stage_idx].get_input_queue(-1))

        # create output queue for each output stage of the pipeline and pass it to the output stage
        for output_stage_idx in self._pipeline_config.get("outputs", []):
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

    def run(self, sample_queue: Queue, event: Event) -> None:
        """
        Read the queries from the loadgen queue and execute the pipeline until a terminating element is retrieved.

        Args:
            sample_queue (Queue): The queue with the queries from the load generator.
            event (Event): An event that is used for synchronization between the pipeline and the load generator.
        """
        # TODO: Make this naming-agnostic. Get the names of the splits from the dataset / dataloader
        epoch_dict = {"train": 0, "val": 0}

        while True:
            data = sample_queue.get()

            # check if done
            if data is None:
                # if so, send the termination element to the following stages and join the threads
                for input_queue in self._input_queues:
                    input_queue.put(None)
                self.join_threads()
                break

            # TODO: Revisit the splits and make it naming agnostic - should get them from the dataset / dataloader
            split = data.get("split", "val")
            submitted = data.get("query_submitted", 0)
            batch_idx = data.get("batch", 0)
            if batch_idx == 0:
                epoch_dict[split] += 1
            logging.info(
                "%s, pipeline - %s, run, start, %.6f, %d, %d",
                self.name,
                split,
                submitted,
                epoch_dict[split],
                batch_idx + 1,
            )

            # populate the pipeline input queues / start pipeline execution
            for input_queue in self._input_queues:
                input_queue.put(data)

            # retrieve the pipeline results
            for output_queue in self._output_queues:
                _ = output_queue.get(data)

            # log the end of pipeline execution
            logging.info(
                "%s, pipeline - %s, run, end, %.6f, %d, %d",
                self.name,
                split,
                submitted,
                epoch_dict[split],
                batch_idx + 1,
            )

            # release the lock for synchronyzed load generators
            event.set()
