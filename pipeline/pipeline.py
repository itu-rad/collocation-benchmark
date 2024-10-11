import logging
from queue import Queue
from typing import Union

from stages import Stage
from utils.component import get_stage_component


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    def __init__(self, pipeline_config):

        print("Got pipeline config", pipeline_config)

        self.name = pipeline_config.get("name", "Unknown pipeline name")
        self.stage_dict = {}

        stage_config = pipeline_config.get("stages", None)
        stage_config_dict = {stage.get("id", None): stage for stage in stage_config}

        # find the input stages
        input_stages_idx = pipeline_config.get("inputs", [])
        if len(input_stages_idx) < 1:
            raise ValueError(
                "At least one input stage (usually a dataset) required for the pipeline."
            )

        # initialize the input stages
        self.input_stages: dict[str, Stage] = dict()
        for input_stage_idx in input_stages_idx:
            input_stage_config = stage_config_dict.get(input_stage_idx, None)
            if input_stage_config is None:
                raise ValueError("Input stage not found")
            stage_obj = get_stage_component(input_stage_config, pipeline_config)
            self.stage_dict[stage_obj.get_id()] = stage_obj
            self.input_stages[input_stage_idx] = stage_obj

        # find the output stages
        output_stages_idx = pipeline_config.get("outputs", [])
        if len(output_stages_idx) < 1:
            raise ValueError("At least one output stage required for the pipeline.")

        # initialize the output stages
        self.output_stages: dict[str, Stage] = dict()
        for output_stage_idx in output_stages_idx:
            output_stage_config = stage_config_dict.get(output_stage_idx, None)
            if output_stage_config is None:
                raise ValueError("output stage not found")
            stage_obj = get_stage_component(output_stage_config, pipeline_config)
            self.stage_dict[stage_obj.get_id()] = stage_obj
            self.output_stages[output_stage_idx] = stage_obj

        # initialize the intermediate (rest of) stages
        self.intermediate_stages: dict[str, Stage] = dict()
        for stage in stage_config:
            stage_idx = stage.get("id", None)
            if stage_idx is None:
                raise ValueError("All stages are required to have IDs.")
            if stage_idx not in input_stages_idx and stage_idx not in output_stages_idx:
                stage_obj = get_stage_component(stage, pipeline_config)
                self.stage_dict[stage_obj.get_id()] = stage_obj
                self.intermediate_stages[stage_idx] = stage_obj

        # populate the stages with the stage dictionary for dynamic method invocation
        for stage in self.stage_dict.values():
            stage.add_stage_dict(self.stage_dict)

        # populate the input stages with the outputs
        for idx, input_stage in self.input_stages.items():
            outputs = stage_config_dict[idx].get("outputs", [])
            if len(outputs) < 1:
                raise ValueError("Output stage ids required for an input stage.")
            for output_idx in outputs:
                if output_idx in self.output_stages:
                    input_stage.add_next_stage(self.output_stages[output_idx])
                else:
                    input_stage.add_next_stage(self.intermediate_stages[output_idx])

        # populate the output stages with the inputs
        for idx, output_stage in self.output_stages.items():
            inputs = stage_config_dict[idx].get("inputs", [])
            if len(inputs) < 1:
                raise ValueError("Input stage ids required for an output stage.")
            for input_idx in inputs:
                if input_idx in self.input_stages:
                    output_stage.add_previous_stage(self.input_stages[input_idx])
                else:
                    output_stage.add_previous_stage(self.intermediate_stages[input_idx])

        # populate intermediate stages with their inputs and outputs
        for idx, stage in self.intermediate_stages.items():
            inputs = stage_config_dict[idx].get("inputs", [])
            if len(inputs) < 1:
                raise ValueError("Input IDs required for all but input stages.")
            # find the input of the stage in input or intermediate stages
            for input_idx in inputs:
                input_stage = self.input_stages.get(input_idx, None)
                if input_stage is None:
                    input_stage = self.intermediate_stages.get(input_idx, None)
                if input_stage is None:
                    raise ValueError("Stage with input ID not found.")
                stage.add_previous_stage(input_stage)
            outputs = stage_config_dict[idx].get("outputs", [])
            if len(outputs) < 1:
                raise ValueError("Output IDs required for all but output stages.")
            # find the output of the stage in output or intermediate stages
            for output_idx in outputs:
                output_stage = self.output_stages.get(output_idx, None)
                if output_stage is None:
                    output_stage = self.intermediate_stages.get(output_idx, None)
                if output_stage is None:
                    raise ValueError("Stage with output ID not found.")
                stage.add_next_stage(output_stage)

        self.input_queues: list[Queue] = []
        # setup output queues
        self.output_queues: list[Queue] = []
        for _, output_stage in self.output_stages.items():
            queue = Queue()
            self.output_queues.append(queue)
            output_stage.set_output_queue(queue)

    def get_dataset_length(self) -> dict[int, dict[str, int]]:
        """Get the size of the datasets in input stages

        Returns:
            dict[int, dict[str, int]]: Dictionary with batch sizes of each input stage
        """
        return {
            i_idx: input_stage.get_num_batches()
            for i_idx, input_stage in self.input_stages.items()
        }

    def prepare(self):
        """Run prepare functions of the stages of the pipelines which contain functionality,
        such as loading/building models."""
        logging.info("%s, pipeline, prepare, start", self.name)
        # create input queues first (needs to be done separately,
        # because prior stages depend on subsequent having created queues)
        for stage in self.input_stages.values():
            stage.create_input_queues()
        for stage in self.intermediate_stages.values():
            stage.create_input_queues()
        for stage in self.output_stages.values():
            stage.create_input_queues()

        for stage in self.input_stages.values():
            stage.prepare()
        for stage in self.intermediate_stages.values():
            stage.prepare()
        for stage in self.output_stages.values():
            stage.prepare()

        # get all the input queues of the pipeline
        for _, input_stage in self.input_stages.items():
            self.input_queues.append(input_stage.get_input_queues()[0])

        logging.info("%s, pipeline, prepare, end", self.name)

    def join_threads(self):
        """join the stage threads upon end of exection"""
        for stage in self.input_stages.values():
            stage.join_thread()
        for stage in self.intermediate_stages.values():
            stage.join_thread()
        for stage in self.output_stages.values():
            stage.join_thread()

    def run(self, sample_queue, event):
        """Read the queries from the loadgen queue and execute the pipeline until a terminating element is retrieved."""

        epoch_dict = {"train": 0, "val": 0}

        while True:
            data = sample_queue.get()

            # check if done
            if data is None:
                for input_queue in self.input_queues:
                    input_queue.put(data)
                self.join_threads()
                break

            # log the start of execution
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
            for input_queue in self.input_queues:
                input_queue.put(data)

            # retrieve the pipeline results
            for output_queue in self.output_queues:
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

            # release the lock such that for synchronyzed schedulers
            event.set()
