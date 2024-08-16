import logging
from queue import Queue

from stages.stage_registry import STAGE_REGISTRY


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    name = ""
    datasets = None
    stages = []
    is_training = []
    input_stages = dict()
    middle_stages = dict()
    output_stages = dict()
    output_queues = []
    input_queues = []

    def __init__(self, pipeline_config):

        print("Got pipeline config", pipeline_config)

        self.name = pipeline_config.get("name", "Unknown pipeline name")

        stage_config = pipeline_config.get("stages", None)
        stage_config_dict = {stage.get("id", None): stage for stage in stage_config}

        input_stages_idx = pipeline_config.get("inputs", [])
        if len(input_stages_idx) < 1:
            raise ValueError(
                "At least one input stage (usually a dataset) required for the pipeline."
            )

        # Find the input stages and initialize them
        for input_stage_idx in input_stages_idx:
            input_stage_config = stage_config_dict.get(input_stage_idx, None)
            if input_stage_config is None:
                raise ValueError("Input stage not found")
            self.input_stages[input_stage_idx] = STAGE_REGISTRY[
                input_stage_config["type"]
            ][input_stage_config["module_name"]](input_stage_config, self.name)

        output_stages_idx = pipeline_config.get("outputs", [])
        if len(output_stages_idx) < 1:
            raise ValueError(
                "At least one output stage (usually a dataset) required for the pipeline."
            )

        # Find the output stages and initialize them
        for output_stage_idx in output_stages_idx:
            output_stage_config = stage_config_dict.get(output_stage_idx, None)
            if output_stage_config is None:
                raise ValueError("output stage not found")
            self.output_stages[output_stage_idx] = STAGE_REGISTRY[
                output_stage_config["type"]
            ][output_stage_config["module_name"]](output_stage_config, self.name)

        # initialize the rest of the stages
        self.intermediate_stages = dict()
        for stage in stage_config:
            stage_idx = stage.get("id", None)
            if stage_idx is None:
                raise ValueError("All stages are required to have IDs.")
            if stage_idx not in input_stages_idx and stage_idx not in output_stages_idx:
                self.intermediate_stages[stage_idx] = STAGE_REGISTRY[stage["type"]][
                    stage["module_name"]
                ](stage, self.name)

        # populate the input stages with the outputs
        for idx, input_stage in self.input_stages.items():
            outputs = stage_config_dict[idx].get("outputs", [])
            if len(outputs) < 1:
                raise ValueError("Output stage ids required for an input stage.")
            for output_idx in outputs:
                input_stage.add_next_stage(self.intermediate_stages[output_idx])

        # populate the output stages with the inputs
        for idx, output_stage in self.output_stages.items():
            inputs = stage_config_dict[idx].get("inputs", [])
            if len(inputs) < 1:
                raise ValueError("Input stage ids required for an output stage.")
            for input_idx in inputs:
                output_stage.add_previous_stage(self.intermediate_stages[input_idx])

        # populate other stages with the previous and next stages
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

        # setup output queues
        for _, output_stage in self.output_stages.items():
            queue = Queue()
            self.output_queues.append(queue)
            output_stage.set_output_queue(queue)

    def get_dataset_length(self):
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
        for stage in self.input_stages.values():
            stage.join_thread()
        for stage in self.intermediate_stages.values():
            stage.join_thread()
        for stage in self.output_stages.values():
            stage.join_thread()

    def run(self, sample_queue, event):
        """Invoke the pipeline and pass data between stages."""

        epoch_dict = {"train": 0, "val": 0}

        while True:
            data = sample_queue.get()

            # check if done
            if data is None:
                for input_queue in self.input_queues:
                    input_queue.put(data)
                self.join_threads()
                break

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

            logging.info(
                "%s, pipeline - %s, run, end, %.6f, %d, %d",
                self.name,
                split,
                submitted,
                epoch_dict[split],
                batch_idx + 1,
            )

            event.set()
