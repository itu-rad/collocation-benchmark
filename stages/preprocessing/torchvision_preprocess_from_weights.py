import torch
from torchvision.models import get_weight

from stages.stage import Stage, log_phase, log_phase_single


class TorchVisionPreprocessFromWeights(Stage):
    weight_name = ""
    weights = None
    preprocess = None

    def __init__(self, stage_config, parent_name):
        """nitialize the stage by parsing the stage configuration.

        Args:
            stage_config (_type_): Stage configuration, such as the TorchVision weight names
            to extract preprocessing steps from.
        """
        super().__init__(stage_config, parent_name)
        stage_config = stage_config.get("config", {})

        self.weight_name = stage_config.get("weights", None)
        if self.weight_name is None:
            raise ValueError("Weight name is required.")

    @log_phase
    def prepare(self):
        """Build the dataloader"""
        super(TorchVisionPreprocessFromWeights, self).prepare()
        self.weights = get_weight(self.weight_name)
        self.preprocess = self.weights.transforms()

    @log_phase
    def run(self):
        """Run inference query

        Args:
            data (Tensor): Input data
        """
        while True:
            data_from_queues = self.get_next_from_queues()

            if self.is_done(data_from_queues):
                self.push_to_output(None)
                break

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "start")

            data_from_first = list(data_from_queues.values())[0]
            inputs = data_from_first.get("data", None)
            if inputs is None:
                raise ValueError("Did not receive any input from the previous stage")

            # NOTE: This is not optimal, but necessary if we want to measure loading and
            # preprocessing separately. Better solution would be to put the preprocessing into
            # the dataset definition, which combines the two latencies in the dataloader's execution.
            preprocessed_inputs = []
            labels = []
            for [sample, label] in inputs:
                preprocessed_sample = self.preprocess(sample)
                preprocessed_inputs.append(preprocessed_sample)
                labels.append(label)

            preprocessed_inputs = torch.stack(preprocessed_inputs)
            labels = torch.Tensor(labels)

            data_from_first["data"] = [preprocessed_inputs, labels]
            self.push_to_output(data_from_first)
            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "end")
