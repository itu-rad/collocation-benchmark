import torch
from torchvision.models import get_weight

from stages.stage import Stage, log_phase


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
            raise Exception("Weight name is required.")

    @log_phase
    def prepare(self):
        """Build the dataloader"""
        self.weights = get_weight(self.weight_name)
        self.preprocess = self.weights.transforms()

    @log_phase
    def run(self, data):
        """Run inference query

        Args:
            data (Tensor): Input data
        """
        inputs = data.get("data", None)
        if inputs is None:
            raise Exception("Did not receive any input from the previous stage")

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

        data["data"] = [preprocessed_inputs, labels]

        return data
