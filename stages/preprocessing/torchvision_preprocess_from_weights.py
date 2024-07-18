from torchvision.models import get_weight

from stages.stage import Stage, log_phase


class TorchVisionPreprocessFromWeights(Stage):
    weight_name = ""
    weights = None
    preprocess = None

    def __init__(self, stage_config):
        """nitialize the stage by parsing the stage configuration.

        Args:
            stage_config (_type_): Stage configuration, such as the TorchVision weight names
            to extract preprocessing steps from.
        """
        super().__init__(stage_config)
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
        return self.preprocess(data)
