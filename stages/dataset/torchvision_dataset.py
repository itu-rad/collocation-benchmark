import torchvision.datasets

from stages.stage import Stage


class TorchVisionDataset(Stage):
    dataset = None
    dataset_name = ""
    split = "val"

    def __init__(self, stage_config: dict):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Configuration, which includes the name of the dataset and split

        Raises:
            Exception: Missing model name.
            Exception: Model checkpoint path is required for inference.
        """
        super().__init__(stage_config)

        self.dataset_name = stage_config.get("dataset_name", None)
        if self.dataset_name is None:
            raise Exception("Missing dataset name.")
        self.dataset = torchvision.datasets.__dict__.get(self.dataset_name, None)
        if self.dataset is None:
            raise Exception(f"Could not find dataset {self.dataset_name}.")
        self.split = stage_config.get("split", "val")
        self.dataset = self.dataset(
            root=f"data/{self.dataset_name}", split=self.split, download=True
        )

    def prepare(self):
        """Build the model according to the config and load the weights"""

        return self.dataset

    def run(self, data):
        """Run inference query

        Args:
            data (Tensor): indices
        """

        return self.dataset[data]
