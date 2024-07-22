import torchvision.datasets
from torchvision.transforms import v2
from torchvision.models import get_weight
import os

from stages.stage import Stage


class TorchVisionDataset(Stage):
    dataset = None
    dataset_name = ""
    split = "val"

    def __init__(self, stage_config: dict, parent_name):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Configuration, which includes the name of the dataset and split

        Raises:
            Exception: Missing model name.
            Exception: Model checkpoint path is required for inference.
        """
        super().__init__(stage_config, parent_name)
        stage_config = stage_config.get("config", {})

        dataset_name = stage_config.get("dataset_name", None)
        if dataset_name is None:
            raise Exception("Missing dataset name.")
        self.dataset = torchvision.datasets.__dict__.get(dataset_name, None)
        if self.dataset is None:
            raise Exception(f"Could not find dataset {dataset_name}.")
        split = stage_config.get("split", "val")
        dataset_downloaded = os.path.exists(
            os.path.join(os.getcwd(), "data", dataset_name)
        )
        print(f"Dataset downloaded? {dataset_downloaded}")
        weights_name = stage_config.get("weights", None)
        if weights_name is None:
            self.dataset = self.dataset(
                root=f"data/{self.dataset_name}",
                split=split,
                download=(not dataset_downloaded),
                transform=v2.ToTensor(),
            )
        else:
            weights = get_weight(weights_name)
            preprocess = weights.transforms()
            self.dataset = self.dataset(
                root=f"data/{self.dataset_name}",
                split=split,
                download=(not dataset_downloaded),
                transform=preprocess,
            )

    def prepare(self):
        """Build the model according to the config and load the weights"""

        return self.dataset

    def get_dataset(self):
        return self.dataset

    def run(self, data):
        """Run inference query

        Args:
            data (Tensor): indices
        """
        inputs = data.get("data", None)
        if inputs is None:
            raise Exception("Did not receive any input from the previous stage")

        data["data"] = self.dataset[inputs]
        return data
