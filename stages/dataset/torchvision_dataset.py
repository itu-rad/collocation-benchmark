import torchvision.datasets
from torchvision.transforms import v2
from torchvision.models import get_weight
import os

from stages.stage import Stage, log_phase_single


class TorchVisionDataset(Stage):
    datasets = dict()
    dataset_name = ""
    split = ["val"]
    batch_size = 1

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
            raise ValueError("Missing dataset name.")
        self.dataset = torchvision.datasets.__dict__.get(dataset_name, None)
        if self.dataset is None:
            raise Exception(f"Could not find dataset {dataset_name}.")
        split = stage_config.get("split", ["val"])
        dataset_downloaded = os.path.exists(
            os.path.join(os.getcwd(), "data", dataset_name)
        )
        # print(f"Dataset downloaded? {dataset_downloaded}")
        weights_name = stage_config.get("weights", None)
        if weights_name is None:
            self.datasets = {
                x: self.dataset(
                    root=f"data/{self.dataset_name}",
                    split=x,
                    download=(not dataset_downloaded),
                    transform=v2.ToTensor(),
                )
                for x in split
            }
        else:
            weights = get_weight(weights_name)
            preprocess = weights.transforms()
            self.datasets = {
                x: self.dataset(
                    root=f"data/{self.dataset_name}",
                    split=x,
                    download=(not dataset_downloaded),
                    transform=preprocess,
                )
                for x in split
            }

        self.batch_size = stage_config.get("batch_size", 1)

    def prepare(self):
        """Build the model according to the config and load the weights"""
        super(TorchVisionDataset, self).prepare()

    def get_datasets(self):
        return self.datasets

    def get_num_batches(self):
        return {k: len(v) // self.batch_size for (k, v) in self.datasets.items()}

    def run(self):
        """Run inference query

        Args:
            data (Tensor): indices
        """
        # inputs = data.get("data", None)
        # if inputs is None:
        #     raise Exception("Did not receive any input from the previous stage")

        # split = data.get("split", "val")
        # data["data"] = self.datasets[split][inputs]
        # return data
        while True:
            inputs = self.get_next_from_queues()
            if self.is_done(inputs):
                self.push_to_output(list(inputs.values())[0])
                break

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "start")
            self.push_to_output(list(inputs.values())[0])
            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "end")
