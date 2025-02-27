import os
import torchvision.datasets
from torchvision.transforms import v2
from torchvision.models import get_weight
from torchvision.datasets import VisionDataset


from stages.stage import Stage, log_phase, log_phase_single
from utils.schemas.pipeline import PipelineModel
from utils.schemas.query import Query
from utils.schemas.stage import StageModel


class PreloadedDataset(VisionDataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.data = []
        for idx in range(len(dataset)):
            self.data.append(dataset[idx])

    def __getitem__(self, index):
        x, y = self.data[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.data)


class TorchVisionDataset(Stage):
    """This stage defines and exposes a TorchVision dataset.
    This stage does not perform any action during pipeline execution and is only
    used for dataloader initialization.
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Configuration, which includes the name of the dataset and split

        Raises:
            Exception: Missing model name.
            Exception: Model checkpoint path is required for inference.
        """
        super().__init__(stage_config, pipeline_config)

        self.preload = self.extra_config.get("preload", False)

        dataset_name = self.extra_config.get("dataset_name", None)
        if dataset_name is None:
            raise ValueError("Missing dataset name.")
        self.dataset = torchvision.datasets.__dict__.get(dataset_name, None)
        if self.dataset is None:
            raise Exception(f"Could not find dataset {dataset_name}.")
        split = self.extra_config.get("split", ["val"])
        dataset_downloaded = os.path.exists(
            os.path.join(os.getcwd(), "data", dataset_name)
        )
        # print(f"Dataset downloaded? {dataset_downloaded}")
        weights_name = self.extra_config.get("weights", None)
        if weights_name is None:
            self.datasets: dict[str, VisionDataset] = {
                x: self.dataset(
                    root=f"data/{dataset_name}",
                    split=x,
                    download=(not dataset_downloaded),
                    transform=v2.ToTensor(),
                )
                for x in split
            }
        else:
            weights = get_weight(weights_name)
            preprocess = weights.transforms()
            self.datasets: dict[str, VisionDataset] = {
                x: self.dataset(
                    root=f"data/{dataset_name}",
                    split=x,
                    download=(not dataset_downloaded),
                    transform=preprocess,
                )
                for x in split
            }

        self.batch_size = self.extra_config.get("batch_size", 1)

    def get_datasets(self):
        """Getter for the datasets

        Returns:
            dict[str, torchvision.datasets.VisionDataset]: dictionary of datasets (train and/or val)
        """
        return self.datasets

    def get_num_batches(self):
        """Calculate the number of batches for each dataset

        Returns:
            dict[str, int]: dictionary with number of batches for each dataset
        """
        return {k: len(v) // self.batch_size for (k, v) in self.datasets.items()}

    @log_phase
    def prepare(self):
        """Preload dataset (specifically useful for modelling inference)"""
        super().prepare()

        if not self.preload:
            return

        for key in self.datasets.keys():
            # get the old transform
            transform = self.datasets[key].transform
            # remove transform from preloading
            self.datasets[key].transform = None
            # add transform to cached dataset
            self.datasets[key] = PreloadedDataset(self.datasets[key], transform)
