import os
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.models import get_weight
from torchvision.datasets import VisionDataset

from stages.stage import Stage, log_phase
from utils.schemas import Query, StageModel, PipelineModel
from utils.component import get_component


class PreloadedDataset(VisionDataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.data = []
        for data_point in dataset:
            self.data.append(data_point)

    def __getitem__(self, index):
        x, y = self.data[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.data)


class TorchVisionDataLoader(Stage):
    """This stage contains the Torch dataloading functionality with the possibility
    of prefetching and preprocessing the data.
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        dataset_cls = get_component(self.extra_config["dataset"]["component"])

        self._splits = self.extra_config.get("split", ["val"])
        dataset_name = self.extra_config["dataset"]["component"].split(".")[-1]
        dataset_root = os.path.join(
            os.getcwd(), "tmp", "torchvision_dataset", dataset_name
        )
        dataset_downloaded = os.path.isdir(dataset_root) and os.listdir(dataset_root)

        print(f"Dataset downloaded: {dataset_downloaded}")

        # Handle ImageNet validation dataset specifically
        if dataset_name.lower() == "imagenet" and "val" in self._splits:
            print("detected imagenet val")
            if not dataset_downloaded:
                print("downloading imagenet val")
                self._download_imagenet_val(dataset_root)

        weights_name = self.extra_config["dataset"].get("weights", None)
        if weights_name is None:
            transform = v2.ToTensor()
        else:
            weights = get_weight(weights_name)
            transform = weights.transforms()
            print(f"Transform: {transform}")

        self._datasets: dict[str, VisionDataset] = {
            x: dataset_cls(
                root=dataset_root,
                split=x,
                **(
                    {"download": not dataset_downloaded}
                    if dataset_name.lower() != "imagenet"
                    else {}
                ),
                transform=transform,
            )
            for x in self._splits
        }

        self._batch_size = self.extra_config.get("batch_size", 1)
        self._dataloaders = {}
        self._dataloader_iter = None

    def _download_imagenet_val(self, dataset_root: str):
        """Download the ImageNet validation dataset using Kaggle API if not already present.

        Args:
            dataset_root (str): Path to the dataset root directory.
        """
        os.makedirs(dataset_root, exist_ok=True)
        kaggle_dataset = "titericz/imagenet1k-val"  # Kaggle dataset identifier

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError(
                "Kaggle API is not installed. Install it using 'pip install kaggle'."
            )

        print(
            f"Downloading ImageNet validation dataset from Kaggle to {dataset_root}..."
        )

        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(kaggle_dataset, path=dataset_root, unzip=True)
        print("Download complete. Ensure the dataset is properly structured.")

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches for each dataset split
        """
        return {k: len(v) // self._batch_size for (k, v) in self._datasets.items()}

    def _preload_dataset(self) -> None:
        for key in self._datasets.keys():
            # get the old transform
            transform = self._datasets[key].transform
            # remove transform from preloading
            self._datasets[key].transform = None
            # add transform to cached dataset
            self._datasets[key] = PreloadedDataset(self._datasets[key], transform)

    def _custom_collate_fn(self, data):
        """Custom collate identity function, allowing us
        to perform preprocessing in a separate stage.
        """
        print("Custom collate function called")
        return data

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()

        # preload the dataset if necessary
        if self.extra_config.get("preload", False):
            self._preload_dataset()

        num_workers = self.extra_config.get("num_workers", 0)

        if self.extra_config.get("preprocess", True):
            self._dataloaders = {
                k: DataLoader(
                    v,
                    self._batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True,
                )
                for (k, v) in self._datasets.items()
            }
        else:
            self._dataloaders = {
                k: DataLoader(
                    v,
                    self._batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=self._custom_collate_fn,
                )
                for (k, v) in self._datasets.items()
            }

    def run(self, query: Query) -> dict[int, Query]:
        if query.batch == 0:
            self._dataloader_iter = iter(self._dataloaders[query.split])
        next_batch = next(self._dataloader_iter)
        query.data = next_batch
        output = {idx: query for idx in self.output_queues}
        return output
