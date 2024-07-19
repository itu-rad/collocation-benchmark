from torch.utils.data import DataLoader
import torch

from stages.stage import Stage, log_phase


class TorchFromDataset(Stage):
    dataset = None
    dataloader = None
    batch_size = 1
    num_workers = 0
    preprocessing = False

    def __init__(self, stage_config):
        """nitialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config)
        stage_config = stage_config.get("config", {})

        self.dataset = stage_config.get("dataset", None)
        if self.dataset is None:
            raise Exception(
                "Dataset is missing. Dataset is required to build a dataloader."
            )

        self.batch_size = stage_config.get("batch_size", 1)
        self.num_workers = stage_config.get("num_workers", 0)
        self.preprocessing = stage_config.get("preprocessing", False)

    def my_collate_fn(self, data):
        return data

    @log_phase
    def prepare(self):
        """Build the dataloader"""
        if self.preprocessing:
            self.dataloader = DataLoader(
                self.dataset,
                self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
            )
        else:
            self.dataloader = DataLoader(
                self.dataset,
                self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
                collate_fn=self.my_collate_fn,
            )
        self.dataloader = iter(self.dataloader)

    @log_phase
    def run(self, data):
        """Run inference query

        Args:
            data (Tensor): Input data
        """
        data["data"] = next(self.dataloader)
        return data
