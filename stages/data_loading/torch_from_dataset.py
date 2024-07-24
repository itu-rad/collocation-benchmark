from torch.utils.data import DataLoader

from stages.stage import Stage, log_phase


class TorchFromDataset(Stage):
    datasets = dict()
    dataloaders = dict()
    batch_size = 1
    num_workers = 0
    preprocessing = False

    def __init__(self, stage_config, parent_name):
        """nitialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, parent_name)
        stage_config = stage_config.get("config", {})

        self.datasets = stage_config.get("dataset", dict())
        if self.datasets is None:
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
            self.dataloaders = {
                k: DataLoader(
                    v,
                    self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=True,
                    drop_last=True,
                )
                for (k, v) in self.datasets.items()
            }
        else:
            self.dataloaders = {
                k: DataLoader(
                    v,
                    self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=self.my_collate_fn,
                )
                for (k, v) in self.datasets.items()
            }
        self.dataloaders = {k: iter(v) for (k, v) in self.dataloaders.items()}

    @log_phase
    def run(self, data):
        """Run inference query

        Args:
            data (Tensor): Input data
        """
        split = data.get("split", "val")
        data["data"] = next(self.dataloaders[split])
        return data
