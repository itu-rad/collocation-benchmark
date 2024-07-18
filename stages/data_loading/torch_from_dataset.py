from torch.utils.data import DataLoader

from stages.stage import Stage


class TorchFromDataset(Stage):
    dataset = None
    dataloader = None
    batch_size = 1
    num_workers = 0

    def __init__(self, stage_config):
        """nitialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config)

        self.dataset = stage_config.get("dataset", None)
        if self.dataset is None:
            raise Exception(
                "Dataset is missing. Dataset is required to build a dataloader."
            )

        self.batch_size = stage_config.get("batch_size", 1)
        self.num_workers = stage_config.get("num_workers", 0)

    def prepare(self):
        """Build the dataloader"""
        self.dataloader = DataLoader(
            self.dataset, self.batch_size, num_workers=self.num_workers, drop_last=True
        )
        self.dataloader = iter(self.dataloader)

    def run(self, data):
        """Run inference query

        Args:
            data (Tensor): Input data
        """
        return next(self.dataloader)
