from itertools import cycle
from torch.utils.data import DataLoader

from stages.stage import Stage, log_phase, log_phase_single


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

        self.batch_size = stage_config.get("batch_size", 1)
        self.num_workers = stage_config.get("num_workers", 0)
        self.preprocessing = stage_config.get("preprocessing", False)

    def my_collate_fn(self, data):
        return data

    @log_phase
    def prepare(self):
        """Build the dataloader"""
        super(TorchFromDataset, self).prepare()
        datasets = self.previous_stages[0].get_datasets()
        if self.preprocessing:
            self.dataloaders = {
                k: DataLoader(
                    v,
                    self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=True,
                    drop_last=True,
                )
                for (k, v) in datasets.items()
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
                for (k, v) in datasets.items()
            }
        # self.dataloaders = {k: iter(v) for (k, v) in self.dataloaders.items()}

    def run(self):
        """Run inference query

        Args:
            data (Tensor): Input data
        """
        while True:
            data_from_queues = self.get_next_from_queues()
            if self.is_done(data_from_queues):
                self.push_to_output(None)
                break

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "start")

            data_from_first = list(data_from_queues.values())[0]
            split = data_from_first.get("split", "val")
            batch_idx = data_from_first.get("batch", 0)
            print("batch", batch_idx)

            # make sure to restart the iterator on every epoch
            # otherwise StopIteration exception is raised
            if batch_idx == 0:
                print("creating new iterator")
                dataloader_iter = iter(self.dataloaders[split])
            data_from_first["data"] = next(dataloader_iter)
            self.push_to_output(data_from_first)

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "end")
