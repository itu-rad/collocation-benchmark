from torch.utils.data import DataLoader
from torchtune.utils.collate import padded_collate
from functools import partial

from stages.stage import Stage, log_phase, log_phase_single


class TorchTuneDataLoader(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        extra_config = stage_config.get("config", {})

        self._batch_size = extra_config.get("batch_size", 1)
        self._split = extra_config.get("split", "train")
        self._shuffle = extra_config.get("shuffle", True)

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()
        # data loader has a single preceeding stage, which is the Torch dataset itself
        datasets = self.previous_stages[0].get_datasets()
        tokenizer = self.previous_stages[0].get_tokenizer()
        # TODO: define get loss_fn function in the finetune stage
        loss_fn = self.next_stages[0].get_loss_fn()

        self._dataloader = DataLoader(
            dataset=datasets[self._split],
            batch_size=self._batch_size,
            collate_fn=partial(
                padded_collate,
                padding_idx=tokenizer.pad_id,
                ignore_idx=loss_fn.ignore_index,
            ),
            shuffle=self._shuffle,
        )

    def run(self):
        """Poll for incoming data in the queues,
        load the next batch of data and pass it onto the output queues."""
        while True:
            data_from_queues = self.get_next_from_queues()
            if self.is_done(data_from_queues):
                self.push_to_output(None)
                break

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "start")

            data_from_first = list(data_from_queues.values())[0]
            batch_idx = data_from_first.get("batch", 0)

            # make sure to restart the iterator on every epoch
            # otherwise StopIteration exception is raised
            if batch_idx == 0:
                dataloader_iter = iter(self._dataloader)
            data_from_first["data"] = next(dataloader_iter)
            self.push_to_output(data_from_first)

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "end")
