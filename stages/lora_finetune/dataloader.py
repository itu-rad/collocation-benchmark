from torch.utils.data import DataLoader
from torchtune.utils.collate import padded_collate
from functools import partial

from stages.stage import Stage, log_phase


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
        self._tokenizer_stage_id = extra_config.get("tokenizer_stage_id", 0)
        self._datasets_stage_id = extra_config.get("dataset_stage_id", 0)
        self._loss_fn_stage_id = extra_config.get("loss_fn_stage_id", 2)
        self._is_training = pipeline_config["loadgen"]["is_training"]

    def get_batch_size(self):
        return self._batch_size

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()
        # data loader has a single preceeding stage, which is the Torch dataset itself
        datasets = self.dispatch_call(self._datasets_stage_id, "get_datasets")
        tokenizer = self.dispatch_call(self._tokenizer_stage_id, "get_tokenizer")
        if self._is_training:
            loss_fn = self.dispatch_call(
                self._loss_fn_stage_id,
                "get_loss_fn",
            )

        self._dataloaders = {
            k: DataLoader(
                dataset=v,
                batch_size=self._batch_size,
                collate_fn=(
                    partial(
                        padded_collate,
                        padding_idx=tokenizer.pad_id,
                        ignore_idx=loss_fn.ignore_index,
                    )
                    if self._is_training
                    else partial(padded_collate, padding_idx=tokenizer.pad_id)
                ),
                shuffle=self._shuffle,
            )
            for (k, v) in datasets.items()
        }

        # self._dataloader_iter = iter(self._dataloader)

    def run(self, inputs):
        """Poll for incoming data in the queues,
        load the next batch of data and pass it onto the output queues."""
        data_from_first_queue = list(inputs.values())[0]
        batch_idx = data_from_first_queue.get("batch", 0)
        split = data_from_first_queue.get("split", "val")

        # make sure to restart the iterator on every epoch
        # otherwise StopIteration exception is raised
        if batch_idx == 0:
            self._dataloader_iter = iter(self._dataloaders[split])
        data_from_first_queue["data"] = next(self._dataloader_iter)
        return data_from_first_queue
