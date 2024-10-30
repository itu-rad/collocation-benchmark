from torch.utils.data import DataLoader, random_split
from torchtune.utils.collate import padded_collate
from functools import partial

from stages.stage import Stage, log_phase
from utils.schemas import Query, StageModel, PipelineModel
from utils.component import get_component


class TorchTuneDataLoader(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """
        Initializes the DataLoader with the given stage and pipeline configurations.
        Args:
            stage_config (StageModel): Configuration specific to the current stage.
            pipeline_config (PipelineModel): Configuration for the entire pipeline.
        """
        super().__init__(stage_config, pipeline_config)

        dataset_class = get_component(self.extra_config["dataset"]["component"])
        tokenizer_class = get_component(self.extra_config["tokenizer"]["component"])

        self._tokenizer = tokenizer_class(path=self.extra_config["tokenizer"]["path"])
        self._dataset = dataset_class(tokenizer=self._tokenizer)

        self._batch_size = self.extra_config.get("batch_size", 1)

        dataset_splits = random_split(self._dataset, [0.8, 0.2])
        dataset_splits = {"train": dataset_splits[0], "val": dataset_splits[1]}
        self._datasets = {
            split: dataset_splits[split]
            for split in self.extra_config.get("split", ["train"])
        }

        # if _loss_fn_stage_id is not set, the run will be deemed as inference
        self._loss_fn_stage_id = self.extra_config.get("loss_fn_stage_id", None)

        self._dataloaders = {}
        self._dataloader_iter = None

    def get_batch_size(self) -> int:
        """
        Retrieve the batch size for the data loader.

        Returns:
            int: The batch size currently set for the data loader.
        """
        return self._batch_size

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches for each dataset split
        """
        return {k: len(v) // self._batch_size for (k, v) in self._datasets.items()}

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()
        if self._loss_fn_stage_id:
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
                        padding_idx=self._tokenizer.pad_id,
                        ignore_idx=loss_fn.ignore_index,
                    )
                    if self._loss_fn_stage_id
                    else partial(padded_collate, padding_idx=self._tokenizer.pad_id)
                ),
                shuffle=self.extra_config.get("shuffle", True),
            )
            for (k, v) in self._datasets.items()
        }

    def run(self, query: Query) -> dict[int, Query]:
        """
        Executes a query to retrieve the next batch of data from the dataloader.

        Args:
            query (Query): Object containing the query information.

        Returns:
            dict[int, Query]: A dictionary mapping output queue indices to the query object.
        """

        # make sure to restart the iterator on every epoch
        # otherwise StopIteration exception is raised
        if query.batch == 0:
            self._dataloader_iter = iter(self._dataloaders[query.split])
        query.data = next(self._dataloader_iter)
        output = {idx: query for idx in self.output_queues}
        return output
