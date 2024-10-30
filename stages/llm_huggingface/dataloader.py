from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
)

from stages.stage import Stage, log_phase
from utils.schemas import Query, StageModel, PipelineModel


class HuggingFaceDataLoader(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """
        Initializes the DataLoader with the given stage and pipeline configurations.
        Args:
            stage_config (StageModel): Configuration specific to the current stage.
            pipeline_config (PipelineModel): Configuration for the entire pipeline.
        """

        super().__init__(stage_config, pipeline_config)

        self._batch_size = self.extra_config.get("batch_size", 1)
        self._split = self.extra_config.get("split", ["train"])

        dataset_config = self.extra_config.get("dataset", {})
        dataset_name = dataset_config["name"]
        self._dataset = load_dataset(dataset_name)
        self._dataset_length = {k: len(v) for (k, v) in self._dataset.items()}
        self._dataset = {
            k: v.to_iterable_dataset(num_shards=self._dataset_length[k])
            for (k, v) in self._dataset.items()
        }
        self._dataset = {k: v for (k, v) in self._dataset.items() if k in self._split}

        # shuffle, if necessary
        if self.extra_config.get("shuffle", True):
            self._dataset = {k: v.shuffle() for (k, v) in self._dataset.items()}

        self._system_column_name = dataset_config.get("system_column_name")
        self._user_column_name = dataset_config.get("user_column_name")
        self._assistant_column_name = dataset_config.get("assistant_column_name")

        self._tokenizer = None
        self._dataloader = None
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
        return {
            split: self._dataset_length[split] // self._batch_size
            for split in self._split
        }

    def _concat(self, sample):
        if sample[self._user_column_name] != "":
            sample["concated_text"] = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample[self._system_column_name]}\n\n### Input:\n{sample[self._user_column_name]}\n\n"
                f"### Response:\n{sample[self._assistant_column_name]}\n\n"
            )
        else:
            sample["concated_text"] = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample[self._system_column_name]}\n\n"
                f"### Response:\n{sample[self._assistant_column_name]}\n\n"
            )
        return sample

    def _tokenize(self, samples):
        inputs = self._tokenizer(
            samples["concated_text"],
            padding=True,
            truncation=True,
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()

        self._tokenizer = self.dispatch_call(
            self.extra_config.get("tokenizer_stage_id", 0), "get_tokenizer"
        )
        self._dataset = self._dataset.map(
            self._concat,
        )
        self._dataset = self._dataset.map(
            self._tokenize,
            batched=True,
        )
        self._dataset = self._dataset.remove_columns(
            [
                self._system_column_name,
                self._user_column_name,
                self._assistant_column_name,
                "concated_text",
            ]
        )
        self._dataset = self._dataset.with_format("torch")
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self._tokenizer, mlm=False
            ),
        )
        self._dataloader_iter = iter(self._dataloader)

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        load the next batch of data and pass it onto the output queues."""

        next_batch = next(self._dataloader_iter)
        query.data = next_batch
        output = {idx: query for idx in self.output_queues}
        return output
