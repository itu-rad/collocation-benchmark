from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
)

from stages.stage import Stage, log_phase
from utils.schemas import Query


class HuggingFaceDataLoader(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._batch_size = self.extra_config.get("batch_size", 1)
        self._split = self.extra_config.get("split", ["train"])
        self._shuffle = self.extra_config.get("shuffle", True)
        self._dataset_config = self.extra_config.get("dataset", {})
        self._tokenizer_stage_id = self.extra_config.get("tokenizer_stage_id", 0)
        dataset_name = self._dataset_config["name"]
        self._dataset = load_dataset(dataset_name)
        self._dataset_length = {k: len(v) for (k, v) in self._dataset.items()}
        self._dataset = {
            k: v.to_iterable_dataset(num_shards=self._dataset_length[k])
            for (k, v) in self._dataset.items()
        }
        self._dataset = self._dataset["train"]
        if self._shuffle:
            self._dataset = self._dataset.shuffle()
        # print("Shards", self._dataset.n_shards)
        self._system_column_name = self._dataset_config.get("system_column_name")
        self._user_column_name = self._dataset_config.get("user_column_name")
        self._assistant_column_name = self._dataset_config.get("assistant_column_name")

    def get_batch_size(self):
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
        self._tokenizer = self._dispatch_call(self._tokenizer_stage_id, "get_tokenizer")
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
        self._device = self._dispatch_call(self._tokenizer_stage_id, "get_device")
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
