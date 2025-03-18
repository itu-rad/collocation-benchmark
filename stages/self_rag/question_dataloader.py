from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from stages.stage import Stage, log_phase
from utils.schemas import Query


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2, tokenizer):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_length = len(dataset1) + len(dataset2)

        self.dataset1 = self.dataset1.rename_column("question", "input")
        self.dataset2 = self.dataset2.rename_column("Query", "input")

        self.tokenizer = tokenizer

        self.dataset1 = self.dataset1.map(self.apply_template)
        self.dataset2 = self.dataset2.map(self.apply_template)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]["input"]
        else:
            return self.dataset2[idx - len(self.dataset1)]["input"]

    def apply_template(self, sample):
        chat = [
            {
                "role": "system",
                "content": """ You are an expert at routing a user question to a sqlite database or web search. 
                                The sqlite database contains accounting records. 
                                Use the sqlite for questions on these topics. Otherwise, use web-search.
                                Structure your response as a json object with the key "search_engine" and the value "sqlite" or "web-search", for example {"search_engine": "sqlite"}.
                                Output nothing but the json object.
                            """,
            },
            {"role": "user", "content": sample["input"]},
        ]
        sample["input"] = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        return sample


class RAGDataLoader(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._batch_size = self.extra_config.get("batch_size", 1)

        self._tokenizer_stage_id = self.extra_config["tokenizer_stage_id"]

        self._dataset = None

        self._dataloader = None
        self._dataloader_iter = None

        self._tokenizer = None

    def get_batch_size(self):
        return self._batch_size

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches for each dataset split
        """
        return {split: len(self._dataset) // self._batch_size for split in ["val"]}

    def _load_datasets(self):
        # Load the datasets here
        # For example, using HuggingFace datasets library

        dataset1 = load_dataset("RUC-NLPIR/FlashRAG_datasets", "web_questions")["test"]
        dataset2 = load_dataset(
            "Exploration-Lab/BookSQL",
            data_files={"val": "BookSQL/val.json"},
        )["val"]

        return dataset1, dataset2

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()
        self._tokenizer = self.dispatch_call(self._tokenizer_stage_id, "get_tokenizer")

        dataset1, dataset2 = self._load_datasets()
        self._dataset = CombinedDataset(dataset1, dataset2, self._tokenizer)

        self._dataloader = DataLoader(
            self._dataset, batch_size=self._batch_size, shuffle=True
        )

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        load the next batch of data and pass it onto the output queues."""

        if query.batch == 0:
            self._dataloader_iter = iter(self._dataloader)
        next_batch = next(self._dataloader_iter)
        print(next_batch)
        query.data = next_batch
        query.context = {"original_query": next_batch}
        output = {idx: query for idx in self.output_queues}
        return output
