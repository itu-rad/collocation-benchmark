from datasets import load_dataset

from stages.stage import Stage, log_phase
from utils.schemas import Query


class VQADataLoader(Stage):
    """Loads image+question pairs from a HuggingFace VQA dataset.

    YAML config example:
        config:
          dataset:
            name: HuggingFaceM4/VQAv2
            split: validation
            image_column: image
            question_column: question
            max_samples: 100
          batch_size: 1
    """

    def __init__(self, stage_config, pipeline_config):
        """Initialize the VQA dataloader stage.

        Args:
            stage_config: Stage configuration from YAML.
            pipeline_config: Pipeline configuration from YAML.
        """
        super().__init__(stage_config, pipeline_config)

        dataset_config = self.extra_config["dataset"]
        self._dataset_name = dataset_config["name"]
        self._dataset_split = dataset_config.get("split", "validation")
        self._image_column = dataset_config.get("image_column", "image")
        self._question_column = dataset_config.get("question_column", "question")
        self._answers_column = dataset_config.get("answers_column", None)
        self._max_samples = dataset_config.get("max_samples", None)
        self._batch_size = self.extra_config.get("batch_size", 1)

        self._dataset = None
        self._data_index = 0

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches per split.
        """
        return {"val": len(self._dataset) // self._batch_size}

    @log_phase
    def prepare(self):
        """Load the VQA dataset from HuggingFace."""
        raw = load_dataset(
            self._dataset_name, split=self._dataset_split
        )

        if self._max_samples and len(raw) > self._max_samples:
            raw = raw.select(range(self._max_samples))

        self._dataset = raw
        print(
            f"VQADataLoader: loaded {len(self._dataset)} samples "
            f"from {self._dataset_name} ({self._dataset_split})"
        )

        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        """Serve the next image+question pair.

        Args:
            query: Incoming query from the load generator.

        Returns:
            dict[int, Query]: Query with data=(PIL_image, question_str).
        """
        if query.batch == 0:
            self._data_index = 0

        sample_idx = self._data_index % len(self._dataset)
        sample = self._dataset[sample_idx]
        self._data_index += 1

        pil_image = sample[self._image_column]
        question = sample[self._question_column]

        query.data = (pil_image, question)
        query.context = {"question": question}

        if self._answers_column is not None:
            raw = sample.get(self._answers_column)
            if raw is not None:
                query.context["golden_answers"] = list(raw)

        output = {out_id: query for out_id in self.output_queues}
        return output
