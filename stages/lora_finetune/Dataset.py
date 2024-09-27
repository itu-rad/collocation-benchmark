from utils.component import get_component
from torch.utils.data import random_split

from stages.stage import Stage, log_phase, log_phase_single


class Dataset(Stage):

    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Configuration, which includes the name of the dataset and split

        Raises:
            Exception: Missing model name.
            Exception: Model checkpoint path is required for inference.
        """
        super().__init__(stage_config, pipeline_config)

        extra_config = stage_config.get("config", {})

        dataset_class = get_component(extra_config["dataset"]["component"])
        tokenizer_class = get_component(extra_config["tokenizer"]["component"])

        self._tokenizer = tokenizer_class(path=extra_config["tokenizer"]["path"])
        self._dataset = dataset_class(tokenizer=self._tokenizer)

        splits = random_split(self._dataset, [0.8, 0.2])
        self._datasets = {"train": splits[0], "val": splits[1]}

        self.batch_size = 16

    def get_datasets(self):
        """Getter for the datasets

        Returns:
            dict[str, torchvision.datasets.VisionDataset]: dictionary of datasets (train and/or val)
        """
        return self._datasets

    def get_num_batches(self):
        """Calculate the number of batches for each dataset

        Returns:
            dict[str, int]: dictionary with number of batches for each dataset
        """
        return {k: len(v) // self.batch_size for (k, v) in self._datasets.items()}

    def get_tokenizer(self):
        """Getter for the tokenizer

        Returns:
            transformers.PreTrainedTokenizer: The tokenizer
        """
        return self._tokenizer

    @log_phase
    def prepare(self):
        """Prepare stage for execution"""
        super().prepare()

    def run(self):
        """Pass the input onto the output (no action to be performed on the data)"""
        while True:
            inputs = self.get_next_from_queues()
            if self.is_done(inputs):
                self.push_to_output(list(inputs.values())[0])
                break

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "start")
            self.push_to_output(list(inputs.values())[0])
            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "end")
