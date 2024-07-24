import logging

from stages.dataset.dataset_registry import DATASET_REGISTRY
from stages.stage_registry import STAGE_REGISTRY


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    name = ""
    datasets = None
    stages = []
    is_training = []

    def __init__(self, pipeline_config):

        print("Got pipeline config", pipeline_config)

        self.name = pipeline_config.get("name", "Unknown pipeline name")

        dataset_config = pipeline_config.get("dataset", None)
        if dataset_config is None:
            raise Exception("Pipeline definition is missing a dataset.")

        self.datasets = DATASET_REGISTRY[dataset_config["module_name"]](
            dataset_config, self.name
        ).get_dataset()

        stage_config = pipeline_config.get("stages", None)

        if stage_config is None:
            raise Exception("Pipeline definition is missing pipeline stages.")
        for stage in stage_config:
            if stage.get("ingest_dataset", False):
                stage["config"]["dataset"] = self.datasets
            self.stages.append(
                STAGE_REGISTRY[stage["type"]][stage["module_name"]](stage, self.name)
            )

    def get_dataset_length(self):
        return {k: len(v) for (k, v) in self.datasets.items()}

    def prepare(self):
        """Run prepare functions of the stages of the pipelines which contain functionality,
        such as loading/building models."""
        logging.info("%s, pipeline, prepare, start", self.name)
        for stage in self.stages:
            stage.prepare()

        logging.info("%s, pipeline, prepare, end", self.name)

    def run(self, sample_queue, event):
        """Invoke the pipeline and pass data between stages."""

        while True:
            data = sample_queue.get()
            if data is None:
                break
            split = data.get("split", "val")
            submitted = data.get("query_submitted", 0)
            logging.info(
                "%s, pipeline - %s, run, start, %.6f", self.name, split, submitted
            )
            for stage in self.stages:
                data = stage.run(data)

            logging.info("%s, pipeline - %s, run, end", self.name, split)
            event.set()
