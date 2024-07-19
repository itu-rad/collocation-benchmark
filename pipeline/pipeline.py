from stages.dataset.dataset_registry import DATASET_REGISTRY
from stages.stage_registry import STAGE_REGISTRY


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    dataset = None
    stages = []
    is_training = []
    logger = None  # import logger once done

    def __init__(self, pipeline_config, is_training, logger):
        self.is_training = is_training
        self.logger = logger

        print("Got pipeline config", pipeline_config)

        dataset_config = pipeline_config.get("dataset", None)
        if dataset_config is None:
            raise Exception("Pipeline definition is missing a dataset.")

        self.dataset = DATASET_REGISTRY[dataset_config["module_name"]](
            dataset_config
        ).get_dataset()

        stage_config = pipeline_config.get("stages", None)
        if stage_config is None:
            raise Exception("Pipeline definition is missing pipeline stages.")
        for stage in stage_config:
            if stage.get("ingest_dataset", False):
                stage["config"]["dataset"] = self.dataset
            self.stages.append(
                STAGE_REGISTRY[stage["type"]][stage["module_name"]](stage)
            )

    def prepare(self):
        """Run prepare functions of the stages of the pipelines which contain functionality,
        such as loading/building models."""
        for stage in self.stages:
            stage.prepare()

    def run(self, id):
        """Invoke the pipeline and pass data between stages."""
        data = {
            "id": id,
        }

        for stage in self.stages:
            data = stage.run(data)

        return data
