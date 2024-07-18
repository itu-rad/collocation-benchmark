from stages.stage_registry import STAGE_REGISTRY


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    stages = []
    is_training = []
    logger = None  # import logger once done

    def __init__(self, pipeline_config, is_training, logger):
        self.is_training = is_training
        self.logger = logger

        print("Got pipeline config", pipeline_config)

        # TODO: handle dataset here. It should not be counted as a stage but rather
        # instantiated and pushed into stage_config of a stage asking for dataset
        # by the name of the dataset module (usually the dataloader module).

        stage_config = pipeline_config.get("stages", None)
        if stage_config is None:
            raise Exception("Pipeline definition is missing pipeline stages.")
        for stage in stage_config:
            STAGE_REGISTRY[stage["type"]][stage["module_name"]](stage)

    def prepare(self):
        """This runs prepare functions of the stages of the pipelines which contain functionality,
        such as loading/building models."""
        pass

    def run(self):
        """This is a function that invokes the pipeline"""
        pass
