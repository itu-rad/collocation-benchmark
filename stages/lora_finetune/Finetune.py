from stages.stage import Stage


class Finetune(Stage):
    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        print("Intializing finetune stage")
