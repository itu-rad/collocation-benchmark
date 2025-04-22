from stages.stage import Stage, log_phase
from utils.schemas.query import Query


class IndexRouter(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._sqlite_stage_id = self.extra_config["sqlite_stage_id"]
        self._llm_stage_id = self.extra_config["llm_stage_id"]

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        format the incoming data into a new prompt and pass it onto the output queues.
        """

        if type(query.data[0]) is not str:
            query.data = [x.dump_model_json() for x in query.data] 

        sqllite_idx = query.data[0].lower().find("sqlite")

        if sqllite_idx != -1:
            output = {self._sqlite_stage_id: query}
        else:
            output = {self._llm_stage_id: query}

        return output
