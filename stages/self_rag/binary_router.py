from stages.stage import Stage, log_phase
from utils.schemas.query import Query


class BinaryRouter(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._yes_stage_id = self.extra_config["yes_stage_id"]
        self._no_stage_id = self.extra_config["no_stage_id"]
        self._end_stage_id = self.extra_config["end_stage_id"]
        self._num_retries = self.extra_config["max_retries"]
        self._retry_is_yes = self.extra_config["retry_is_yes"]
        self._query_dict = {}

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        format the incoming data into a new prompt and pass it onto the output queues.
        """

        yes_idx = query.data[0].lower().find("yes")
        query_id = query.query_id

        if yes_idx != -1:
            if self._retry_is_yes:
                if query_id not in self._query_dict:
                    self._query_dict[query_id] = self._num_retries
                self._query_dict[query_id] -= 1
                if self._query_dict[query_id] < 0:
                    # No more retries left, send to end stage
                    print("No more retries left, sending to end stage")
                    # del self._query_dict[query_id]
                    query.data = "Error: No more retries left"
                    output = {self._end_stage_id: query}
                else:
                    output = {self._yes_stage_id: query}
            else:
                # if query_id in self._query_dict:
                #     del self._query_dict[query_id]
                output = {self._yes_stage_id: query}
        else:
            if self._retry_is_yes:
                # if query_id in self._query_dict:
                #     del self._query_dict[query_id]
                output = {self._no_stage_id: query}
            else:
                if query_id not in self._query_dict:
                    self._query_dict[query_id] = self._num_retries
                self._query_dict[query_id] -= 1
                if self._query_dict[query_id] < 0:
                    # No more retries left, send to end stage
                    print("No more retries left, sending to end stage")
                    # del self._query_dict[query_id]
                    query.data = "Error: No more retries left"
                    output = {self._end_stage_id: query}
                else:
                    output = {self._no_stage_id: query}
        print(f"BinaryRouter: {query_id} -> {output}")
        return output
