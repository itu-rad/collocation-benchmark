from stages.stage import Stage, log_phase
from utils.schemas.query import Query


class SQLQueryGeneratorFormatter(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._tokenizer_stage_id = self.extra_config["tokenizer_stage_id"]

        self._tokenizer = None

        self._schema_stage_id = self.extra_config["schema_stage_id"]

        self._schema = ""

    @log_phase
    def prepare(self):
        """Load the tokenizer and db schema from another stage"""
        super().prepare()
        self._tokenizer = self.dispatch_call(self._tokenizer_stage_id, "get_tokenizer")
        self._schema = self.dispatch_call(self._schema_stage_id, "get_schema")

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        format the incoming data into a new prompt and pass it onto the output queues.
        """

        if query.context.get("is_retry", False):
            # this is a loopback, so we need to update the original query
            query.context["is_retry"] = False
            query.context["original_query"] = query.data

        orig_queries = query.context["original_query"]

        chat = [
            {
                "role": "system",
                "content": """
                You are an expert SQL query writer. Given the sqlite database and user-provided natural language query, translate the natural language query into a sql query.
                Output nothing but the sql query.
                """,
            },
            {
                "role": "user",
                "content": f"SQLite schema: \n\n {self._schema} \n\n User question: {orig_queries[0]}\n",
            },
        ]

        query.data = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        print("Final query", query)

        output = {idx: query for idx in self.output_queues}
        return output
