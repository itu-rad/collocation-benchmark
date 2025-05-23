from stages.stage import Stage, log_phase
from utils.schemas.query import Query


class AnswerGeneratorFormatter(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._tokenizer_stage_id = self.extra_config["tokenizer_stage_id"]

        self._tokenizer = None

    @log_phase
    def prepare(self):
        """Load the tokenizer from another stage"""
        super().prepare()
        self._tokenizer = self.dispatch_call(self._tokenizer_stage_id, "get_tokenizer")

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        format the incoming data into a new prompt and pass it onto the output queues.
        """

        orig_query = query.context["original_query"]
        retrieved_documents = query.context["retrieved_documents"]

        chat = [
            {
                "role": "system",
                "content": """
                You are an answer rewriter. Given the retrieved data from a sqlite database and the original query, output an answer to the query supported by the retrieved data. \n 
                Give a concise answer.
                """,
            },
            {
                "role": "user",
                "content": f"User question: \n\n {orig_query} \n\n Retrieved documents: {retrieved_documents}",
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
