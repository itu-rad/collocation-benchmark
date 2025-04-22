from stages.stage import Stage, log_phase
from utils.schemas.query import Query


class QuestionRewriterFormatter(Stage):
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

        print("QuestionRewriterFormatter: ", query)
        orig_query = query.context["original_query"]

        chat = [
            {
                "role": "system",
                "content": """
                You a question re-writer that converts an input question to a better version that is optimized \n 
                for sqlite retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
                """,
            },
            {
                "role": "user",
                "content": f"Here is the initial question: \n\n {orig_query} \n Formulate an improved question.",
            },
        ]

        query.data = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        query.context["is_retry"] = True

        print("Final query", query)

        output = {idx: query for idx in self.output_queues}
        return output
