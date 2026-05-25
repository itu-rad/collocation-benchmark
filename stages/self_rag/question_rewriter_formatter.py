from stages.stage import Stage, log_phase
from utils.chat import apply_chat_template_safe
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
                "content": (
                    "You are a question re-writer that converts an input question into a better version "
                    "optimised for document retrieval. Output ONLY the rewritten question on a single line. "
                    "Do not add quotation marks, preamble, explanation, lists, or any other text."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {orig_query}\n\nRewritten question:",
            },
        ]

        query.data = apply_chat_template_safe(
            self._tokenizer,
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        query.context["is_retry"] = True

        print("Final query", query)

        output = {idx: query for idx in self.output_queues}
        return output
