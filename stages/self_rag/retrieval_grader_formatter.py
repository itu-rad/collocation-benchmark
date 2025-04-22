from stages.stage import Stage, log_phase
from utils.schemas.query import Query


class RetrievalGraderFormatter(Stage):
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

        retrieved_documents = query.data
        query.context["retrieved_documents"] = retrieved_documents
        orig_query = query.context["original_query"]

        chat = [
            {
                "role": "system",
                "content": """
                You are a grader assessing relevance of a retrieved document to a user question. \n 
                If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. Output nothing but the 'yes' or 'no' score. 
                """,
            },
            {
                "role": "user",
                "content": f"Retrieved document: \n\n {retrieved_documents} \n\n User question: {orig_query}",
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
