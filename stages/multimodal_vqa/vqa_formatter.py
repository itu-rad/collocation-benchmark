from stages.stage import Stage, log_phase
from utils.chat import apply_chat_template_safe
from utils.schemas import Query


class VQAPromptFormatter(Stage):
    """Formats retrieved documents + question into an LLM-ready prompt.

    YAML config example:
        config:
          tokenizer_stage_id: 4
          system_prompt: "You are a helpful assistant."
    """

    def __init__(self, stage_config, pipeline_config):
        """Initialize the VQA prompt formatter.

        Args:
            stage_config: Stage configuration from YAML.
            pipeline_config: Pipeline configuration from YAML.
        """
        super().__init__(stage_config, pipeline_config)

        self._tokenizer_stage_id = self.extra_config[
            "tokenizer_stage_id"
        ]
        self._system_prompt = self.extra_config.get(
            "system_prompt",
            "You are a helpful assistant. Use the provided context "
            "to answer the question concisely."
        )
        self._tokenizer = None

    @log_phase
    def prepare(self):
        """Load the tokenizer from the LLM inference stage."""
        super().prepare()
        self._tokenizer = self.dispatch_call(
            self._tokenizer_stage_id, "get_tokenizer"
        )

    def run(self, query: Query) -> dict[int, Query]:
        """Format retrieved docs + question into a chat prompt.

        Args:
            query: Query with data=[doc_str_1, ...] and
                context={"question": str}.

        Returns:
            dict[int, Query]: Query with data=formatted_prompt_string.
        """
        retrieved_docs = query.data
        question = query.context["question"]

        context_block = "Context:\n" + "\n\n".join(retrieved_docs)

        chat = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"{context_block}\n\nQuestion: {question}",
            },
        ]

        query.data = apply_chat_template_safe(
            self._tokenizer,
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = {idx: query for idx in self.output_queues}
        return output
