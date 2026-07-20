from threading import Lock
import mlx.core as mx
from mlx_lm import load, stream_generate
from transformers import AutoTokenizer

from stages.stage import Stage, log_phase, log_first_token, log_generated_tokens
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


class Inference(Stage):
    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        # MLX mostly manages devices automatically (Apple Silicon), but we can allow some config if needed via extra_config
        # For now, we rely on MLX defaults (Metal/GPU if available).

        self._max_queries = pipeline_config.loadgen.max_queries

        self._model_path = self.extra_config["model"]["name"]
        self._tokenizer_config = self.extra_config["model"].get("tokenizer_config", {})
        self._gen_kwargs = self.extra_config["model"].get("gen_kwargs", {})

        # data model for structured generation (placeholder - mlx_lm doesn't support outlines directly yet)
        self._data_model = None
        data_model_path = self.extra_config.get("data_model", None)
        if data_model_path:
            self._data_model = get_component(data_model_path)

        self._model = None

        # Load tokenizer eagerly in __init__ (matching HuggingFace Inference pattern)
        # so that get_tokenizer() is available immediately for other stages'
        # prepare() calls. The full model is loaded later in prepare().
        self._depends_on_id = self.extra_config.get("depends_on_id")
        if not self._depends_on_id:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        else:
            self._tokenizer = None

        self._mutex = None
        if not self._depends_on_id:
            self._mutex = Lock()

    def get_tokenizer(self):
        """Getter for the tokenizer, matching the HuggingFace Inference interface.

        Returns:
            The tokenizer loaded by mlx_lm.
        """
        return self._tokenizer

    def get_model_lock(self):
        return self._model, self._mutex

    def _setup_model(self):
        print(f"Loading MLX model from {self._model_path}")
        # trust_remote_code=True might be needed for some models, can make it configurable
        self._model, self._tokenizer = load(
            self._model_path, tokenizer_config=self._tokenizer_config
        )

    @log_phase
    def prepare(self):
        # Load the model BEFORE starting the worker thread (super().prepare()),
        # because other stages' prepare() methods may dispatch_call
        # get_tokenizer() which requires the tokenizer to be ready.
        if not self._depends_on_id:
            print("Setting up model in ", self.name)
            self._setup_model()

        super().prepare()

    def pre_run(self):
        if self._depends_on_id:
            self._model, self._mutex = self.dispatch_call(
                self._depends_on_id, "get_model_lock"
            )
            self._tokenizer = self.dispatch_call(self._depends_on_id, "get_tokenizer")

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        # Normalize: formatter stages set query.data to a single string
        # via apply_chat_template, but we need a list to iterate over.
        if isinstance(batch, str):
            batch = [batch]

        if self._mutex:
            self._mutex.acquire()

        try:
            # MLX handles one prompt string at a time; we iterate the batch.
            # mlx_lm.generate(verbose=False) is a thin wrapper over
            # stream_generate — it only concatenates response.text over the
            # stream (verified in mlx_lm 0.31.3 generate.py:756) — so
            # consuming stream_generate directly with the same kwargs is
            # text- and token-identical while exposing the first-token
            # instant for TTFT. The "first_token" pair is emitted once per
            # run() call (first prompt only) so sub-phase rows stay 1:1 with
            # the stage's run start/end pairs (staged_lib pairs by index).
            model_out = []
            n_generated = 0
            log_first_token(self, "start")
            for i, prompt in enumerate(batch):
                awaiting_first = i == 0
                text = ""
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    **self._gen_kwargs,
                ):
                    if awaiting_first:
                        log_first_token(self, "end")
                        awaiting_first = False
                    text += response.text
                # generation_tokens of the last yielded response = number of
                # decode steps for this prompt (mlx_lm counts EOS/length stop)
                n_generated += response.generation_tokens
                model_out.append(text)

            log_generated_tokens(self, n_generated)

        finally:
            if self._mutex:
                self._mutex.release()

        if self._data_model:
            # Attempt validation if a data model is provided
            model_out = [self._data_model.model_validate_json(x) for x in model_out]

        query.data = model_out

        outputs = {idx: query for idx in self.output_queues}
        return outputs
