import litellm
from transformers import AutoTokenizer

from stages.stage import Stage, log_phase
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query
from utils.schemas.server import ServerModel
from utils.server_manager import ServerManager


def _translate_gen_kwargs(gen_kwargs: dict) -> dict:
    """Map HuggingFace-style gen kwargs (used by existing configs) to the
    OpenAI/litellm names the server endpoints expect. Unknown keys pass through.
    """
    out = dict(gen_kwargs)
    if "max_new_tokens" in out:
        out["max_tokens"] = out.pop("max_new_tokens")
    # HF `do_sample: false` == greedy decoding == temperature 0.
    do_sample = out.pop("do_sample", None)
    if do_sample is False and "temperature" not in out:
        out["temperature"] = 0
    return out


class Inference(Stage):
    """Inference stage backed by a local engine server (vLLM / Ollama).

    The *owner* stage (no ``depends_on_id``) launches the engine as a subprocess
    via :class:`ServerManager` and serves requests over its OpenAI-compatible
    HTTP API using litellm. Dependent stages reuse the same running server.

    Prompts arrive already chat-templated (formatter stages call
    ``get_tokenizer().apply_chat_template``), so we hit the *text-completion*
    endpoint to avoid the engine re-applying a template on top.

    YAML config example::

        config:
          server:
            engine: vllm
            model: Qwen/Qwen3.5-4B
            port: 0
            gpu_memory_utilization: 0.9
          model:
            gen_kwargs:
              max_new_tokens: 256
              do_sample: false
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._max_queries = pipeline_config.loadgen.max_queries

        self._gen_kwargs = _translate_gen_kwargs(
            self.extra_config.get("model", {}).get("gen_kwargs", {})
        )

        # data model for structured generation (parity with HF/MLX stages)
        self._data_model = None
        data_model_path = self.extra_config.get("data_model", None)
        if data_model_path:
            self._data_model = get_component(data_model_path)

        self._depends_on_id = self.extra_config.get("depends_on_id")

        self._server: ServerManager | None = None
        self._litellm_model: str | None = None
        self._litellm_api_base: str | None = None

        if not self._depends_on_id:
            self._server_config = ServerModel(**self.extra_config["server"])
            # Formatter stages dispatch get_tokenizer() during their prepare(),
            # so the tokenizer must exist eagerly. Defaults to the served model
            # id; override with `tokenizer_name` for engines whose model id is
            # not a HF repo (e.g. some Ollama tags).
            tokenizer_name = self.extra_config.get(
                "tokenizer_name", self._server_config.model
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self._server_config = None
            self._tokenizer = None

    def get_tokenizer(self):
        """Getter for the tokenizer, matching the HF/MLX Inference interface."""
        return self._tokenizer

    def get_server_handle(self) -> tuple[str, str]:
        """Return (litellm_model, litellm_api_base) for dependent stages."""
        return self._litellm_model, self._litellm_api_base

    @log_phase
    def prepare(self):
        # Launch + health-check the server BEFORE starting the worker thread
        # (super().prepare()), so dependent stages' pre_run() and the first
        # queries never hit a server that isn't listening yet. Mirrors the
        # MLX "load model before thread start" ordering.
        if not self._depends_on_id:
            print("Launching inference server in ", self.name)
            self._server = ServerManager(self._server_config)
            self._server.start()
            self._litellm_model = self._server.litellm_model
            self._litellm_api_base = self._server.litellm_api_base

        super().prepare()

    def pre_run(self):
        if self._depends_on_id:
            self._litellm_model, self._litellm_api_base = self.dispatch_call(
                self._depends_on_id, "get_server_handle"
            )
            self._tokenizer = self.dispatch_call(self._depends_on_id, "get_tokenizer")

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        # Normalize: formatter stages set query.data to a single string via
        # apply_chat_template, but we iterate over a list of prompts (mirrors
        # stages/llm_huggingface/inference.py).
        if isinstance(batch, str):
            batch = [batch]

        model_out = []
        for prompt in batch:
            # Prompts are already chat-templated upstream, so use the text
            # completion endpoint (NOT chat) to avoid re-applying a template.
            response = litellm.text_completion(
                model=self._litellm_model,
                prompt=prompt,
                api_base=self._litellm_api_base,
                api_key="EMPTY",  # local servers don't authenticate
                **self._gen_kwargs,
            )
            model_out.append(response.choices[0].text)

        if self._data_model:
            model_out = [self._data_model.model_validate_json(x) for x in model_out]

        query.data = model_out

        return {idx: query for idx in self.output_queues}

    def post_run(self):
        # Clean-path teardown for the owner. The real guarantee is the
        # ServerManager registry reaper wired into main.py (SIGTERM + pre
        # os._exit), since post_run() may be skipped on timeout/crash.
        if self._server is not None:
            self._server.stop()
