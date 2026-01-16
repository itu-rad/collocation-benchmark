from threading import Lock
import mlx.core as mx
from mlx_lm import load, generate

from stages.stage import Stage, log_phase
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
        self._tokenizer = None

        self._depends_on_id = self.extra_config.get("depends_on_id")
        self._mutex = None
        if not self._depends_on_id:
            self._mutex = Lock()

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
        super().prepare()

        if not self._depends_on_id:
            print("Setting up model in ", self.name)
            self._setup_model()

    def pre_run(self):
        if self._depends_on_id:
            # If we were sharing models across stages (e.g. prefill/decode split), we'd grab it here.
            # MLX sharing might be different, but keeping structure consistent.
            self._model, self._mutex = self.dispatch_call(
                self._depends_on_id, "get_model_lock"
            )

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        if self._mutex:
            self._mutex.acquire()

        try:
            # MLX generate typically handles one prompt string.
            # We iterate over the batch.
            # Note: mlx_lm.generate prints to stdout by default, verbose=False suppresses it.
            model_out = []
            for prompt in batch:
                response = generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    verbose=False,
                    **self._gen_kwargs,
                )
                model_out.append(response)

                print("generated: ", response)

        finally:
            if self._mutex:
                self._mutex.release()

        if self._data_model:
            # Attempt validation if a data model is provided
            model_out = [self._data_model.model_validate_json(x) for x in model_out]

        query.data = model_out

        outputs = {idx: query for idx in self.output_queues}
        return outputs
