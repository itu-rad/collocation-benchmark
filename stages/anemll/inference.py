import sys
import os
from threading import Lock
from pathlib import Path
from transformers import AutoTokenizer
import torch
import mlflow
import threading


from stages.stage import Stage, log_phase
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query

# We need to add the evaluate directory to sys.path so ANE_Model can find its dependencies
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
EVALUATE_DIR = REPO_ROOT / "stages" / "anemll" / "Anemll" / "evaluate"

if str(EVALUATE_DIR) not in sys.path:
    # print(f"Adding {EVALUATE_DIR} to sys.path")
    sys.path.append(str(EVALUATE_DIR))

# Now we can import ANE_Model
try:
    from ane.ane_model import ANE_Model
except ImportError:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ane_model", EVALUATE_DIR / "ane" / "ane_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ane_model"] = module
    spec.loader.exec_module(module)
    ANE_Model = module.ANE_Model


class Inference(Stage):
    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._max_queries = pipeline_config.loadgen.max_queries

        self._model_path = self.extra_config["model"]["name"]

        # Extra gen args
        self._max_new_tokens = self.extra_config["model"].get("max_new_tokens", 100)
        self._gen_kwargs = self.extra_config["model"].get("gen_kwargs", {})

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
        model_path = Path(self._model_path)
        if not model_path.exists():
            model_path = REPO_ROOT / self._model_path

        print(f"Loading Anemll model from {model_path}")
        self._model = ANE_Model(model_path)

        print(f"Loading Tokenizer from {model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        if hasattr(self._tokenizer, "pad_token") and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @log_phase
    def prepare(self):
        super().prepare()

        if not self._depends_on_id:
            print("Setting up model in ", self.name)
            self._setup_model()

    def pre_run(self):
        if self._depends_on_id:
            self._model, self._mutex = self.dispatch_call(
                self._depends_on_id, "get_model_lock"
            )

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        # Pre-tokenize batch
        # ANE_Model processes one sequence at a time (stateful).
        # We need to loop over the batch.

        model_out = []

        if self._mutex:
            self._mutex.acquire()

        for prompt in batch:
            # Tokenize
            with mlflow.start_span(
                name="tokenizer.encode", attributes={"thread_id": threading.get_ident()}
            ):
                inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(
                    torch.int32
                )
                input_ids = inputs
            print(f"\nPrompt: '{prompt}'")

            # Generate
            # ANE_Model.generate returns LIST of token IDs
            with mlflow.start_span(
                name="model.generate", attributes={"thread_id": threading.get_ident()}
            ):
                generated_ids = self._model.generate(
                    input_ids, max_new_tokens=self._max_new_tokens, **self._gen_kwargs
                )

            # Decode
            with mlflow.start_span(
                name="tokenizer.decode", attributes={"thread_id": threading.get_ident()}
            ):
                text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            print("generated: ", text)
            model_out.append(text)

        if self._mutex:
            self._mutex.release()

        if self._data_model:
            try:
                model_out = [self._data_model.model_validate_json(x) for x in model_out]
            except Exception:
                pass  # validation might fail

        query.data = model_out

        outputs = {idx: query for idx in self.output_queues}
        return outputs
