# from pydantic import ValidationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import transformers

# from torch.profiler import profile, record_function, ProfilerActivity

import outlines


from threading import Lock

from stages.stage import Stage, log_phase
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


class Inference(Stage):
    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._device = self._parse_device(self.extra_config.get("device", None))
        print(f"Device: {self._device}")
        self._dtype = get_component(self.extra_config.get("dtype", "torch.float32"))

        self._max_queries = pipeline_config.loadgen.max_queries

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.extra_config["model"]["name"]
        )

        # data model for structured generation
        self._data_model = None
        data_model_path = self.extra_config.get("data_model", None)
        if data_model_path:
            self._data_model = get_component(data_model_path)

        self._outlines_generator = None


        self._gen_kwargs = self.extra_config["model"].get("gen_kwargs", {})

        self._model = None

        self._depends_on_id = self.extra_config.get("depends_on_id")
        self._mutex = None
        if not self._depends_on_id:
            self._mutex = Lock()

    def _parse_device(self, device: str | None) -> torch.device:
        # Parse the device string and return the appropriate torch.device.
        # If no device is specified, return the appropriate torch.device based on the available devices.
        #
        # Args:
        #     device (str | None): The device string, or None if no device is specified.
        #
        # Returns:
        #     torch.device: The parsed device.
        if device:
            return torch.device(device)
        else:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

    def get_device(self) -> torch.device:
        """Getter for the device

        Returns:
            torch.device: The device
        """
        return self._device

    def get_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Getter for the tokenizer

        Returns:
            transformers.PreTrainedTokenizer: The tokenizer
        """
        return self._tokenizer

    def get_accelerator(self) -> None:
        """Getter for the accelerator

        Returns:
            Accelerator: The accelerator
        """
        return None

    def get_model_lock(self):
        return self._model, self._mutex

    def _setup_model(self):
        if self.extra_config["model"]["quantize"]:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.extra_config["model"]["name"],
                # attn_implementation="flash_attention_2",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16
                        if self._device == torch.device("cuda")
                        and torch.cuda.is_bf16_supported()
                        else torch.float16
                    ),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.extra_config["model"]["name"],
                # attn_implementation="flash_attention_2",
                # torch_dtype=(
                #     torch.bfloat16
                #     if self._device == torch.device("cuda")
                #     and torch.cuda.is_bf16_supported()
                #     else torch.float16
                # ),
                torch_dtype="auto",
                device_map="auto",
            )

    def _setup_outlines(self) -> None:
        if self._data_model and not self._outlines_generator:
            # outlines.models.Transformers expects (model, tokenizer)
            self._outlines_model = outlines.models.Transformers(
                self._model, self._tokenizer
            )
            self._outlines_generator = outlines.Generator(
                self._outlines_model, self._data_model
            )

    @log_phase
    def prepare(self):
        super().prepare()

        if not self._depends_on_id:
            print("Setting up model in ", self.name)
            self._setup_model()
            self._setup_outlines()


    def pre_run(self):

        if self._depends_on_id:
            self._model, self._mutex = self.dispatch_call(
                self._depends_on_id, "get_model_lock"
            )
            self._setup_outlines()

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        print("Input data:", batch)

        if self._outlines_generator:
            if self._mutex:
                self._mutex.acquire()
            try:
                # outlines generator returns the completion string directly
                model_out = self._outlines_generator(batch, **self._gen_kwargs)
            finally:
                if self._mutex:
                    self._mutex.release()

        else:
            self._tokenizer.pad_token = (
                self._tokenizer.eos_token
            )  # Most LLMs don't have a pad token by default
            model_inputs = self._tokenizer(batch, return_tensors="pt", padding=True).to(
                self._model.device
            )

            # # Acquire the lock if the model is shared
            if self._mutex:
                self._mutex.acquire()

            generated_ids = self._model.generate(
                **model_inputs,
                **self._gen_kwargs,
                # logits_processor=self._logits_processors, # No longer used here
            )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            if self._mutex:
                self._mutex.release()

            model_out = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        print("Raw model output: ", model_out)


        if self._data_model:
            # the validation might fail, however, regeneration does not seem to help
            # try:
            #     model_out = [self._data_model.model_validate_json(x) for x in model_out]
            # except ValidationError:
            #     return self.run(query)
            model_out = [self._data_model.model_validate_json(x) for x in model_out]

        # print("Model output: ", model_out)
        query.data = model_out

        outputs = {idx: query for idx in self.output_queues}
        return outputs
