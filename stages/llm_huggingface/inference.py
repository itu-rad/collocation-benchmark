from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import transformers
from outlines import models, generate

from stages.stage import Stage, log_phase
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


class Inference(Stage):
    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._device = self._parse_device(self.extra_config.get("device", None))
        self._dtype = get_component(self.extra_config.get("dtype", "torch.float32"))

        self._max_queries = pipeline_config.loadgen.max_queries

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.extra_config["model"]["name"]
        )

        # data model for structured generation
        data_model_path = self.extra_config.get("data_model", None)
        if data_model_path:
            self._data_model = get_component(self._data_model)

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
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.extra_config["model"]["name"],
                # attn_implementation="flash_attention_2",
                torch_dtype=(
                    torch.bfloat16
                    if self._device == torch.device("cuda")
                    and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )

        self._model.to(self._device)

        if self._data_model:
            self._model = models.Transformers(self._model, self._tokenizer)
            self._generator = generate.json(self._model, self._data_model)

    @log_phase
    def prepare(self):
        super().prepare()

        self._setup_model()

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        if self._data_model:
            model_out = self._generator(batch)
        else:
            self._tokenizer.pad_token = (
                self._tokenizer.eos_token
            )  # Most LLMs don't have a pad token by default
            model_inputs = self._tokenizer(batch, return_tensors="pt", padding=True).to(
                self._device
            )
            generated_ids = self._model.generate(**model_inputs, max_new_tokens=50)
            model_out = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        print("Model output: ", model_out)
        query.data = model_out

        outputs = {idx: query for idx in self.output_queues}
        return outputs
