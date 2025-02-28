from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from accelerate import Accelerator

from stages.stage import Stage, log_phase
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


class Finetune(Stage):
    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._device = self._parse_device(self.extra_config["device"])
        self._dtype = get_component(self.extra_config.get("dtype", "torch.float32"))

        self._max_queries = pipeline_config.loadgen.max_queries
        self._gradient_accumulation_steps = self.extra_config[
            "gradient_accumulation_steps"
        ]

        self._running_loss = 0.0
        self._current_step = 0

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.extra_config["model"]["name"]
        )

        self._accelerator = None
        if self.extra_config["accelerator"]["is_enabled"]:
            self._accelerator = Accelerator()
            self._device = self._accelerator.device

        self._model = None
        self._optimizer = None
        self._lr_scheduler = None

    def _parse_device(self, device: str | None) -> torch.device:
        """
        Parse the device string and return the appropriate torch.device.
        If no device is specified, return the appropriate torch.device based on the available devices.

        Args:
            device (str | None): The device string, or None if no device is specified.

        Returns:
            torch.device: The parsed device.
        """
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

    def get_accelerator(self) -> Accelerator:
        """Getter for the accelerator

        Returns:
            Accelerator: The accelerator
        """
        return self._accelerator

    def _setup_model(self):
        if self.extra_config["model"]["quantize"]:
            model = AutoModelForCausalLM.from_pretrained(
                self.extra_config["model"]["name"],
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                        else torch.float16
                    ),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            )
            # setup for quantized training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.extra_config["model"]["name"]
            )

        lora_config = LoraConfig(
            r=self.extra_config["model"]["lora_rank"],
            lora_alpha=self.extra_config["model"]["lora_alpha"],
            target_modules=self.extra_config["model"]["lora_attn_modules"],
            lora_dropout=self.extra_config["model"]["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(model, lora_config)
        if not self._accelerator:
            self._model.to(self._device)
        self._model.train()

    def _setup_optimizer(self):
        config = self.extra_config["optimizer"]
        optimizer_cls = get_component(config.pop("component"))
        self._optimizer = optimizer_cls(params=self._model.parameters(), **config)

    def _setup_lr_scheduler(self):
        config = self.extra_config["lr_scheduler"]
        name = config.pop("name")
        self._lr_scheduler = get_scheduler(
            name=name,
            optimizer=self._optimizer,
            num_training_steps=self._max_queries,
            **config,
        )

    @log_phase
    def prepare(self):
        super().prepare()

        self._setup_model()
        self._setup_optimizer()
        self._setup_lr_scheduler()

        # enable distributed training
        if self._accelerator:
            self._model, self._optimizer, self._lr_scheduler = (
                self._accelerator.prepare(
                    self._model, self._optimizer, self._lr_scheduler
                )
            )

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        if not self._accelerator:
            batch = {k: v.to(self._device) for k, v in batch.items()}
        outputs = self._model(**batch)
        loss = outputs.loss
        loss = loss / self._gradient_accumulation_steps
        self._running_loss += loss
        loss.backward()

        self._optimizer.step()
        self._lr_scheduler.step()
        self._optimizer.zero_grad()

        if (self._current_step + 1) % self._gradient_accumulation_steps == 0:
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            self._lr_scheduler.step()

            print(
                f"{self._current_step}/{self._max_queries} | Running loss: {self._running_loss.item()}"
            )
            # Reset running stats for the next step
            self._running_loss = 0

        self._current_step += 1

        query.data = loss.item()
        outputs = {idx: query for idx in self.output_queues}
        return outputs
