from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    default_data_collator,
    get_scheduler,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch

from stages.stage import Stage, log_phase
from utils.component import get_component


class Finetune(Stage):
    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        self._extra_config = stage_config.get("config", {})

        self._device = self._parse_device(self._extra_config["device"])
        self._dtype = self._parse_dtype(self._extra_config["dtype"])
        self._max_queries = pipeline_config["loadgen"]["max_queries"]
        self._gradient_accumulation_steps = self._extra_config[
            "gradient_accumulation_steps"
        ]
        self._running_loss = 0.0
        self._current_step = 0
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._extra_config["model"]["name"]
        )

    def _parse_dtype(self, dtype):
        if dtype == "bf16":
            return torch.bfloat16
        elif dtype == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

    def _parse_device(self, device):
        if device == "cpu":
            return torch.device("cpu")
        elif device == "cuda":
            return torch.device("cuda")
        else:
            raise ValueError(f"Invalid device: {device}")

    def get_device(self):
        return self._device

    def get_tokenizer(self):
        """Getter for the tokenizer

        Returns:
            transformers.PreTrainedTokenizer: The tokenizer
        """
        return self._tokenizer

    def _setup_model(self):
        if self._extra_config["model"]["quantize"]:
            model = AutoModelForCausalLM.from_pretrained(
                self._extra_config["model"]["name"],
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
                self._extra_config["model"]["name"]
            )

        lora_config = LoraConfig(
            r=self._extra_config["model"]["lora_rank"],
            lora_alpha=self._extra_config["model"]["lora_alpha"],
            target_modules=self._extra_config["model"]["lora_attn_modules"],
            lora_dropout=self._extra_config["model"]["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(model, lora_config)
        self._model.to(self._device)
        self._model.train()

    def _setup_optimizer(self):
        config = self._extra_config["optimizer"]
        optimizer_cls = get_component(config.pop("component"))
        self._optimizer = optimizer_cls(params=self._model.parameters(), **config)

    def _setup_lr_scheduler(self):
        config = self._extra_config["lr_scheduler"]
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

    def run(self, inputs):
        data_from_first_queue = list(inputs.values())[0]
        batch = data_from_first_queue["data"]

        batch = {k: v.to(self._device) for k, v in batch.items()}
        outputs = self._model(**batch)
        loss = outputs.loss
        loss.backward()

        self._optimizer.step()
        self._lr_scheduler.step()
        self._optimizer.zero_grad()

        print("Loss", loss.item())

        # TODO: Add support for gradient accumulation

        data_from_first_queue["data"] = loss.item()
        return data_from_first_queue
