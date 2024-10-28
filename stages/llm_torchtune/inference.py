from torchtune.utils._checkpointing._checkpointer import FullModelHFCheckpointer
from torchtune.utils.precision import set_default_dtype
from torchtune.utils.constants import MODEL_KEY, ADAPTER_KEY, OPT_KEY
from torchtune.utils._generation import (
    generate_next_token,
    update_stop_tokens_tracker,
)
import torch

from stages.stage import Stage, log_phase, log_phase_single
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


class Inference(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._checkpoint_dict = self._load_checkpoint(self.extra_config["checkpointer"])

        self._device = self._parse_device(self.extra_config["device"])
        self._dtype = get_component(self.extra_config.get("dtype", "torch.float32"))

        self._max_queries = pipeline_config["loadgen"]["max_queries"]

        self._temperature = self.extra_config.get("temperature", 0.6)
        self._top_k = self.extra_config.get("top_k", 300)

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

    def _load_checkpoint(self, checkpoint_conf):
        """Load a checkpoint from the checkpoint config.

        Args:
            checkpoint_conf (dict): The checkpoint configuration.

        Returns:
            dict: The loaded checkpoint.
        """
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=checkpoint_conf["checkpoint_dir"],
            checkpoint_files=checkpoint_conf["checkpoint_files"],
            model_type=checkpoint_conf["model_type"],
            output_dir=checkpoint_conf["output_dir"],
        )

        checkpoint_dict = checkpointer.load_checkpoint()
        return checkpoint_dict

    def _setup_model(self):
        # Configure device and precision
        with set_default_dtype(self._dtype), self._device:
            self._model = get_component(self.extra_config["model"]["component"])()

        # Load the base model checkpoint
        base_model_state_dict = self._checkpoint_dict[MODEL_KEY]
        self._model.load_state_dict(base_model_state_dict, strict=False)

        batch_size = self.dispatch_call(
            self.extra_config.get("batch_size_stage_id", 1), "get_batch_size"
        )
        with self._device:
            self._model.setup_caches(batch_size=batch_size, dtype=self._dtype)

        self._model.eval()

    @log_phase
    def prepare(self):
        super().prepare()

        self._setup_model()
        self._tokenizer = self.dispatch_call(0, "get_tokenizer")

    def run(self, query: Query) -> dict[int, Query]:
        with torch.no_grad():
            batch = query.data

            prompt = batch["tokens"]

            prompt = prompt.to(self._device)

            stop_tokens = torch.tensor(self._tokenizer.stop_tokens, device=self._device)
            bsz, prompt_length = prompt.size()
            generated_tokens = prompt.clone()
            # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
            stop_token_reached = torch.zeros(
                bsz, dtype=torch.bool, device=prompt.device
            )
            # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
            # that already hit a stop token
            stop_token_mask = torch.ones(
                (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
            )

            tokens = generate_next_token(
                self._model,
                input_pos=torch.arange(0, prompt_length, device=prompt.device),
                x=prompt,
                temperature=self._temperature,
                top_k=self._top_k,
            )

            generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

            # stop early if we reach a stop token in every seq
            if stop_tokens is not None:
                stop_token_reached = update_stop_tokens_tracker(
                    tokens, stop_tokens, stop_token_reached
                )
                if stop_token_reached.all().item():
                    query.data = self._tokenizer.decode(generated_tokens[0].tolist())
                    outputs = {idx: query for idx in self.output_queues}
                    return outputs

            input_pos = torch.tensor([prompt_length], device=prompt.device)
            for _ in range(self.extra_config.get("max_new_tokens", 300) - 1):
                # update stop_token_mask if we reached a stop token in a previous step
                # by appending the logical not of stop_token_reached to the end of the mask
                # reshaped to be bsz first
                if stop_tokens is not None:
                    stop_token_mask = torch.cat(
                        [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
                    )

                tokens = generate_next_token(
                    self._model,
                    input_pos=input_pos,
                    x=tokens,
                    temperature=self._temperature,
                    top_k=self._top_k,
                )

                generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
                input_pos += 1

                if stop_tokens is not None:
                    stop_token_reached = update_stop_tokens_tracker(
                        tokens, stop_tokens, stop_token_reached
                    )
                    if stop_token_reached.all().item():
                        break

            # mask out generated tokens in seqs that already hit a stop token
            if stop_tokens is not None:
                generated_tokens = generated_tokens * stop_token_mask
                # if pad_id is not 0, replace 0 with pad_id
                if self._tokenizer.pad_id != 0:
                    generated_tokens[generated_tokens == 0] = self._tokenizer.pad_id

            query.data = self._tokenizer.decode(generated_tokens[0].tolist())
            outputs = {idx: query for idx in self.output_queues}
            return outputs
