import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import get_model

from stages.stage import Stage, log_phase, log_phase_single


class TorchVisionClassification(Stage):
    """This stage loads a TorchVision model and performs transfer learning
    and/or inference using the pretrained model."""

    def __init__(self, stage_config: dict, parent_name):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Configuration, which includes the name of the model,
            possible checkpoint and change to the head of the classifier.

        Raises:
            Exception: Missing model name.
            Exception: Model checkpoint path is required for inference.
        """
        super().__init__(stage_config, parent_name)

        self.model = None
        self.optimizer = None
        self.criterion = None

        stage_config = stage_config.get("config", {})

        self.model_name = stage_config.get("model", None)
        if self.model_name is None:
            raise ValueError("Missing model name.")
        self.model_checkpoint_path = stage_config.get("model_checkpoint_path", None)
        if self.model_checkpoint_path is None:
            raise ValueError("Model checkpoint path is required for inference.")
        self.replace_classifier = stage_config.get("replace_classifier", False)
        self.num_classes = stage_config.get("num_classes", 1000)

    def replace_last_module(self, module):
        """Replace the last module/layer inside the classifier head of the model.

        Args:
            module (tuple[str, torch.Module]): The last (name, module) tuple in the neural network
        """
        last_module_name = module[0].split(".")

        if len(last_module_name) == 1:
            self.model[last_module_name[0]] = Linear(
                module[1].in_features, self.num_classes
            )
        else:
            self.model.__getattr__(".".join(last_module_name[:-1]))[
                int(last_module_name[-1])
            ] = Linear(module[1].in_features, self.num_classes)

    def get_device(self):
        """Decides between GPU acceleration based on availability.

        Returns:
            str: String representing the type of device used for inference
        """
        if torch.cuda.is_available():
            # print("Using cuda for inference")
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            # print("Using Metal Performance Shaders for inference")
            device = "mps:0"
        else:
            # print("Using CPU for inference")
            device = "cpu"
        return device

    @log_phase
    def prepare(self):
        """Build the model according to the config and load the weights"""
        super().prepare()

        # load the correct torchvision model
        self.model = get_model(self.model_name, weights=None)

        if self.replace_classifier:
            # get the last module of the model and replace it
            *_, last_module = self.model.named_modules()
            self.replace_last_module(last_module)

        # load from checkpoint file
        self.model.load_state_dict(torch.load(self.model_checkpoint_path))

        # set only subset of the weights as trainable parameter (transfer learning)
        for param in self.model.parameters():
            param.requires_grad = False

        # TODO: Parametrize the number of layers to be trainable
        params_to_update = []
        for name, param in self.model.named_parameters():
            if last_module[0] in name:
                param.requires_grad = True
                params_to_update.append(param)

        device = self.get_device()
        self.model = self.model.to(device)

        self.optimizer = SGD(params_to_update, lr=0.001, momentum=0.9)

        self.criterion = CrossEntropyLoss()

    def run(self):
        """Continuously poll for the incomng data in the queue and perform inference"""
        while True:
            data_from_queues = self.get_next_from_queues()

            # check whether the execution has been terminated,
            # if yes send the termination element to following stages.
            if self.is_done(data_from_queues):
                self.push_to_output(None)
                break

            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "start")

            # get data only from the first queue.
            # It is the responsibility of the stage to handle extra inputs if necessary
            data_from_first = list(data_from_queues.values())[0]
            inputs = data_from_first.get("data", None)
            if inputs is None:
                raise ValueError("Did not receive any input from the previous stage")
            [inputs, labels] = inputs
            device = self.get_device()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # check whether training or validation and perform appropriate actions
            split = data_from_first.get("split", "val")
            if split == "val":
                self.model.eval()
            else:
                self.model.train()

            with torch.set_grad_enabled(split == "train"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                if split == "train":
                    loss.backward()
                    self.optimizer.step()

            data_from_first["data"] = [preds, labels]
            self.push_to_output(data_from_first)
            if not self.disable_logs:
                log_phase_single(self.parent_name, self.name, "run", "end")
