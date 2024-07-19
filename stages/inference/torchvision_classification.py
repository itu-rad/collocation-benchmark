import torch
from torch.nn import Linear
from torchvision.models import get_model

from stages.stage import Stage, log_phase


class TorchVisionClassification(Stage):
    model = None
    model_name = None
    model_checkpoint_path = None
    replace_classifier = False
    num_classes = 1000

    def __init__(self, stage_config: dict):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Configuration, which includes the name of the model,
            possible checkpoint and change to the head of the classifier.

        Raises:
            Exception: Missing model name.
            Exception: Model checkpoint path is required for inference.
        """
        super().__init__(stage_config)
        stage_config = stage_config.get("config", {})

        self.model_name = stage_config.get("model", None)
        if self.model_name is None:
            raise Exception("Missing model name.")
        self.model_checkpoint_path = stage_config.get("model_checkpoint_path", None)
        if self.model_name is None:
            raise Exception("Model checkpoint path is required for inference.")
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
        # load the correct torchvision model
        self.model = get_model(self.model_name, weights=None)

        if self.replace_classifier:
            # get the last module of the model and replace it
            *_, last_module = self.model.named_modules()
            self.replace_last_module(last_module)

        self.model.load_state_dict(torch.load(self.model_checkpoint_path))

        for param in self.model.parameters():
            param.requires_grad = False

        device = self.get_device()
        self.model = self.model.to(device)
        self.model.eval()

    @log_phase
    def run(self, data):
        """Run inference query

        Args:
            data (Tensor): Input data in the Torch format with size matching the input
            dimensons of the model.
        """

        inputs = data.get("data", None)
        if inputs is None:
            raise Exception("Did not receive any input from the previous stage")

        [inputs, labels] = inputs
        device = self.get_device()
        inputs = inputs.to(device)

        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)

        data["data"] = [preds, labels]

        return data
