from .stage import Stage

from .data_loading.torch_from_dataset import TorchFromDataset
from .classification.torchvision_classification import TorchVisionClassification
from .dataset.torchvision_dataset import TorchVisionDataset
from .preprocessing.torchvision_preprocess_from_weights import (
    TorchVisionPreprocessFromWeights,
)


STAGE_REGISTRY: dict[str, dict[str, Stage]] = {
    "data_loading": {"torch_from_dataset": TorchFromDataset},
    "inference": {"torchvision_classification": TorchVisionClassification},
    "preprocessing": {
        "torchvision_preprocess_from_weights": TorchVisionPreprocessFromWeights
    },
    "dataset": {"torchvision_dataset": TorchVisionDataset},
}
