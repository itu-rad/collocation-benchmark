from stages.data_loading.torch_from_dataset import TorchFromDataset
from stages.inference.torchvision_classification import TorchVisionClassification
from stages.preprocessing.torchvision_preprocess_from_weights import (
    TorchVisionPreprocessFromWeights,
)


STAGE_REGISTRY = {
    "data_loading": {"torch_from_dataset": TorchFromDataset},
    "inference": {"torchvision_classification": TorchVisionClassification},
    "preprocessing": {
        "torchvision_preprocess_from_weights": TorchVisionPreprocessFromWeights
    },
}
