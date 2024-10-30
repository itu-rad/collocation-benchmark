"""Prints all of the TorchVision models and classifcation models specifically.
"""

import torchvision.models

print("All models: ", torchvision.models.list_models())
print("Classification models: ", torchvision.models.list_models(torchvision.models))
