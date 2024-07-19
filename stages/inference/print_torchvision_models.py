import torchvision.models

print("All models: ", torchvision.models.list_models())
print("Classification models: ", torchvision.models.list_models(torchvision.models))
