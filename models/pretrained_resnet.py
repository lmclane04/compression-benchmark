# import torch
# import torchvision.models as models

# def load_resnet18(pretrained=True):
#     """
#     Load a ResNet18 model.
    
#     Args:
#         pretrained (bool): Whether to load pretrained weights.
    
#     Returns:
#         torch.nn.Module: ResNet18 model.
#     """
#     model = models.resnet18(pretrained=pretrained)
#     model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes (example for MNIST)
#     return model

import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import the correct weights enum

def load_resnet18(weights=ResNet18_Weights.DEFAULT):
    """
    Load a ResNet18 model.
    
    Args:
        weights (torchvision.models.ResNet18_Weights): The type of pretrained weights to load.
    
    Returns:
        torch.nn.Module: ResNet18 model.
    """
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes (example for MNIST)
    return model