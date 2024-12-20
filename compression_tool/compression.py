import torch
import torch.nn as nn
from torch.nn.utils import prune
from copy import deepcopy

def apply_pruning(model, amount=0.3):
    """
    Apply pruning to the model.
    
    Args:
        model: The model to prune
        amount: The proportion of weights to prune (0-1)
    """
    model_copy = deepcopy(model)
    
    for name, module in model_copy.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')
    
    return model_copy

def apply_quantization(model):
    """
    Apply dynamic quantization to the model.
    """
    # Dynamic quantization is more widely supported than static quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},  # Quantize both linear and conv layers
        dtype=torch.qint8
    )
    return quantized_model