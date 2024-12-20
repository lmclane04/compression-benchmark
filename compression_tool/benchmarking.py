import time
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from .compression import apply_pruning, apply_quantization

def evaluate_model(model: torch.nn.Module, 
                  data_loader: torch.utils.data.DataLoader, 
                  device: str = "cpu") -> Tuple[float, float]:
    """
    Evaluate the accuracy and inference time of the model.
    
    Returns:
        Tuple[float, float]: (accuracy percentage, average inference time per batch)
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_time = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time for this batch
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            total_time += end_time - start_time
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    avg_inference_time = total_time / len(data_loader)
    
    return accuracy, avg_inference_time


def get_model_size(model: torch.nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 * 1024)

def benchmark_compression(model, data_loader, device="cpu"):
    """
    Run benchmarks for both the original and compressed model.
    """
    # Evaluate original model
    original_acc, original_time = evaluate_model(model, data_loader, device)
    
    # Move to CPU for compression operations
    model = model.cpu()
    
    # Apply compression
    compressed_model = model  # Start with original model
    
    # Apply pruning first
    compressed_model = apply_pruning(compressed_model)
    
    # Then apply quantization
    try:
        compressed_model = apply_quantization(compressed_model)
    except Exception as e:
        print(f"Warning: Quantization failed with error: {str(e)}")
        print("Proceeding with only pruning...")
    
    # Evaluate compressed model
    compressed_acc, compressed_time = evaluate_model(compressed_model, data_loader, device)
    
    results = {
        "original": {
            "accuracy": original_acc, 
            "inference_time": original_time
        },
        "compressed": {
            "accuracy": compressed_acc, 
            "inference_time": compressed_time
        }
    }
    
    return results