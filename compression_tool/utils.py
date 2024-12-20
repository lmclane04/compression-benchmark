import os
import json
from pathlib import Path
from typing import Dict, Any

def save_results(results: Dict[str, Any], filepath: str = "results/metrics.json"):
    """Save benchmark results to JSON file"""
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

def load_results(filepath: str = "results/metrics.json") -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found at {filepath}")
    
    with open(filepath, "r") as f:
        return json.load(f)