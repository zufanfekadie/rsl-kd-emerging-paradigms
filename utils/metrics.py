"""
utils/metrics.py
----------------
Benchmark utilities:
  - Model parameter counting
  - Precision CPU inference latency benchmarking
  - Test set accuracy computation
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable floating-point parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_cpu_latency(model: nn.Module, input_dim: int = 64, num_runs: int = 1000) -> float:
    """
    Measures average CPU inference latency in milliseconds per 100 samples.
    Performs 100 warm-up runs to stabilize CPU cache.
    """
    model.eval()
    dummy_input = torch.randn(100, input_dim)

    # Warm-up
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # Timed benchmark
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    total_time = time.perf_counter() - start_time

    return (total_time / num_runs) * 1000.0

def evaluate_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    """Computes top-1 classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0
