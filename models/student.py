"""
models/student.py
-----------------
Compact, lightweight Multi-Layer Perceptron (Student).
Designed for low-latency, edge/CPU deployment with minimal parameter count.
"""

import torch
import torch.nn as nn

class StudentMLP(nn.Module):
    def __init__(self, input_dim: int = 64, num_classes: int = 10):
        super(StudentMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    s = StudentMLP()
    print(f"Student Total Trainable Parameters: {count_parameters(s):,}")
