"""
models/teacher.py
-----------------
Deep, high-capacity Multi-Layer Perceptron (Teacher).
"""

import torch
import torch.nn as nn

class TeacherMLP(nn.Module):
    def __init__(self, input_dim: int = 64, num_classes: int = 10):
        super(TeacherMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    t = TeacherMLP()
    print(f"Teacher Total Trainable Parameters: {count_parameters(t):,}")
