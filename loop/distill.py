"""
loop/distill.py
---------------
Knowledge Distillation Engine (Hinton et al., 2015).

Trains two student variants for rigorous comparison:
  1. Hard-Label Student: Trained ONLY on one-hot targets using Cross-Entropy (alpha=1.0)
  2. Distilled Student: Trained with combined Hard + Soft Targets loss (Equation from Slide 21)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class DistillationLoss(nn.Module):
    """
    Combined Distillation Loss:
      L = alpha * CrossEntropy(y, student_logits) + (1 - alpha) * T^2 * KL(teacher_soft || student_soft)
    """
    def __init__(self, alpha: float = 0.3, temperature: float = 3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # 1. Hard label loss at T=1
        hard_loss = self.ce_loss(student_logits, labels)

        # 2. Soft targets loss at Temperature T
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft)

        # 3. Scale soft loss by T^2 to match gradient magnitudes (Slide 21)
        combined_loss = (self.alpha * hard_loss) + ((1.0 - self.alpha) * (self.temperature ** 2) * soft_loss)
        return combined_loss, hard_loss, soft_loss

def train_hard_student(student_model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 1e-3, batch_size: int = 32):
    """Trains a baseline student purely on hard ground-truth labels."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)

    student_model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = student_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return student_model

def train_distilled_student(
    student_model: nn.Module,
    teacher_model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    alpha: float = 0.3,
    temperature: float = 3.0
):
    """
    Trains a student model using soft targets from the frozen teacher + ground truth labels.
    """
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    kd_criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)

    teacher_model.eval()
    student_model.train()

    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(bx)

            student_logits = student_model(bx)
            loss, _, _ = kd_criterion(student_logits, teacher_logits, by)
            loss.backward()
            optimizer.step()

    return student_model
