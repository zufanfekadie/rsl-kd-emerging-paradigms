"""
demo.py
-------
Live Interactive Presentation Demo.
Designed for the 5-minute "Results & Demo" segment of your presentation (Slide 34).
Runs in ~10 seconds on CPU.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import get_data_splits
from models.teacher import TeacherMLP
from models.student import StudentMLP
from loop.self_learning_loop import train_model
from loop.distill import train_hard_student, train_distilled_student
from utils.metrics import count_parameters

def live_presentation_demo():
    print("\n" + "="*75)
    print("   LIVE PRESENTATION DEMO: RSL & KNOWLEDGE DISTILLATION IN ACTION")
    print("="*75)
    print("[*] Initializing environment and loading Digits benchmark...")

    data = get_data_splits(seed=42, batch_size=32)
    X_seed, y_seed = data["X_seed"], data["y_seed"]
    X_test = data["test_loader"].dataset.tensors[0]
    y_test = data["test_loader"].dataset.tensors[1]

    # Fast training on seed
    print("\n[1/3] Fast-training Teacher (~60k params) and Student models (~2.4k params)...")
    torch.manual_seed(42)
    teacher = TeacherMLP(input_dim=64, num_classes=10)
    train_model(teacher, X_seed, y_seed, epochs=15)

    hard_student = StudentMLP(input_dim=64, num_classes=10)
    train_hard_student(hard_student, X_seed, y_seed, epochs=15)

    distilled_student = StudentMLP(input_dim=64, num_classes=10)
    train_distilled_student(distilled_student, teacher, X_seed, y_seed, epochs=15, alpha=0.3, temperature=3.5)
    print(">>> Models trained and loaded in RAM!\n")

    # Select an interesting test sample
    sample_idx = 32
    sample_x = X_test[sample_idx:sample_idx+1]
    true_label = y_test[sample_idx].item()

    print(f"[2/3] Evaluating Ambiguous Test Sample #{sample_idx} (Ground Truth: '{true_label}')")
    print("-" * 75)

    # Teacher inference
    teacher.eval()
    with torch.no_grad():
        t_logits = teacher(sample_x)[0]
        p_t1 = F.softmax(t_logits, dim=0).numpy()
        p_t4 = F.softmax(t_logits / 4.0, dim=0).numpy()

    # Latency benchmarks
    hard_student.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            hs_out = hard_student(sample_x)
    hard_time = (time.perf_counter() - t0) / 500 * 1000

    distilled_student.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            ds_out = distilled_student(sample_x)
    dist_time = (time.perf_counter() - t0) / 500 * 1000

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            t_out = teacher(sample_x)
    teacher_time = (time.perf_counter() - t0) / 500 * 1000

    print(f"  {'Model Type':<20} | {'Parameters':<10} | {'Pred':<4} | {'Top-1 Confidence':<16} | {'Latency (per item)':<18}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*4}-+-{'-'*16}-+-{'-'*18}")
    print(f"  {'Teacher (Large)':<20} | {count_parameters(teacher):>10,d} |  {torch.argmax(t_logits).item()}   | {np.max(p_t1)*100:>14.1f}% | {teacher_time:.4f} ms")
    print(f"  {'Hard-Label Student':<20} | {count_parameters(hard_student):>10,d} |  {torch.argmax(hs_out).item()}   | {torch.max(F.softmax(hs_out, dim=1)).item()*100:>14.1f}% | {hard_time:.4f} ms")
    print(f"  {'Distilled Student':<20} | {count_parameters(distilled_student):>10,d} |  {torch.argmax(ds_out).item()}   | {torch.max(F.softmax(ds_out, dim=1)).item()*100:>14.1f}% | {dist_time:.4f} ms")

    print("\n[3/3] Inspecting Dark Knowledge in Teacher Logits for this sample:")
    print("  Class:          0     1     2     3     4     5     6     7     8     9")
    print("  -------------------------------------------------------------------------")
    print("  T=1.0 (Sharp): " + "  ".join([f"{p:.2f}" for p in p_t1]))
    print("  T=4.0 (Soft):  " + "  ".join([f"{p:.2f}" for p in p_t4]))

    top2 = np.argsort(p_t4)[-2:]
    print(f"\n  >>> Dark Knowledge Identified: The Teacher gives secondary probability to class '{top2[0]}' ({p_t4[top2[0]]:.2f})")
    print(f"      while predicting true class '{top2[1]}' ({p_t4[top2[1]]:.2f}). The student learns this relationship!")
    print("="*75 + "\n")

if __name__ == "__main__":
    live_presentation_demo()
