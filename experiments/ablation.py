"""
experiments/ablation.py
-----------------------
Ablation Study on Temperature (T) and Alpha (Loss Balance).

Directly addresses the question from Demo B (Slide 26):
"What would raising T, or setting alpha = 1, do to training?"
"Answer key: alpha = 1 removes the teacher entirely... Very high T flattens
targets towards uniform, so the signal fades."
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_data_splits
from models.teacher import TeacherMLP
from models.student import StudentMLP
from loop.verifier import ExternalVerifier
from loop.self_learning_loop import run_bounded_rsl_loop, train_model
from loop.distill import train_distilled_student, train_hard_student
from utils.metrics import evaluate_accuracy

def run_ablation_study(seed: int = 42):
    print("="*80)
    print("RUNNING KNOWLEDGE DISTILLATION ABLATION STUDY: TEMPERATURE (T) & ALPHA")
    print("="*80)

    data = get_data_splits(seed=seed, batch_size=32)
    X_seed, y_seed = data["X_seed"], data["y_seed"]
    X_unlabeled, y_unlabeled_true = data["X_unlabeled"], data["y_unlabeled_true"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    # 1. Train Improved Teacher
    torch.manual_seed(seed)
    teacher = TeacherMLP(input_dim=64, num_classes=10)
    verifier = ExternalVerifier(ground_truth_labels=y_unlabeled_true, confidence_threshold=0.85)
    improved_teacher, final_X, final_y, _ = run_bounded_rsl_loop(
        teacher_v0=teacher,
        X_seed=X_seed,
        y_seed=y_seed,
        X_unlabeled=X_unlabeled,
        verifier=verifier,
        val_loader=val_loader,
        num_rounds=2,
        epsilon_gain=0.005
    )

    # 2. Temperature Sweep (with fixed alpha = 0.5)
    temperatures = [1.0, 1.5, 2.5, 5.0, 10.0, 20.0]
    temp_accuracies = []

    print("\n--- Temperature Sweep (alpha=0.5) ---")
    for T in temperatures:
        torch.manual_seed(seed)
        student = StudentMLP(input_dim=64, num_classes=10)
        train_distilled_student(student, improved_teacher, final_X, final_y, epochs=35, alpha=0.5, temperature=T)
        acc = evaluate_accuracy(student, test_loader)
        temp_accuracies.append(acc * 100)
        print(f"  • Temperature T={T:<4} -> Test Accuracy: {acc*100:.2f}%")

    # 3. Alpha Sweep (with optimal T = 2.0)
    alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
    alpha_accuracies = []

    print("\n--- Alpha Sweep (Temperature T=2.0) ---")
    for a in alphas:
        torch.manual_seed(seed)
        student = StudentMLP(input_dim=64, num_classes=10)
        train_distilled_student(student, improved_teacher, final_X, final_y, epochs=35, alpha=a, temperature=2.0)
        acc = evaluate_accuracy(student, test_loader)
        alpha_accuracies.append(acc * 100)
        label_desc = "Pure Soft Targets (no labels)" if a == 0.0 else ("Pure Hard Labels (no teacher)" if a == 1.0 else "Combined")
        print(f"  • Alpha alpha={a:<4} ({label_desc}) -> Test Accuracy: {acc*100:.2f}%")

    # 4. Generate Plot
    output_dir = "artifacts/figures"
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    # Plot 1: Temperature vs Accuracy
    ax1.plot(temperatures, temp_accuracies, marker='o', linewidth=2.2, color='#7c3aed')
    ax1.set_title("Sensitivity to Temperature (T)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Temperature T (Log scale)", fontsize=11)
    ax1.set_ylabel("Student Test Accuracy (%)", fontsize=11)
    ax1.set_xscale('log')
    ax1.set_xticks(temperatures)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    for t_val, acc in zip(temperatures, temp_accuracies):
        ax1.annotate(f"{acc:.1f}%", (t_val, acc + 0.15), ha='center', fontsize=9, fontweight='bold')

    # Plot 2: Alpha vs Accuracy
    ax2.plot(alphas, alpha_accuracies, marker='s', linewidth=2.2, color='#0284c7')
    ax2.set_title("Sensitivity to Alpha (Loss Balancing)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Alpha (0.0 = Pure Soft, 1.0 = Pure Hard)", fontsize=11)
    ax2.set_ylabel("Student Test Accuracy (%)", fontsize=11)
    for a_val, acc in zip(alphas, alpha_accuracies):
        ax2.annotate(f"{acc:.1f}%", (a_val, acc + 0.15), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "ablation_curve.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"\n[*] Ablation chart saved to: {chart_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_ablation_study(seed=42)
