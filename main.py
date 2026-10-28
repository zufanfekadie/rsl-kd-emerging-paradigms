"""
main.py
-------
Master Execution Script for the Combined RSL + Knowledge Distillation Assignment.
Meets all criteria for Track A, Track B, and the Stretch Goal.
"""

import os
import copy
import torch
import numpy as np

from data.dataset import get_data_splits
from models.teacher import TeacherMLP
from models.student import StudentMLP
from loop.verifier import ExternalVerifier
from loop.self_learning_loop import run_bounded_rsl_loop, train_model
from loop.distill import train_hard_student, train_distilled_student
from utils.metrics import count_parameters, benchmark_cpu_latency, evaluate_accuracy
from utils.plotting import generate_evaluation_charts

def run_experiment(seed: int = 42, num_rsl_rounds: int = 3, kd_temp: float = 3.5, kd_alpha: float = 0.3):
    print("="*80)
    print("   ICOG LABS AI INTERNSHIP - APPLIED ASSIGNMENT")
    print("   STRETCH GOAL: BOUNDED RECURSIVE SELF-LEARNING + KNOWLEDGE DISTILLATION")
    print("="*80)

    # 1. Dataset Split
    data = get_data_splits(seed=seed, batch_size=32)
    X_seed, y_seed = data["X_seed"], data["y_seed"]
    X_unlabeled, y_unlabeled_true = data["X_unlabeled"], data["y_unlabeled_true"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    # 2. Stage 1: Initial Teacher v0
    torch.manual_seed(seed)
    teacher_v0 = TeacherMLP(input_dim=64, num_classes=10)
    train_model(teacher_v0, X_seed, y_seed, epochs=30)
    t0_test_acc = evaluate_accuracy(teacher_v0, test_loader)
    print(f"[*] Initial Teacher v0 Trained on Seed Data (15%). Test Accuracy: {t0_test_acc*100:.2f}%")

    # 3. Stage 2: Recursive Self-Learning Loop
    oracle_verifier = ExternalVerifier(ground_truth_labels=y_unlabeled_true, confidence_threshold=0.85)
    improved_teacher, final_X_train, final_y_train, rsl_history = run_bounded_rsl_loop(
        teacher_v0=teacher_v0,
        X_seed=X_seed,
        y_seed=y_seed,
        X_unlabeled=X_unlabeled,
        verifier=oracle_verifier,
        val_loader=val_loader,
        num_rounds=num_rsl_rounds,
        epsilon_gain=0.005
    )
    t_improved_test_acc = evaluate_accuracy(improved_teacher, test_loader)
    print(f"[*] Improved Teacher Test Accuracy: {t_improved_test_acc*100:.2f}% (RSL Gain: {(t_improved_test_acc - t0_test_acc)*100:+.2f}%)")

    # 4. Stage 3: Knowledge Distillation
    print("\n" + "="*70)
    print("STARTING KNOWLEDGE DISTILLATION (KD) EXPERIMENTS")
    print("="*70)

    # Variant 1: Baseline Student (Hard Labels only)
    torch.manual_seed(seed)
    hard_student = StudentMLP(input_dim=64, num_classes=10)
    print("[*] Training Hard-Label Student Baseline (alpha=1.0, no teacher)...")
    train_hard_student(hard_student, final_X_train, final_y_train, epochs=35)
    hard_student_test_acc = evaluate_accuracy(hard_student, test_loader)
    print(f"    -> Hard Student Test Accuracy: {hard_student_test_acc*100:.2f}%")

    # Variant 2: Distilled Student (Teacher Soft Targets + Temperature)
    torch.manual_seed(seed)
    distilled_student = StudentMLP(input_dim=64, num_classes=10)
    print(f"[*] Training Distilled Student (alpha={kd_alpha}, Temperature T={kd_temp})...")
    train_distilled_student(
        student_model=distilled_student,
        teacher_model=improved_teacher,
        X=final_X_train,
        y=final_y_train,
        epochs=35,
        alpha=kd_alpha,
        temperature=kd_temp
    )
    distilled_student_test_acc = evaluate_accuracy(distilled_student, test_loader)
    print(f"    -> Distilled Student Test Accuracy: {distilled_student_test_acc*100:.2f}%")
    print("="*70 + "\n")

    # 5. Benchmarking Latency and Parameters
    print("[*] Benchmarking Model Parameters and CPU Latency (1,000 runs)...")
    models = {
        "Teacher v0 (Seed)": teacher_v0,
        "Teacher Final (RSL)": improved_teacher,
        "Hard-Label Student": hard_student,
        "Distilled Student": distilled_student
    }

    results = {}
    for name, model in models.items():
        params = count_parameters(model)
        latency = benchmark_cpu_latency(model, input_dim=64, num_runs=1000)
        test_acc = evaluate_accuracy(model, test_loader)
        results[name] = {
            "params": params,
            "latency_ms": latency,
            "test_acc": test_acc
        }

    # 6. Final Comparative Results Table
    print("\n" + "#"*80)
    print("                         FINAL BENCHMARK RESULTS")
    print("#"*80)
    print(f"{'Model Architecture':<24} | {'Test Acc (%)':<14} | {'Params':<10} | {'Compression':<12} | {'CPU Latency (ms)':<16}")
    print("-" * 84)

    teacher_params = results["Teacher Final (RSL)"]["params"]
    for name, m in results.items():
        comp_ratio = f"{teacher_params / m['params']:.1f}x" if m['params'] < teacher_params else "1.0x"
        print(f"{name:<24} | {m['test_acc']*100:>10.2f}%   | {m['params']:>8,d} | {comp_ratio:>10} | {m['latency_ms']:>12.2f} ms")
    print("#"*80 + "\n")

    # 7. Generate Visualizations
    generate_evaluation_charts(results, rsl_history, output_dir="artifacts/figures")
    return results, rsl_history

if __name__ == "__main__":
    run_experiment(seed=42, num_rsl_rounds=3, kd_temp=3.5, kd_alpha=0.3)
