"""
experiments/multi_seed.py
-------------------------
Multi-Seed Scientific Evaluation Suite.
Meets the explicit requirement on Slide 32:
"Required comparison: Accuracy, model size and latency, over at least two random seeds."

Runs experiments across seeds [42, 1337, 2026], aggregates results,
and computes Mean +/- Standard Deviation.
"""

import os
import sys
import numpy as np
import torch

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_data_splits
from models.teacher import TeacherMLP
from models.student import StudentMLP
from loop.verifier import ExternalVerifier
from loop.self_learning_loop import run_bounded_rsl_loop, train_model
from loop.distill import train_hard_student, train_distilled_student
from utils.metrics import count_parameters, benchmark_cpu_latency, evaluate_accuracy

def run_multi_seed_benchmark(seeds=[42, 1337, 2026]):
    print("="*85)
    print(f"RUNNING MULTI-SEED SCIENTIFIC BENCHMARK ACROSS SEEDS: {seeds}")
    print("="*85)

    all_results = {
        "Teacher v0 (Seed)": {"test_accs": [], "latencies": []},
        "Teacher Final (RSL)": {"test_accs": [], "latencies": []},
        "Hard-Label Student": {"test_accs": [], "latencies": []},
        "Distilled Student": {"test_accs": [], "latencies": []}
    }

    rsl_stats = {"total_rounds": 0, "accepted_rounds": 0, "rejected_rounds": 0}

    for seed in seeds:
        print(f"\n>>>>>>>>>>>> [STARTING RUN WITH SEED: {seed}] <<<<<<<<<<<<")
        data = get_data_splits(seed=seed, batch_size=32)
        X_seed, y_seed = data["X_seed"], data["y_seed"]
        X_unlabeled, y_unlabeled_true = data["X_unlabeled"], data["y_unlabeled_true"]
        val_loader = data["val_loader"]
        test_loader = data["test_loader"]

        # 1. Teacher v0
        torch.manual_seed(seed)
        teacher_v0 = TeacherMLP(input_dim=64, num_classes=10)
        train_model(teacher_v0, X_seed, y_seed, epochs=30)
        t0_acc = evaluate_accuracy(teacher_v0, test_loader)
        t0_lat = benchmark_cpu_latency(teacher_v0, input_dim=64, num_runs=500)
        all_results["Teacher v0 (Seed)"]["test_accs"].append(t0_acc)
        all_results["Teacher v0 (Seed)"]["latencies"].append(t0_lat)

        # 2. RSL Loop
        verifier = ExternalVerifier(ground_truth_labels=y_unlabeled_true, confidence_threshold=0.85)
        improved_teacher, final_X, final_y, history = run_bounded_rsl_loop(
            teacher_v0=teacher_v0,
            X_seed=X_seed,
            y_seed=y_seed,
            X_unlabeled=X_unlabeled,
            verifier=verifier,
            val_loader=val_loader,
            num_rounds=3,
            epsilon_gain=0.005
        )
        t_rsl_acc = evaluate_accuracy(improved_teacher, test_loader)
        t_rsl_lat = benchmark_cpu_latency(improved_teacher, input_dim=64, num_runs=500)
        all_results["Teacher Final (RSL)"]["test_accs"].append(t_rsl_acc)
        all_results["Teacher Final (RSL)"]["latencies"].append(t_rsl_lat)

        for h in history:
            rsl_stats["total_rounds"] += 1
            if h["accepted"]:
                rsl_stats["accepted_rounds"] += 1
            else:
                rsl_stats["rejected_rounds"] += 1

        # 3. Hard-Label Student
        torch.manual_seed(seed)
        hard_student = StudentMLP(input_dim=64, num_classes=10)
        train_hard_student(hard_student, final_X, final_y, epochs=35)
        hard_acc = evaluate_accuracy(hard_student, test_loader)
        hard_lat = benchmark_cpu_latency(hard_student, input_dim=64, num_runs=500)
        all_results["Hard-Label Student"]["test_accs"].append(hard_acc)
        all_results["Hard-Label Student"]["latencies"].append(hard_lat)

        # 4. Distilled Student
        torch.manual_seed(seed)
        distilled_student = StudentMLP(input_dim=64, num_classes=10)
        train_distilled_student(distilled_student, improved_teacher, final_X, final_y, epochs=35, alpha=0.3, temperature=3.5)
        dist_acc = evaluate_accuracy(distilled_student, test_loader)
        dist_lat = benchmark_cpu_latency(distilled_student, input_dim=64, num_runs=500)
        all_results["Distilled Student"]["test_accs"].append(dist_acc)
        all_results["Distilled Student"]["latencies"].append(dist_lat)

    # Statistical Aggregation
    print("\n" + "#"*90)
    print("               MULTI-SEED STATISTICAL BENCHMARK RESULTS (Mean +/- Std)")
    print("#"*90)
    print(f"{'Model Architecture':<24} | {'Test Accuracy (%)':<22} | {'CPU Latency (ms)':<20} | {'Params':<10}")
    print("-" * 90)

    sample_teacher = TeacherMLP()
    sample_student = StudentMLP()
    params_map = {
        "Teacher v0 (Seed)": count_parameters(sample_teacher),
        "Teacher Final (RSL)": count_parameters(sample_teacher),
        "Hard-Label Student": count_parameters(sample_student),
        "Distilled Student": count_parameters(sample_student),
    }

    for model_name, data in all_results.items():
        accs = np.array(data["test_accs"]) * 100
        lats = np.array(data["latencies"])
        p = params_map[model_name]

        acc_str = f"{accs.mean():.2f}% +/- {accs.std():.2f}%"
        lat_str = f"{lats.mean():.2f}ms +/- {lats.std():.2f}ms"
        print(f"{model_name:<24} | {acc_str:<22} | {lat_str:<20} | {p:>8,d}")

    print("#"*90)
    print(f"\nRSL Gating Summary across {rsl_stats['total_rounds']} total rounds:")
    print(f"  - Accepted Updates: {rsl_stats['accepted_rounds']} ({rsl_stats['accepted_rounds']/rsl_stats['total_rounds']*100:.1f}%)")
    print(f"  - Rejected & Rolled Back: {rsl_stats['rejected_rounds']} ({rsl_stats['rejected_rounds']/rsl_stats['total_rounds']*100:.1f}%)")
    print("="*90 + "\n")

if __name__ == "__main__":
    run_multi_seed_benchmark(seeds=[42, 1337, 2026])
