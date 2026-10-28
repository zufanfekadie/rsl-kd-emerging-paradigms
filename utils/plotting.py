"""
utils/plotting.py
-----------------
Automated chart generation for the final presentation and report.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_evaluation_charts(results: dict, rsl_history: list, output_dir: str = "artifacts/figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Chart 1: RSL Progression Across Rounds
    rounds = [0] + [h["round"] for h in rsl_history]
    accs = [rsl_history[0]["baseline_acc"] * 100] + [h["proposal_acc"] * 100 for h in rsl_history]
    accepted_status = [True] + [h["accepted"] for h in rsl_history]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(rounds, accs, marker='o', linewidth=2.5, color='#0284c7', label='Holdout Accuracy (%)')

    for r, acc, accpt in zip(rounds, accs, accepted_status):
        color = '#16a34a' if accpt else '#dc2626'
        label = 'Accepted' if accpt else 'Rejected (Rolled back)'
        ax.scatter(r, acc, color=color, s=120, zorder=5)
        ax.annotate(f"{acc:.1f}%", (r, acc + 0.4), ha='center', fontweight='bold', fontsize=9)

    ax.set_title("Track A: Bounded Recursive Self-Learning Trajectory", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("RSL Loop Round", fontsize=11)
    ax.set_ylabel("Holdout Validation Accuracy (%)", fontsize=11)
    ax.set_xticks(rounds)
    ax.set_ylim(min(accs) - 3, max(accs) + 4)
    ax.legend(loc='lower right')
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "rsl_trajectory.png")
    plt.savefig(chart1_path)
    plt.close()

    # Chart 2: Model Comparison (Accuracy vs Parameter Count vs Latency)
    model_names = list(results.keys())
    test_accs = [results[m]["test_acc"] * 100 for m in model_names]
    params = [results[m]["params"] for m in model_names]
    latencies = [results[m]["latency_ms"] for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Test Accuracy vs Parameters
    colors = ['#64748b', '#0284c7', '#ea580c', '#16a34a']
    bars = ax1.bar(model_names, test_accs, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    ax1.set_title("Final Test Accuracy by Model", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Accuracy on Untouched Test Set (%)", fontsize=11)
    ax1.set_ylim(min(test_accs) - 5, 102)
    ax1.tick_params(axis='x', rotation=18)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot 2: Inference Latency
    bars2 = ax2.bar(model_names, latencies, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    ax2.set_title("CPU Inference Latency (ms per 100 samples)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Latency (ms) - Lower is Faster", fontsize=11)
    ax2.tick_params(axis='x', rotation=18)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}ms", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(chart2_path)
    plt.close()

    print(f"[*] Visualizations saved:")
    print(f"    - {chart1_path}")
    print(f"    - {chart2_path}")
    return chart1_path, chart2_path
