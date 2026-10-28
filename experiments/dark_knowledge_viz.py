"""
experiments/dark_knowledge_viz.py
---------------------------------
Visualizes "Dark Knowledge" (Hinton et al., 2015; Slide 17 & 18).

Compares:
  1. One-hot hard ground truth
  2. Teacher output distribution at T=1.0 (Sharp)
  3. Teacher output distribution at T=4.0 (Flattened to reveal inter-class similarities)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_data_splits
from models.teacher import TeacherMLP
from loop.self_learning_loop import train_model

def visualize_dark_knowledge(seed: int = 42):
    print("="*80)
    print("VISUALIZING DARK KNOWLEDGE (TEMPERATURE SCALING IN LOGITS)")
    print("="*80)

    data = get_data_splits(seed=seed, batch_size=32)
    X_seed, y_seed = data["X_seed"], data["y_seed"]
    X_unlabeled = data["X_unlabeled"]
    y_unlabeled_true = data["y_unlabeled_true"]

    # 1. Train Teacher
    torch.manual_seed(seed)
    teacher = TeacherMLP(input_dim=64, num_classes=10)
    # Train on combined seed + pool to get rich logits
    X_full = np.concatenate([X_seed, X_unlabeled], axis=0)
    y_full = np.concatenate([y_seed, y_unlabeled_true], axis=0)
    train_model(teacher, X_full, y_full, epochs=25)
    teacher.eval()

    # 2. Find an interesting/ambiguous sample in test set
    X_test = data["test_loader"].dataset.tensors[0]
    y_test = data["test_loader"].dataset.tensors[1]

    with torch.no_grad():
        logits_all = teacher(X_test)
        probs_all = F.softmax(logits_all, dim=1).numpy()

    # Find a sample where top-1 prob is between 0.70 and 0.92 (ambiguous with a secondary class)
    max_probs = np.max(probs_all, axis=1)
    ambiguous_indices = np.where((max_probs > 0.65) & (max_probs < 0.93))[0]
    idx = ambiguous_indices[0] if len(ambiguous_indices) > 0 else 0

    target_sample = X_test[idx:idx+1]
    true_label = y_test[idx].item()

    with torch.no_grad():
        sample_logits = teacher(target_sample)[0]
        p_t1 = F.softmax(sample_logits / 1.0, dim=0).numpy()
        p_t4 = F.softmax(sample_logits / 4.0, dim=0).numpy()

    # 3. Create High-Impact Comparative Bar Plot
    classes = [str(c) for c in range(10)]
    output_dir = "artifacts/figures"
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Subplot 1: Hard Ground Truth Label
    one_hot = np.zeros(10)
    one_hot[true_label] = 1.0
    ax1.bar(classes, one_hot, color='#334155', width=0.6)
    ax1.set_title(f"1. Hard Label (Ground Truth: '{true_label}')", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Probability", fontsize=10)
    ax1.set_ylim(0, 1.1)

    # Subplot 2: Teacher at T=1.0
    ax2.bar(classes, p_t1, color='#0284c7', width=0.6)
    ax2.set_title("2. Teacher Output at T = 1.0 (Standard)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Probability", fontsize=10)
    ax2.set_ylim(0, 1.1)

    # Subplot 3: Teacher at T=4.0 (Dark Knowledge revealed)
    bars3 = ax3.bar(classes, p_t4, color='#7c3aed', width=0.6)
    ax3.set_title("3. Soft Target at T = 4.0 ('Dark Knowledge')", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Softened Probability", fontsize=10)
    ax3.set_ylim(0, 0.45)

    # Annotate secondary probabilities
    top2_idx = np.argsort(p_t4)[-2:]
    for i in top2_idx:
        ax3.annotate(f"{p_t4[i]:.2f}", (classes[i], p_t4[i] + 0.015), ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "dark_knowledge.png")
    plt.savefig(chart_path)
    plt.close()

    print(f"[*] Ambiguous Sample Index: {idx} (True Label: {true_label})")
    print(f"[*] Top class at T=1: {np.argmax(p_t1)} (prob={np.max(p_t1):.4f})")
    print(f"[*] Chart successfully saved to: {chart_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    visualize_dark_knowledge(seed=42)
