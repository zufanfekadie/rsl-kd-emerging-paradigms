"""
loop/self_learning_loop.py
--------------------------
Bounded Recursive Self-Learning (RSL) loop with an independent verifier,
protected holdout validation gate, and automatic rollback on regression.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def evaluate_model(model: nn.Module, data_loader: DataLoader) -> float:
    """Computes categorical accuracy on a DataLoader."""
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

def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 25, lr: float = 1e-3, batch_size: int = 32):
    """Trains a model with Adam optimizer and Cross-Entropy loss."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

def run_bounded_rsl_loop(
    teacher_v0: nn.Module,
    X_seed: np.ndarray,
    y_seed: np.ndarray,
    X_unlabeled: np.ndarray,
    verifier,
    val_loader: DataLoader,
    num_rounds: int = 3,
    epsilon_gain: float = 0.005
):
    """
    Executes N rounds of bounded self-learning with validation gating.
    """
    print("\n" + "="*70)
    print("STARTING BOUNDED RECURSIVE SELF-LEARNING (RSL) LOOP")
    print("="*70)

    # 1. Establish baseline from initial seed training
    current_teacher = copy.deepcopy(teacher_v0)
    train_model(current_teacher, X_seed, y_seed, epochs=30)
    baseline_val_acc = evaluate_model(current_teacher, val_loader)
    print(f"[Initial] Teacher v0 Validation Accuracy on Holdout: {baseline_val_acc*100:.2f}%\n")

    current_X_train = np.copy(X_seed)
    current_y_train = np.copy(y_seed)

    loop_history = []

    for round_id in range(1, num_rounds + 1):
        print(f"--- [Round {round_id} / {num_rounds}] ---")

        # Stage 1: Generate predictions & confidences on the unlabelled pool
        current_teacher.eval()
        with torch.no_grad():
            tensor_unlabeled = torch.from_numpy(X_unlabeled)
            logits = current_teacher(tensor_unlabeled)
            probabilities = torch.softmax(logits, dim=1).numpy()
            proposed_labels = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)

        # Stage 2: Independent verification
        verif_result = verifier.verify(proposed_labels, confidences)
        mask = verif_result["verified_mask"]

        print(f"  • Candidate generation: {verif_result['n_total']} unlabeled items")
        print(f"  • High-confidence outputs: {verif_result['n_high_conf']}")
        print(f"  • Verified by Oracle: {verif_result['n_verified']} (Passed into training pool)")
        print(f"  • Hallucinations rejected: {verif_result['n_rejected_hallucinations']}")

        if verif_result['n_verified'] == 0:
            print("  [!] Warning: Zero verified samples found in this round. Skipping update.")
            continue

        # Stage 3: Curate dataset (combine seed with newly verified samples)
        X_verified = X_unlabeled[mask]
        y_verified = proposed_labels[mask]

        candidate_X_train = np.concatenate([X_seed, X_verified], axis=0)
        candidate_y_train = np.concatenate([y_seed, y_verified], axis=0)

        # Stage 4: Train on a COPY (never mutate baseline directly)
        proposal_teacher = copy.deepcopy(current_teacher)
        train_model(proposal_teacher, candidate_X_train, candidate_y_train, epochs=25)

        # Stage 5: Evaluate on the protected validation holdout
        proposal_val_acc = evaluate_model(proposal_teacher, val_loader)
        gain = proposal_val_acc - baseline_val_acc
        print(f"  • Candidate Holdout Acc: {proposal_val_acc*100:.2f}% (Gain: {gain*100:+.2f}%)")

        # Stage 6: Gate decision
        round_info = {
            "round": round_id,
            "baseline_acc": baseline_val_acc,
            "proposal_acc": proposal_val_acc,
            "gain": gain,
            "verified_samples": verif_result['n_verified'],
            "accepted": False
        }

        if proposal_val_acc >= (baseline_val_acc + epsilon_gain):
            print(f"  >>> ACCEPTED: Gain exceeds margin eps={epsilon_gain*100:.2f}%. Updating checkpoint.")
            current_teacher = proposal_teacher
            baseline_val_acc = proposal_val_acc
            current_X_train = candidate_X_train
            current_y_train = candidate_y_train
            round_info["accepted"] = True
        else:
            print(f"  >>> REJECTED & ROLLED BACK: Gain below threshold. Preserving previous checkpoint.")
            del proposal_teacher

        loop_history.append(round_info)
        print()

    print("="*70)
    print(f"RSL LOOP COMPLETED. Final Improved Teacher Holdout Acc: {baseline_val_acc*100:.2f}%")
    print("="*70 + "\n")

    return current_teacher, current_X_train, current_y_train, loop_history
