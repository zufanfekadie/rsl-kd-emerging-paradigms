"""
data/dataset.py
---------------
Strict 4-way data splitting with deterministic seeding for scientific reproducibility.

Splits:
  - Seed Labeled Data (15%): Initial training for Teacher v0
  - Unlabeled Pool (55%): Self-training exploration pool for RSL
  - Validation Holdout (15%): Protected gating evaluation set
  - Final Test Set (15%): Untouched evaluation set for final reporting
"""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def get_data_splits(seed: int = 42, batch_size: int = 32):
    """
    Loads Digits dataset, normalizes features, and partitions deterministically into 4 splits.
    """
    # 1. Set global deterministic seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Load Digits (1797 samples, 64 features, 10 classes)
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    # 3. Standardize features (mean=0, variance=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. First partition: carve out Final Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )

    # 5. Second partition: carve out Validation Holdout (15% of total = ~17.65% of remainder)
    val_ratio = 0.15 / (1.0 - 0.15)
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, stratify=y_train_val
    )

    # 6. Third partition: Split the remaining 70% into Seed (15% of total) and Unlabeled Pool (55% of total)
    seed_ratio = 0.15 / 0.70
    X_seed, X_unlabeled, y_seed, y_unlabeled_ground_truth = train_test_split(
        X_pool, y_pool, train_size=seed_ratio, random_state=seed, stratify=y_pool
    )

    print(f"[*] Dataset Partitioned (seed={seed}):")
    print(f"    - Seed Labeled Set:   {X_seed.shape[0]} samples (Initial Teacher Supervision)")
    print(f"    - Unlabeled Pool:     {X_unlabeled.shape[0]} samples (For Self-Learning Exploration)")
    print(f"    - Holdout Validation: {X_val.shape[0]} samples (Protected Gate Decider)")
    print(f"    - Final Test Set:     {X_test.shape[0]} samples (Untouched Benchmark)")

    # 7. Package PyTorch DataLoaders
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "X_seed": X_seed,
        "y_seed": y_seed,
        "X_unlabeled": X_unlabeled,
        "y_unlabeled_true": y_unlabeled_ground_truth,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler
    }

if __name__ == "__main__":
    splits = get_data_splits()
