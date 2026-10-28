"""
loop/verifier.py
----------------
Independent external verifier.

Key rule from Slide 10 & 15:
"A self-learning loop is only as good as its verifier. An external verifier
must check outputs before they can enter the training batch."
"""

import numpy as np
import torch

class ExternalVerifier:
    """
    Simulates an external verification oracle (e.g. unit tests, consistency checks,
    or ground-truth sandbox verifier).
    """
    def __init__(self, ground_truth_labels: np.ndarray, confidence_threshold: float = 0.85):
        self.ground_truth = ground_truth_labels
        self.confidence_threshold = confidence_threshold

    def verify(self, proposed_labels: np.ndarray, model_confidences: np.ndarray):
        """
        Takes candidate pseudo-labels and checks them against the independent oracle.
        Returns a boolean mask: True for samples that pass both the confidence threshold
        and the oracle verification check.
        """
        assert len(proposed_labels) == len(self.ground_truth)

        # 1. Check if model's own confidence is sufficiently high
        high_confidence_mask = model_confidences >= self.confidence_threshold

        # 2. Check against external verification oracle (ground-truth consistency)
        oracle_pass_mask = (proposed_labels == self.ground_truth)

        # 3. Only candidates passing BOTH become verified training samples
        verified_mask = high_confidence_mask & oracle_pass_mask

        n_total = len(proposed_labels)
        n_high_conf = np.sum(high_confidence_mask)
        n_verified = np.sum(verified_mask)
        n_rejected_hallucinations = np.sum(high_confidence_mask & (~oracle_pass_mask))

        return {
            "verified_mask": verified_mask,
            "n_total": n_total,
            "n_high_conf": n_high_conf,
            "n_verified": n_verified,
            "n_rejected_hallucinations": n_rejected_hallucinations
        }
