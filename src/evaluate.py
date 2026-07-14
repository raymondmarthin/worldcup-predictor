"""evaluate.py - probability-quality metrics for the knockout predictor."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score


def evaluate_probs(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }
