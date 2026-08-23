"""Evaluation for a 1:578 imbalanced problem: AUPRC first, plus the operating point the business runs at."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pick_threshold(y_true: np.ndarray, scores: np.ndarray, min_precision: float = 0.90) -> float:
    """Highest-recall threshold whose precision is at least min_precision; falls back to the max-F1 point.

    Fraud teams review flagged transactions by hand, so the operating point is set by the precision they can
    afford, not by the default 0.5.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    best = None
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and (best is None or r > best[0]):
            best = (r, t)
    if best is not None:
        return float(best[1])
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Threshold-free metrics plus the confusion counts at the chosen operating point."""
    flagged = scores >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auprc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "precision_at_threshold": round(precision, 4),
        "recall_at_threshold": round(recall, 4),
        "flagged": int(flagged.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "positives": int((y_true == 1).sum()),
        "n": int(len(y_true)),
    }
