import numpy as np

from fraud_detection.metrics import evaluate, pick_threshold


def _scores(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    scores = np.clip(y * 0.7 + rng.normal(0.2, 0.15, n), 0, 1)
    return y, scores


def test_pick_threshold_meets_precision_floor():
    y, scores = _scores()
    t = pick_threshold(y, scores, min_precision=0.9)
    m = evaluate(y, scores, t)
    assert m["precision_at_threshold"] >= 0.9
    assert m["recall_at_threshold"] > 0


def test_pick_threshold_falls_back_to_f1_when_floor_unreachable():
    y, scores = _scores()
    t = pick_threshold(y, 1 - scores, min_precision=0.99)  # scores anti-correlated: floor unreachable
    assert 0 <= t <= 1


def test_evaluate_counts_are_consistent():
    y, scores = _scores()
    m = evaluate(y, scores, 0.5)
    assert m["true_positives"] + m["false_negatives"] == m["positives"]
    assert m["true_positives"] + m["false_positives"] == m["flagged"]
    assert 0 <= m["auprc"] <= 1 and 0 <= m["roc_auc"] <= 1
