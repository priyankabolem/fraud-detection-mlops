import numpy as np
import pandas as pd

from fraud_detection.drift import drift_report, psi


def test_identical_distributions_are_stable():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.05


def test_shifted_distribution_raises_alert():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000), "y": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 5000), "y": rng.normal(0, 1, 5000)})
    report = drift_report(ref, cur)
    assert report["status"] == "alert" and report["worst_feature"] == "x"


def test_constant_feature_does_not_break():
    assert psi(np.ones(100), np.ones(100)) == 0.0
