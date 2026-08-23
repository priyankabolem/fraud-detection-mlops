import pandas as pd

from fraud_detection.features import MODEL_FEATURES, prepare


def _frame(n=3):
    row = {"Time": 3600.0 * 26, "Amount": 100.0, **{f"V{i}": 0.1 for i in range(1, 29)}}
    return pd.DataFrame([row] * n)


def test_prepare_returns_fixed_columns_in_order():
    out = prepare(_frame())
    assert list(out.columns) == MODEL_FEATURES
    assert "Time" not in out.columns


def test_engineered_values():
    out = prepare(_frame())
    assert out["hour"].iloc[0] == 2  # 26 hours in seconds wraps to hour 2
    assert abs(out["log_amount"].iloc[0] - 4.6151) < 1e-3
