"""Feature preparation.

V1 to V28 are already PCA components; the engineered additions are the two raw columns made model-friendly:
log-scaled amount (spend is heavy tailed) and the hour of day cycled from the Time offset (fraud rate varies
by hour in this dataset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES

MODEL_FEATURES = [f for f in FEATURES if f != "Time"] + ["log_amount", "hour"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input frame in a fixed column order."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"].clip(lower=0))
    out["hour"] = (out["Time"] // 3600) % 24
    return out[MODEL_FEATURES]
