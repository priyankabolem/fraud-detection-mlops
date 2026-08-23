"""Input drift monitoring with the Population Stability Index.

PSI compares the distribution a model was trained on with what it sees in production. The usual reading:
below 0.10 stable, 0.10 to 0.25 moderate shift worth watching, above 0.25 significant shift, retrain or
investigate. Implemented directly (quantile bins on the reference window) so the check has no heavy
dependencies and runs anywhere, including CI.

    python -m fraud_detection.drift   # compares the last day of the dataset against the first day
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

PSI_ALERT = 0.25
PSI_WATCH = 0.10


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near) constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    cur_frac = np.clip(np.histogram(current, edges)[0] / len(current), 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """PSI per shared numeric column plus an overall status."""
    values = {c: round(psi(reference[c].to_numpy(), current[c].to_numpy()), 4) for c in reference.columns if c in current}
    worst = max(values, key=values.get) if values else ""
    status = "alert" if values and values[worst] >= PSI_ALERT else "watch" if values and values[worst] >= PSI_WATCH else "stable"
    return {"status": status, "worst_feature": worst, "psi": dict(sorted(values.items(), key=lambda kv: -kv[1]))}


def main() -> None:
    from .data import load_dataset
    from .features import prepare

    df = load_dataset()
    day = 24 * 3600
    report = drift_report(prepare(df[df["Time"] < day]), prepare(df[df["Time"] >= df["Time"].max() - day]))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
