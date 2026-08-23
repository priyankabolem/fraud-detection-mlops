"""Dataset loading.

The ULB credit card fraud dataset (284,807 transactions, 492 frauds, PCA-anonymized features V1 to V28 plus
Time and Amount). Loaded from data/raw/creditcard.csv when present (the Kaggle download), otherwise fetched
once from OpenML (dataset id 42175, the same data with all 31 columns) and cached to that path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "creditcard.csv"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Return the full dataset ordered by Time, with an integer Class column."""
    path = Path(path) if path else RAW_PATH
    if not path.exists():
        _fetch_from_openml(path)
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(str).str.strip("'").astype(float).astype(int)
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def _fetch_from_openml(path: Path) -> None:
    from sklearn.datasets import fetch_openml

    print("downloading the dataset from OpenML (one time, ~144 MB) ...")
    bunch = fetch_openml(data_id=42175, as_frame=True, parser="auto")
    df = bunch.frame
    df = df.rename(columns={"class": TARGET})
    if TARGET in df.columns and df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].astype(str).str.strip("'")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def time_split(df: pd.DataFrame, train: float = 0.7, val: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: train on the earliest transactions, validate and test on later ones.

    A random split would leak future fraud patterns into training; fraud detection is deployed forward in time.
    """
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
