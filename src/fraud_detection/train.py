"""Train the fraud model: chronological split, logistic regression baseline, XGBoost champion.

Every run is tracked in MLflow (./mlruns). The champion, its operating threshold and its test metrics are
written to models/ so the API and the Docker image serve exactly what was evaluated.

    python -m fraud_detection.train [--data data/raw/creditcard.csv] [--min-precision 0.9]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import TARGET, load_dataset, time_split
from .features import MODEL_FEATURES, prepare
from .metrics import evaluate, pick_threshold

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.06,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    eval_metric="aucpr",
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--min-precision", type=float, default=0.90)
    args = ap.parse_args()

    df = load_dataset(args.data)
    train_df, val_df, test_df = time_split(df)
    x_train, y_train = prepare(train_df), train_df[TARGET].to_numpy()
    x_val, y_val = prepare(val_df), val_df[TARGET].to_numpy()
    x_test, y_test = prepare(test_df), test_df[TARGET].to_numpy()
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"rows train/val/test: {len(x_train)}/{len(x_val)}/{len(x_test)}; positive weight {pos_weight:.0f}")

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="baseline-logistic"):
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        baseline.fit(x_train, y_train)
        scores = baseline.predict_proba(x_test)[:, 1]
        base_metrics = evaluate(y_test, scores, pick_threshold(y_val, baseline.predict_proba(x_val)[:, 1], args.min_precision))
        mlflow.log_metrics({k: v for k, v in base_metrics.items() if isinstance(v, float)})
        print("baseline  ", json.dumps(base_metrics))

    with mlflow.start_run(run_name="xgboost"):
        model = XGBClassifier(scale_pos_weight=pos_weight, **XGB_PARAMS)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        mlflow.log_params({**XGB_PARAMS, "scale_pos_weight": round(pos_weight, 1), "min_precision": args.min_precision})
        threshold = pick_threshold(y_val, model.predict_proba(x_val)[:, 1], args.min_precision)
        test_metrics = evaluate(y_test, model.predict_proba(x_test)[:, 1], threshold)
        mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, float)})
        print("xgboost   ", json.dumps(test_metrics))

        MODELS_DIR.mkdir(exist_ok=True)
        model.save_model(MODELS_DIR / "model.json")
        data_path = Path(args.data) if args.data else Path(__file__).resolve().parents[2] / "data/raw/creditcard.csv"
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()[:12]
        metadata = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "features": MODEL_FEATURES,
            "threshold": test_metrics["threshold"],
            "test_metrics": test_metrics,
            "baseline_test_metrics": base_metrics,
            "data_sha256_12": digest,
            "split": "chronological 70/15/15",
        }
        (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=1))
        mlflow.log_artifact(str(MODELS_DIR / "metadata.json"))
        print(f"saved models/model.json and metadata; threshold {threshold:.4f}")


if __name__ == "__main__":
    main()
