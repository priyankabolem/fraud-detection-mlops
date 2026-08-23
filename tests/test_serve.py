"""API test on a tiny model trained on synthetic data, so CI never needs the 144 MB dataset."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from fraud_detection import serve
from fraud_detection.features import prepare


def _tiny_model(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    cols = {"Time": rng.uniform(0, 172800, n), "Amount": rng.exponential(80, n)}
    cols.update({f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)})
    frame = pd.DataFrame(cols)
    y = (frame["V1"] > 1).astype(int)
    model = XGBClassifier(n_estimators=10, max_depth=2)
    model.fit(prepare(frame), y)
    (tmp_path / "model.json").parent.mkdir(exist_ok=True)
    model.save_model(tmp_path / "model.json")
    (tmp_path / "metadata.json").write_text('{"trained_at": "test", "threshold": 0.5, "features": []}')


def test_predict_roundtrip(monkeypatch, tmp_path):
    _tiny_model(tmp_path)
    monkeypatch.setattr(serve, "MODELS_DIR", tmp_path)
    serve._load()
    client = TestClient(serve.app)
    assert client.get("/health").json()["status"] == "ok"
    txn = {"Time": 1000, "Amount": 25.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    r = client.post("/predict", json={"transactions": [txn, {**txn, "V1": 3.0}]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert 0 <= body["results"][0]["fraud_score"] <= 1
    assert body["results"][1]["fraud_score"] > body["results"][0]["fraud_score"]


def test_predict_rejects_missing_features(monkeypatch, tmp_path):
    _tiny_model(tmp_path)
    monkeypatch.setattr(serve, "MODELS_DIR", tmp_path)
    serve._load()
    client = TestClient(serve.app)
    r = client.post("/predict", json={"transactions": [{"Time": 0, "Amount": 5.0}]})
    assert r.status_code == 422
