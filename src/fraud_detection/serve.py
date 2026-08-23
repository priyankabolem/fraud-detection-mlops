"""FastAPI scoring service for the trained model.

    uvicorn fraud_detection.serve:app --host 0.0.0.0 --port 8000

POST /predict takes a list of transactions (Time, V1 to V28, Amount) and returns a fraud score per
transaction plus whether it crosses the operating threshold chosen at training time.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from .features import prepare

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_state: dict = {}


def _load() -> None:
    model_path, meta_path = MODELS_DIR / "model.json", MODELS_DIR / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise RuntimeError("no trained model in models/; run: python -m fraud_detection.train")
    model = XGBClassifier()
    model.load_model(model_path)
    _state["model"] = model
    _state["meta"] = json.loads(meta_path.read_text())


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(title="fraud-detection", version="1.0", lifespan=lifespan)


class Transaction(BaseModel):
    Time: float = 0.0
    Amount: float = Field(ge=0)
    # V1 to V28 arrive as extra fields, validated below
    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict:
    meta = _state.get("meta", {})
    return {"status": "ok", "trained_at": meta.get("trained_at"), "threshold": meta.get("threshold")}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    started = time.perf_counter()
    rows = [t.model_dump() for t in req.transactions]
    missing = [f"V{i}" for i in range(1, 29) if any(f"V{i}" not in r for r in rows)]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {missing[:5]}")
    frame = prepare(pd.DataFrame(rows))
    scores = _state["model"].predict_proba(frame)[:, 1]
    threshold = _state["meta"]["threshold"]
    return {
        "results": [
            {"fraud_score": round(float(s), 6), "flagged": bool(s >= threshold)} for s in scores
        ],
        "threshold": threshold,
        "model_trained_at": _state["meta"]["trained_at"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
