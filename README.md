# Credit Card Fraud Detection: Production MLOps

[![CI](https://github.com/priyankabolem/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/priyankabolem/fraud-detection-mlops/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)]()

Real-time fraud scoring on the ULB credit card dataset (284,807 transactions, 492 frauds, 1:578 imbalance),
built and evaluated the way a production system is: chronological split, AUPRC-driven model selection, an
operating threshold chosen for a precision budget, experiment tracking, a serving API with the exact artifact
that was evaluated, input drift monitoring, tests and CI.

## Results

Chronological 70/15/15 split: the model trains on the earliest transactions and is evaluated on the latest
42,722 (never seen, later in time). Threshold chosen on the validation window for precision >= 0.90,
then frozen before touching the test set.

| Model | AUPRC | ROC-AUC | Precision @ threshold | Recall @ threshold |
|---|---|---|---|---|
| Logistic regression (balanced) | 0.718 | 0.978 | 0.94 | 0.63 |
| **XGBoost (champion)** | **0.765** | **0.979** | **0.76** | **0.75** |

At the chosen threshold the champion flags 51 of 42,722 test transactions and catches
39 of 52 frauds with 12 false alarms. With 0.17% fraud prevalence, accuracy would be 99.8%
for a model that flags nothing; AUPRC and the precision-recall operating point are the metrics that matter.

Two honest observations the random-split versions of this dataset hide: precision at the frozen threshold
degrades from the 0.90 validation target to 0.76 on the later test window, and the PSI drift monitor flags
exactly why: the feature distributions shift between the two days of data (PSI 1.57 on V1, 1.43 on V3,
day 1 vs day 2). The monitoring catches the same phenomenon that moved the operating point.

## What is here

- **`fraud_detection.data`**: dataset loading (Kaggle CSV or an automatic one-time OpenML fetch) and the
  chronological split. Random splits leak future fraud patterns into training; this one never does.
- **`fraud_detection.train`**: baseline + XGBoost with class weighting and early stopping on validation
  AUPRC, threshold selection for a precision budget, both runs tracked in MLflow, artifact + metadata saved
  with the data checksum.
- **`fraud_detection.serve`**: FastAPI service, `POST /predict` scores up to 1,000 transactions per call and
  reports per-request latency; `GET /health` reports the model version. Serves exactly the evaluated artifact.
- **`fraud_detection.drift`**: Population Stability Index monitoring implemented directly (quantile bins,
  standard 0.10 / 0.25 bands), comparing live inputs against the training window. No heavy dependencies.
- **`tests/`**: 10 tests covering metrics, threshold selection, drift detection, feature preparation and the
  API (on a tiny synthetic model, so CI never needs the 144 MB dataset). GitHub Actions runs lint + tests on
  every push.

## Run it

```bash
pip install -e ".[dev]"

python -m fraud_detection.train        # fetches the dataset on first run, trains, writes models/
python -m fraud_detection.drift        # PSI report: last day of data vs first day
uvicorn fraud_detection.serve:app --port 8000

curl -s localhost:8000/health
# score a transaction (V1..V28 from the dataset, Time in seconds, Amount in dollars)
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @docs/example_request.json
```

Docker:

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

MLflow UI over past runs: `mlflow ui` (runs are tracked in `./mlruns`).

## Design decisions

1. **Chronological split, not random.** Fraud models score tomorrow's transactions, so evaluation must too.
   AUPRC here is lower than the numbers reported on random splits of this dataset; that is the honest number.
2. **The threshold is a business decision.** Review capacity sets a precision budget; the model commits to an
   operating point on validation data and is judged there on test data.
3. **The served artifact is the evaluated artifact.** `models/model.json` + `models/metadata.json` (threshold,
   metrics, feature list, data checksum) travel together into the Docker image.
4. **Drift first, retraining second.** PSI against the training window is the cheap early warning that the
   input distribution moved; the metadata records what "the training window" was.

## Honest limitations

- Features V1 to V28 are PCA components from the dataset authors; no raw feature engineering is possible.
- Single dataset, two days of transactions: drift monitoring is demonstrated on a day-vs-day comparison, not
  months of production traffic.
- No orchestration layer or cloud deployment in this repository; the pieces are containerized and stateless
  so either is an infrastructure step, not a modeling step.

## Author

**Priyanka Bolem**, Machine Learning Engineer
[Portfolio](https://priyankabolem.github.io) · [LinkedIn](https://www.linkedin.com/in/priyanka-bolem) · [GitHub](https://github.com/priyankabolem)
