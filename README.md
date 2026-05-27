# Credit Card Fraud Detection — Production MLOps System

[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in_development-yellow.svg)]()

> End-to-end machine learning system for detecting fraudulent credit card transactions in real-time, built with a complete MLOps stack including monitoring, drift detection, and LLM-powered explainability.

## 🎯 Problem Statement

Credit card fraud causes billions in annual losses globally. Detection systems must catch fraud in real-time (sub-100ms latency) while minimizing false alarms that frustrate legitimate customers. This project builds a production-grade fraud detection system from data ingestion through deployment, monitoring, and automated retraining.

**Business Goal:** Detect fraud with high recall (catch most fraud) while maintaining acceptable precision (don't annoy legitimate users).

## 📊 Dataset

Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **284,807 transactions** from European cardholders (September 2013)
- **492 fraud cases** (0.17% — severely imbalanced)
- **31 features:** Time, V1-V28 (PCA-anonymized), Amount, Class
- **48-hour time span**

**Key Insight:** Imbalance ratio of 1:578 makes accuracy a misleading metric. We use AUPRC, F1-score, and Precision-Recall curves for evaluation.

## 🛠️ Tech Stack

**Core ML**
- Python 3.14, Pandas, NumPy
- scikit-learn, XGBoost
- Jupyter for exploration

**MLOps**
- MLflow — experiment tracking and model registry
- Apache Airflow — pipeline orchestration
- Evidently AI — drift monitoring
- GitHub Actions — CI/CD

**Serving & Deployment**
- FastAPI — model serving API
- Docker — containerization
- AWS (ECR, ECS Fargate, S3, CloudWatch) — cloud deployment

**LLM Layer (Final Phase)**
- Anthropic Claude API — fraud explanation
- Vector database — RAG over fraud pattern documentation

## 📁 Project Structure
fraud-detection-mlops/
├── data/
│   ├── raw/              # Original dataset (gitignored)
│   └── processed/        # Cleaned/feature-engineered data
├── docs/                 # Documentation and daily log
├── models/               # Trained model artifacts (gitignored)
├── notebooks/            # Jupyter notebooks for exploration
│   └── 01_initial_exploration.ipynb
├── src/
│   └── fraud_detection/  # Production Python package
├── tests/                # Unit and integration tests
├── pyproject.toml        # Project configuration and dependencies
├── uv.lock              # Locked dependency versions
└── README.md            # You are here

## 🚀 Getting Started

### Prerequisites

- macOS / Linux
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Homebrew (for `libomp` on macOS)

### Setup

```bash
# Clone the repo
git clone https://github.com/priyankabolem/fraud-detection-mlops.git
cd fraud-detection-mlops

# Install dependencies (uv will auto-install Python 3.14 if needed)
uv sync --extra dev

# Activate the environment
source .venv/bin/activate

# Install libomp for XGBoost (Mac only)
brew install libomp

# Download the dataset from Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in data/raw/
```

### Run the exploration notebook

```bash
cd notebooks
jupyter notebook 01_initial_exploration.ipynb
```

## 📈 Project Roadmap

This project is being built over a 10-week sprint (May 26 — August 3, 2026):

- [x] **Week 1:** Engineering foundations + EDA
- [ ] **Week 2:** Problem framing + baseline model
- [ ] **Week 3:** Model training with MLflow tracking
- [ ] **Week 4:** FastAPI serving layer
- [ ] **Week 5:** Docker + AWS deployment
- [ ] **Week 6:** CI/CD + observability
- [ ] **Week 7:** Airflow training pipelines
- [ ] **Week 8:** Evidently AI drift monitoring
- [ ] **Week 9:** Portfolio polish + architecture diagrams
- [ ] **Week 10:** LLM-powered explainability (RAG layer)

See [`docs/daily-log.md`](docs/daily-log.md) for daily progress notes.

## 📊 Current Status

🚧 **In active development** — Currently in Week 1: Foundations + Data Exploration

**Completed:**
- ✅ Development environment setup (Python 3.14, uv, VS Code)
- ✅ Project structure following industry conventions
- ✅ Initial EDA: class imbalance analysis, feature exploration
- ✅ Documented dataset characteristics and engineering implications

**In Progress:**
- 🔄 Deep exploratory data analysis with visualizations
- 🔄 Formal problem statement document

## 👤 Author

**Priyanka Bolem**  
Machine Learning Engineer | NLP • LLMs • MLOps  
🔗 [LinkedIn](https://www.linkedin.com/in/priyanka-bolem) • [GitHub](https://github.com/priyankabolem)

---

⭐ **This project is part of a focused 10-week ML Engineer capstone sprint.** Star the repo to follow along.