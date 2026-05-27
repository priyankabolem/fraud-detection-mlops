# Daily Log — Credit Card Fraud Detection MLOps Project

10-week ML Engineer capstone sprint  
Start: Tuesday, May 26, 2026 | Target finish: Sunday, August 3, 2026

---

## Day 1 — Tuesday, May 26, 2026

### What I built today
- Verified core tooling on Mac (Python 3.12.7, Git 2.50.1, Homebrew 4.6.7)
- Disabled Conda auto-activation permanently (clean Python env for project work)
- Installed VS Code with 13 extensions (Python, Pylance, GitLens, Container Tools, Even Better TOML, Jupyter suite)
- Installed `uv` package manager (modern Python tooling)
- Created project folder structure with proper layout
- Initialized Git locally
- Configured comprehensive `.gitignore` for ML projects

### What I learned
- `uv` is way faster than pip and now industry-standard
- Mac's `code .` command needs one-time shell install (Cmd+Shift+P → "shell command install")
- Conda and uv shouldn't fight each other — disable Conda auto-activation per project

### What confused me / questions
- Difference between Conda environments vs uv environments — when to use which?

### Tomorrow's focus
- Fill `pyproject.toml`, install dependencies, download dataset

---

## Day 2 — Wednesday, May 27, 2026

### What I built today
- Configured `pyproject.toml` with 50+ ML dependencies (pandas, scikit-learn, xgboost, jupyter, pytest, ruff)
- Ran `uv sync --extra dev` — uv auto-installed Python 3.14.5 and all packages cleanly
- Fixed XGBoost OpenMP error on Mac via `brew install libomp`
- Downloaded Credit Card Fraud Detection dataset from Kaggle (~144MB, 284,807 transactions)
- Verified `.gitignore` protects the 144MB CSV from Git
- Created `01_initial_exploration.ipynb` with 7 analysis cells + markdown observations
- Confirmed class imbalance: 492 fraud vs 284,315 normal (0.17%)
- Made first professional Git commit (10 files, 3725 insertions)
- Pushed project live to GitHub: github.com/priyankabolem/fraud-detection-mlops

### What I learned
- XGBoost on Mac needs `libomp` (OpenMP runtime) — common first-time error
- VS Code auto-activates `.venv` when it detects `pyproject.toml` — saves manual activation
- Class imbalance ratio 1:578 makes accuracy useless as a metric — must use AUPRC/F1/PR-curve
- Fraud has HIGHER mean ($122) but LOWER median ($9) vs normal — bimodal "card testing" pattern
- Fraud max amount ($2,125) is much lower than normal max ($25,691) — fraudsters avoid huge amounts
- Dataset spans only 48 hours — limits temporal pattern learning

### What confused me / questions
- How does git diff handle `.ipynb` (JSON format) files? Will future notebook changes be reviewable?
- Should we be worried about Python 3.14 being too new for some ML libraries? Pandas 3.0.3 worked fine but unsure about others
- When we move to deployment, will the 2013 dataset age create drift in the model evaluation step?

### Tomorrow's focus (Day 3)
- Deep EDA with visualizations (matplotlib, seaborn)
- Plot class distribution, feature distributions, Amount distribution by class
- Correlation analysis of V1-V28 features with the Class label
- Time-based analysis: are there hourly patterns in fraud?
- Write formal problem statement document in docs/