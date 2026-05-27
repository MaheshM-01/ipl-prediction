# IPL Match Outcome Prediction

Professional, reproducible pipeline for predicting Indian Premier League (IPL) match outcomes.

Table of contents
- Project overview
- Quickstart
- Project structure
- Prerequisites
- Installation
- Configuration
- Data layout
- Running (train / predict / API)
- MLflow & artifacts
- Docker
- Testing & linting
- Contributing
- Contact

Project overview
This repository contains a machine-learning pipeline used to train and serve models that predict IPL match outcomes. It includes data preprocessing, model training, evaluation, a simple API, and utilities for experiments tracked with MLflow.

Quickstart
1. Create and activate a Python virtual environment (Windows):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Or on macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure the project (see Configuration section) and run a quick prediction (example):

```bash
python src/pipeline/predict_pipeline.py
```

Project structure (major files/folders)
- `config/` — configuration loader and `config.yaml`.
- `data/raw/` — raw CSV data (source deliveries.csv, matches.csv).
- `data/processed/` — cleaned and processed datasets used for training.
- `src/` — project source code: pipelines, features, model code, API, monitoring.
- `notebooks/` — exploratory notebooks and experiments.
- `mlartifacts/`, `mlruns/` — MLflow artifacts and run history.
- `start_services.sh` — helper script to start local services (where applicable).

Prerequisites
- Python 3.9+ recommended (adjust for your environment).
- Git for version control.
- Optional: Docker & Docker Compose for containerized runs.

Installation
1. Clone the repository:

```bash
git clone <repo-url>
cd ipl-prediction
```

2. Create and activate a virtual environment and install dependencies (see Quickstart above).

Configuration
The project uses `config/config.yaml` for runtime configuration. Edit values there or set environment-specific overrides in `config/config.py`.

Example config keys you may need to adjust:
- data paths (raw / processed)
- MLflow tracking URI
- model hyperparameters

Data
- Raw source files are in `data/raw/`.
- Cleaned and merged datasets are in `data/processed/` (already included in this repo for convenience).

Running the pipeline
- Train a model (example):

```bash
python -m src.pipeline.train_pipeline
```

- Run a prediction / inference pipeline (example):

```bash
python -m src.pipeline.predict_pipeline --input data/processed/merged_clean.csv --output predictions.csv
```

- Start the API (FastAPI / Uvicorn):

```bash
pip install uvicorn
uvicorn src.api.main_api:app --reload --host 0.0.0.0 --port 8000
```

MLflow & experiment tracking
- Run the MLflow UI to inspect runs and artifacts:

```bash
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
```

- Trained models and artifacts are stored under `mlartifacts/` and `mlruns/`.

Docker
- A `dockerfile` exists at the repo root for containerizing the app. Example build/run:

```bash
docker build -t ipl-prediction:latest .
docker run -p 8000:8000 ipl-prediction:latest
```

Testing & linting
- Tests live in the `test/` folder. Run tests with:

```bash
pytest -q
```

- Code style: use `black` and `isort` to format imports. Example:

```bash
black .
isort .
```

Contributing
See `docs/CONTRIBUTING.md` for contribution guidelines.

Contact
- For questions or help, open an issue or contact the maintainer listed in the repository metadata.

Notes
- This README provides an operational overview. For detailed usage examples and developer guidelines, see the `docs/` folder.
