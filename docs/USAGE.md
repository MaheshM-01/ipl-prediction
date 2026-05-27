Usage Guide

This document provides examples for common developer and user workflows.

Environment setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Training

Run the training pipeline (adjust paths/flags as needed):

```bash
python -m src.pipeline.train_pipeline
```

The pipeline will read configuration from `config/config.yaml` and write artifacts to `mlartifacts/` and `mlruns/`.

Inference / Predict

Run the predict pipeline to produce a CSV of predictions:

```bash
python -m src.pipeline.predict_pipeline --input data/processed/merged_clean.csv --output predictions.csv
```

API

Start the FastAPI app locally (requires `uvicorn`):

```bash
pip install uvicorn
uvicorn src.api.main_api:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to interact with the API (Swagger UI).

MLflow

Start MLflow UI to inspect experiments:

```bash
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
```

Docker

Build and run the container (example):

```bash
docker build -t ipl-prediction:latest .
docker run -p 8000:8000 ipl-prediction:latest
```

Debugging & Logs

- Logs are written to the `logs/` folder; use those for troubleshooting.
- Monitoring helpers live in `src/monitoring`.

If something fails

- Check configuration in `config/config.yaml`.
- Validate data files in `data/processed/`.
- Re-run tests with `pytest -q`.

Contact

Open an issue for help or clarification.
