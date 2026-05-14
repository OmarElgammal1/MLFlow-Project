# Bank Consumer Churn Prediction

A Random Forest classifier for bank customer churn, served through a Litestar HTTP API with thorough logging shipped live to [HyperDX](https://www.hyperdx.io/) over OTLP. MLflow is used for experiment tracking during training; `uv` manages the Python environment.

## Project Structure

- [dataset/](dataset/) — `Churn_Modelling.csv` source data.
- [src/](src/)
  - [preprocessing.py](src/preprocessing.py) — load, rebalance, transform features.
  - [model.py](src/model.py) — Random Forest training and evaluation, with MLflow logging.
  - [train.py](src/train.py) — training entry point. Dumps `model.pkl` at repo root.
  - [src/api/](src/api/) — Litestar app, schemas, predictor, logging setup.
- [tests/](tests/) — pytest suite covering predictor (function tests) and API (endpoint tests).
- [pyproject.toml](pyproject.toml) — uv project and pytest configuration.
- [uv.lock](uv.lock) — committed lockfile for reproducible installs.
- `.env.example` — template for HyperDX env variables.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.10+
- Python 3.10–3.12 (uv will fetch one if needed)

## Setup

```bash
uv sync
```

Copy the example env file and fill in your HyperDX ingest key (optional — without it, logs go to stdout only):

```bash
cp .env.example .env
# edit .env: set HYPERDX_API_KEY=...
```

## Train the model

The training script logs metrics, params, and artifacts to MLflow at `http://localhost:5000`, and writes `model.pkl` next to `transformer.pkl` at the repo root.

```bash
# 1. Start MLflow (fresh sqlite store)
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# 2. In a second terminal, run training
uv run python src/train.py
```

After it finishes you should see `model.pkl` and `confusion_matrix.png` at the repo root, plus a new run in the MLflow UI.

## Run the API

```bash
uv run litestar --app src.api.app:app run --port 8000
```

Endpoints:

| Method | Path       | Purpose                                       |
| ------ | ---------- | --------------------------------------------- |
| GET    | `/`        | Service metadata (name, version, docs link).  |
| GET    | `/health`  | Liveness + whether `model.pkl` is loaded.     |
| POST   | `/predict` | Single-record churn prediction.               |

OpenAPI / Swagger UI is auto-mounted by Litestar at `/schema/swagger`.

Example prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "content-type: application/json" \
  -d '{
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
# -> {"prediction":1,"probability":0.87}
```

## Live logs with HyperDX

[src/api/logging_setup.py](src/api/logging_setup.py) wires the Python root logger to an OTLP exporter pointed at HyperDX. Configure via `.env`:

```
HYPERDX_API_KEY=your-hyperdx-ingest-key
OTEL_SERVICE_NAME=churn-api
OTEL_EXPORTER_OTLP_ENDPOINT=https://in-otel.hyperdx.io
```

When the key is unset, only stdout logging is configured (useful for tests and offline development).

What gets logged on every request:

- request entry (path, client IP) — at the route handler.
- prediction event with feature hash, prediction, probability, latency in ms — at `predict()`.
- artifact load / startup events.
- unhandled exceptions with stack traces.

## Testing

```bash
uv run pytest
```

The suite enforces ≥70% coverage on `src/api` (`--cov-fail-under=70`, configured in [pyproject.toml](pyproject.toml)):

- [tests/test_predictor.py](tests/test_predictor.py) — function tests for `predictor.predict`, feature hashing, and artifact-missing error paths.
- [tests/test_api.py](tests/test_api.py) — endpoint tests for `/`, `/health`, `/predict` (happy path + two validation failures), using Litestar's `TestClient`.

## Model details

- **Algorithm**: Random Forest Classifier (`n_estimators=100`, `random_state=42`).
- **Features**: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
- **Target**: `Exited` (0 = retained, 1 = churned).
- **Preprocessing**: a pre-fitted `ColumnTransformer` (StandardScaler + OneHotEncoder) is loaded from `transformer.pkl`.
