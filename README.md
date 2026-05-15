# Bank Consumer Churn Prediction API

A Litestar HTTP service that serves a pre-trained Random Forest classifier for bank customer churn, with thorough logging shipped live to [HyperDX](https://www.hyperdx.io/) over OTLP. `uv` manages the Python environment.

The model and preprocessing transformer are loaded from `model.pkl` and `transformer.pkl` at the repo root — training is intentionally out of scope for this project.

## Project Structure

- [src/](src/)
  - [app.py](src/app.py) — Litestar app and route handlers.
  - [schemas.py](src/schemas.py) — Pydantic request/response models.
  - [predictor.py](src/predictor.py) — artifact loading and the single-record `predict()` function.
  - [logging_setup.py](src/logging_setup.py) — stdout + HyperDX OTLP wiring.
- [tests/](tests/) — pytest suite (function tests + endpoint tests).
- [pyproject.toml](pyproject.toml) — uv project + pytest configuration.
- [uv.lock](uv.lock) — committed lockfile for reproducible installs.
- `.env.example` — template for HyperDX env variables.
- `model.pkl`, `transformer.pkl` — prediction artifacts loaded at startup.

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

## Run the API

```bash
uv run litestar --app src.app:app run --port 8000
```

### Run with Docker

Build the image once:

```bash
docker build -t churn-api:latest .
```

Run the container, mapping host port 8000 to container port 8000 and loading the local `.env` (HyperDX key + OTEL settings):

```bash
docker run -d --name churn-api --env-file .env -p 8000:8000 churn-api:latest
```

Without HyperDX (stdout logs only):

```bash
docker run -d --name churn-api -p 8000:8000 churn-api:latest
```

Helpful follow-ups:

```bash
docker logs -f churn-api      # tail logs
docker ps                     # confirm it is running
docker rm -f churn-api        # stop and remove
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

[src/logging_setup.py](src/logging_setup.py) wires the Python root logger to an OTLP exporter pointed at HyperDX. Configure via `.env`:

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

The suite enforces ≥70% coverage on `src` (`--cov-fail-under=70`, configured in [pyproject.toml](pyproject.toml)):

- [tests/test_predictor.py](tests/test_predictor.py) — function tests for `predictor.predict`, feature hashing, and artifact-missing error paths.
- [tests/test_api.py](tests/test_api.py) — endpoint tests for `/`, `/health`, `/predict` (happy path + two validation failures), using Litestar's `TestClient`.

## Load testing

A Locust swarm in [loadtest/](loadtest/) exercises the deployed (or local) API. Locust is in its own dep group so it never leaks into the runtime image or the test job.

```bash
uv sync --group load
LOCUST_HOST=http://127.0.0.1:8000 uv run locust -f loadtest/locustfile.py
# open http://localhost:8089, set Users + Spawn rate, click "Start swarming"
```

Point `LOCUST_HOST` at your EC2 deployment to run the same swarm against production. See [loadtest/README.md](loadtest/README.md) for the experiment matrix and which charts to read.

## Model details

- **Algorithm**: Random Forest Classifier.
- **Features**: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
- **Target**: `Exited` (0 = retained, 1 = churned).
- **Preprocessing**: a pre-fitted `ColumnTransformer` (StandardScaler + OneHotEncoder) loaded from `transformer.pkl`.
