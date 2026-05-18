# Bank Consumer Churn Prediction API

A Litestar HTTP service that serves a pre-trained Random Forest classifier for bank customer churn, with structured events shipped live to [Axiom](https://axiom.co/) for dashboards and alerting. `uv` manages the Python environment.

The model and preprocessing transformer are loaded from `model.pkl` and `transformer.pkl` at the repo root — training is intentionally out of scope for this project.

## Project Structure

- [src/](src/)
  - [app.py](src/app.py) — Litestar app and route handlers.
  - [schemas.py](src/schemas.py) — Pydantic request/response models.
  - [predictor.py](src/predictor.py) — artifact loading and the single-record `predict()` function.
  - [logging_setup.py](src/logging_setup.py) — stdout + Axiom shipping handler.
- [tests/](tests/) — pytest suite (function tests + endpoint tests).
- [pyproject.toml](pyproject.toml) — uv project + pytest configuration.
- [uv.lock](uv.lock) — committed lockfile for reproducible installs.
- `.env.example` — template for Axiom env variables.
- `model.pkl`, `transformer.pkl` — prediction artifacts loaded at startup.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.10+
- Python 3.10–3.12 (uv will fetch one if needed)

## Setup

```bash
uv sync
```

Copy the example env file and fill in your Axiom ingest token + dataset (optional — without them, logs go to stdout only):

```bash
cp .env.example .env
# edit .env: set AXIOM_TOKEN=... and AXIOM_DATASET=churn-api
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

Run the container, mapping host port 8000 to container port 8000 and loading the local `.env` (Axiom token + dataset):

```bash
docker run -d --name churn-api --env-file .env -p 8000:8000 churn-api:latest
```

Without Axiom (stdout logs only):

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

## Live observability with Axiom

[src/logging_setup.py](src/logging_setup.py) wires the Python root logger to a custom `AxiomBufferedHandler` that batches each `LogRecord` and ships it to an Axiom dataset via the [axiom-py](https://github.com/axiomhq/axiom-py) SDK. Configure via `.env`:

```
AXIOM_TOKEN=your-axiom-ingest-token
AXIOM_DATASET=churn-api
```

When either is unset, only stdout logging is configured (useful for tests and offline development).

### Event streams emitted

| Event | Source | Top-level fields |
| --- | --- | --- |
| `http_request` | [HttpEventMiddleware](src/app.py) on every request | `method`, `path`, `status_code`, `duration_ms`, `client_ip` |
| `predict_request` | [predict_endpoint](src/app.py) before inference | all 10 features (`credit_score`, `geography`, `age`, `balance`, …) + `client` |
| `prediction served` | [predictor.predict](src/predictor.py) after inference | `prediction` (0/1), `probability` (0-1), `latency_ms`, `feature_hash` |

`extra={...}` keys are flattened to top-level event fields so APL queries can chart them directly — no nested `attributes.*` paths.

### Dashboard metrics (5 across the 3 required categories)

**Model**
1. **Prediction probability distribution** — `summarize histogram(probability, 20) by bin(_time, 1m)` where `message == "prediction served"`. *Why:* detects model confidence drift; a healthy model is bimodal, a degraded one collapses toward 0.5.
2. **Churn rate** — `summarize avg(prediction) by bin(_time, 1m)`. *Why:* class-balance drift is the cheapest early warning that the input distribution has shifted.

**Data**
3. **Validation-failure rate** — `where event == "http_request" | summarize countif(status_code == 400) * 1.0 / count() by bin(_time, 1m)`. *Why:* a spike means upstream callers are sending malformed payloads against the Pydantic schema in [src/schemas.py](src/schemas.py).
4. **Feature distribution** — `where event == "predict_request" | summarize percentile(age, 50), percentile(age, 95), percentile(credit_score, 50) by bin(_time, 1m)`. *Why:* drift in the most-weighted features is the proximate cause of probability collapse — pair this with #1 to know *why* the model is misbehaving.

**Server**
5. **Latency p50/p95/p99 + RPS + status code mix** — `where event == "http_request" | summarize percentile(duration_ms, 50/95/99), count() by bin(_time, 1m)`. *Why:* p95 is the SLO the monitor watches; RPS gives demand context; status breakdown surfaces 5xx.

### Monitor (alert)

A threshold monitor in Axiom watches the p95 latency:

```
['churn-api']
| where event == "http_request"
| summarize p95 = percentile(duration_ms, 95) by bin(_time, 1m)
```

Trigger: `p95 > 2000` for 5 consecutive minutes → email `ishraqahmedjamaluddin@gmail.com`. Demoable by ramping Locust users until the API saturates.

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
