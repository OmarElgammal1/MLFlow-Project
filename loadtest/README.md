# Load testing the Churn API with Locust

A Locust swarm hits the deployed (or local) API and reports request rate, latency, and failures via Locust's built-in web UI.

## Prerequisites

- The Litestar API is reachable at some HTTP URL (local Docker container, or the EC2 deployment from [.github/workflows/actions.yml](../.github/workflows/actions.yml)).
- One env var:

  ```
  LOCUST_HOST=http://<your-host>:8000
  ```

  Set this in the repo-root [.env](../.env) (or export it in your shell). Locust reads it natively to prefill the **Host** field in the UI.

## Install the load-test dependencies

```bash
uv sync --group load
```

Locust is in its own dep group so it never leaks into the production image or the test job.

## Run the web UI

```bash
uv run locust -f loadtest/locustfile.py
```

Then open <http://localhost:8089>. Three fields:

- **Number of users** — virtual users spawned (the swarm size).
- **Spawn rate** — users added per second until the target is reached.
- **Host** — prefilled from `LOCUST_HOST`.

Click **Start swarming**. Stop the swarm at any time with the top-right button; the charts and the **Download Data** button stay populated.

## What the locustfile does

[loadtest/locustfile.py](locustfile.py) defines a single `ChurnApiUser`. Each virtual user waits 1–3 seconds between requests and picks one of three tasks with weights **1 : 2 : 7**:

| Weight | Endpoint        | Why                                                     |
| ------ | --------------- | ------------------------------------------------------- |
| 1      | `GET /`         | Cheap metadata endpoint — closest to noise.             |
| 2      | `GET /health`   | Liveness probe — short, no model work.                  |
| 7      | `POST /predict` | The expensive path — model + transformer execution.    |

The payload is a fixed canonical record (one Pydantic-valid customer). Same input every call — the test characterizes the *system*, not the model's branching.

## Experiment matrix

| Users | Spawn rate | Duration | What to look for                                            |
| ----- | ---------- | -------- | ----------------------------------------------------------- |
| 10    | 1/s        | 60 s     | Baseline latency, comfortable RPS for a warm app.           |
| 100   | 10/s       | 90 s     | First **p95 knee** — when latency starts climbing.          |
| 500   | 25/s       | 120 s    | First failures / 5xx. EC2 CPU pegged? Memory OK?            |
| 1000  | 50/s       | 120 s    | Saturation. Where does the service start dropping requests? |

## Which charts to watch

In the **Charts** tab, three matter most:

- **Total Requests per Second** — the throughput curve. A flat top means the system can't accept more work no matter how many users you add.
- **Response Times (ms)** — p50 / p95 / p99. p95 crossing an SLO (e.g. 250 ms) tells you the practical user limit.
- **Number of Users** — confirms the swarm grew as configured before drawing conclusions from the other two.

The **Failures** tab lists every error response by endpoint and reason — usually how you spot the breaking point.

## Quick local check before targeting EC2

```bash
docker run -d --name churn-api -p 8000:8000 churn-api:latest
LOCUST_HOST=http://127.0.0.1:8000 uv run locust -f loadtest/locustfile.py
```

If that works, switch `LOCUST_HOST` to the EC2 address and re-run — same locustfile, no other changes.
