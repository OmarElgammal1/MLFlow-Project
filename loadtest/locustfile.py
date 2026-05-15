"""Locust load test for the deployed churn API.

Run via the web UI:

    uv sync --group load
    uv run locust -f loadtest/locustfile.py

Then open http://localhost:8089. The host is prefilled from $LOCUST_HOST.
"""

from locust import HttpUser, between, task

CANONICAL_PAYLOAD = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
}


class ChurnApiUser(HttpUser):
    """Single virtual user. Locust will spawn N of these per the swarm config.

    Task weights (1 : 2 : 7) approximate real traffic where /predict is the
    main endpoint and the others are infrequent metadata / liveness probes.
    """
    # search for churn api user 0%

    wait_time = between(1, 3)

    @task(1)
    def home(self) -> None:
        self.client.get("/", name="GET /")

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")

    @task(7)
    def predict(self) -> None:
        self.client.post("/predict", json=CANONICAL_PAYLOAD, name="POST /predict")
