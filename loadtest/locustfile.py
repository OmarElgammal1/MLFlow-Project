"""Locust load test for the churn API with drift scenes.

Cycles through scenarios that visibly shift input distributions so the Axiom
dashboard shows data + concept-drift signals instead of flat lines.

Every request carries an `X-Scene` header. The API middleware captures it,
so APL queries can group/filter by `scene`.

Scenes (rotate every SCENE_DURATION_SEC):
    steady             -- uniform sample from the full dataset (baseline)
    germany_surge      -- 90% Germany customers (geography drift)
    senior_surge       -- 85% Age 50-65   (age drift, high churn band)
    wealthy_surge      -- 85% Balance>=100k (wealth drift)
    atypical_segment   -- 70% NumOfProducts>=3 (rare segment, 80-100% churn)
    input_noise        -- jitter Age/Balance/CreditScore (concept drift)

Run via the web UI:

    uv sync --group load
    uv run --group load locust -f loadtest/locustfile.py
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

from locust import HttpUser, between, task

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "dataset" / "Churn_Modelling.csv"

SCENE_DURATION_SEC = 120

SCENES = [
    "steady",
    "germany_surge",
    "senior_surge",
    "wealthy_surge",
    "atypical_segment",
    "input_noise",
]


def _load_payloads() -> list[dict]:
    with DATASET_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "CreditScore": int(r["CreditScore"]),
            "Geography": r["Geography"],
            "Gender": r["Gender"],
            "Age": int(r["Age"]),
            "Tenure": int(r["Tenure"]),
            "Balance": float(r["Balance"]),
            "NumOfProducts": int(r["NumOfProducts"]),
            "HasCrCard": int(r["HasCrCard"]),
            "IsActiveMember": int(r["IsActiveMember"]),
            "EstimatedSalary": float(r["EstimatedSalary"]),
        }
        for r in rows
    ]


PAYLOADS = _load_payloads()
BY_GEO = {
    g: [p for p in PAYLOADS if p["Geography"] == g]
    for g in ("France", "Germany", "Spain")
}
SENIORS = [p for p in PAYLOADS if 50 <= p["Age"] <= 65]
WEALTHY = [p for p in PAYLOADS if p["Balance"] >= 100000]
ATYPICAL = [p for p in PAYLOADS if p["NumOfProducts"] >= 3]

BAD_PAYLOADS = [
    {**PAYLOADS[0], "Geography": "Atlantis"},
    {**PAYLOADS[0], "Age": -5},
    {**PAYLOADS[0], "CreditScore": 5000},
    {**PAYLOADS[0], "NumOfProducts": 99},
    {k: v for k, v in PAYLOADS[0].items() if k != "Balance"},
]


class _SceneClock:
    """Wall-clock scene rotator shared across all virtual users."""

    def __init__(self, scenes: list[str], duration: float) -> None:
        self._scenes = scenes
        self._duration = duration
        self._idx = 0
        self._started_at = time.time()

    def current(self) -> str:
        if time.time() - self._started_at >= self._duration:
            self._idx = (self._idx + 1) % len(self._scenes)
            self._started_at = time.time()
        return self._scenes[self._idx]


SCENE = _SceneClock(SCENES, SCENE_DURATION_SEC)


def _sample_for_scene(scene: str) -> dict:
    if scene == "germany_surge":
        pool = BY_GEO["Germany"] if random.random() < 0.9 else PAYLOADS
        return dict(random.choice(pool))
    if scene == "senior_surge":
        pool = SENIORS if random.random() < 0.85 else PAYLOADS
        return dict(random.choice(pool))
    if scene == "wealthy_surge":
        pool = WEALTHY if random.random() < 0.85 else PAYLOADS
        return dict(random.choice(pool))
    if scene == "atypical_segment":
        pool = ATYPICAL if random.random() < 0.7 else PAYLOADS
        return dict(random.choice(pool))
    if scene == "input_noise":
        base = dict(random.choice(PAYLOADS))
        base["Age"] = max(18, min(92, base["Age"] + random.randint(-20, 20)))
        base["Balance"] = max(
            0.0, base["Balance"] + random.uniform(-80000, 80000)
        )
        base["CreditScore"] = max(
            350, min(850, base["CreditScore"] + random.randint(-150, 150))
        )
        return base
    return dict(random.choice(PAYLOADS))


class ChurnApiUser(HttpUser):
    """Single virtual user. Task weights 1 : 2 : 7 : 1 (home : health : predict : bad)."""

    wait_time = between(0.3, 1.2)

    @task(1)
    def home(self) -> None:
        self.client.get("/", name="GET /")

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")

    @task(7)
    def predict(self) -> None:
        scene = SCENE.current()
        payload = _sample_for_scene(scene)
        self.client.post(
            "/predict",
            json=payload,
            name="POST /predict",
            headers={"X-Scene": scene},
        )

    @task(1)
    def predict_bad(self) -> None:
        scene = SCENE.current()
        payload = random.choice(BAD_PAYLOADS)
        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict (bad)",
            catch_response=True,
            headers={"X-Scene": scene},
        ) as response:
            if response.status_code == 400:
                response.success()
