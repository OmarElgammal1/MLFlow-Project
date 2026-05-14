"""Endpoint tests for the Litestar churn API."""

from __future__ import annotations

from litestar.testing import TestClient


def test_home_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "churn-api"
    assert "version" in body


def test_health_endpoint_reports_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint_returns_prediction(
    client: TestClient, sample_payload: dict
) -> None:
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 201
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_endpoint_validates_missing_field(
    client: TestClient, sample_payload: dict
) -> None:
    bad_payload = {k: v for k, v in sample_payload.items() if k != "Age"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400


def test_predict_endpoint_validates_bad_geography(
    client: TestClient, sample_payload: dict
) -> None:
    bad_payload = {**sample_payload, "Geography": "Atlantis"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
