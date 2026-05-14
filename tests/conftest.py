"""Shared fixtures for the churn API tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from litestar.testing import TestClient

os.environ.setdefault("HYPERDX_API_KEY", "")

from src import predictor  # noqa: E402
from src.app import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _load_artifacts() -> None:
    predictor.reset()
    predictor.load_artifacts()


@pytest.fixture()
def sample_payload() -> dict:
    return {
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


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app=app) as test_client:
        yield test_client
