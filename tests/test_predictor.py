"""Function-level tests for src.api.predictor."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api import predictor
from src.api.schemas import PredictRequest, PredictResponse


def test_predict_returns_valid_response(sample_payload: dict) -> None:
    """predict() should return a 0/1 label and a probability in [0, 1]."""
    request = PredictRequest(**sample_payload)
    result = predictor.predict(request)

    assert isinstance(result, PredictResponse)
    assert result.prediction in (0, 1)
    assert 0.0 <= result.probability <= 1.0


def test_predict_feature_hash_is_stable(sample_payload: dict) -> None:
    """Same input always hashes to the same short digest."""
    request = PredictRequest(**sample_payload)
    h1 = predictor._hash_features(request)
    h2 = predictor._hash_features(request)
    assert h1 == h2
    assert len(h1) == 12


def test_is_loaded_after_load(sample_payload: dict) -> None:
    assert predictor.is_loaded() is True


def test_load_artifacts_missing_model_raises(tmp_path: Path) -> None:
    """Pointing at a non-existent model.pkl produces a clean RuntimeError."""
    predictor.reset()
    try:
        with pytest.raises(RuntimeError, match="Model artifact not found"):
            predictor.load_artifacts(
                model_path=tmp_path / "missing.pkl",
                transformer_path=tmp_path / "missing_transformer.pkl",
            )
    finally:
        predictor.load_artifacts()


def test_load_artifacts_missing_transformer_raises(tmp_path: Path) -> None:
    """Pointing at a non-existent transformer.pkl produces a clean RuntimeError."""
    predictor.reset()
    fake_model = tmp_path / "model.pkl"
    fake_model.write_bytes(b"placeholder")
    try:
        with pytest.raises(RuntimeError, match="Transformer artifact not found"):
            predictor.load_artifacts(
                model_path=fake_model,
                transformer_path=tmp_path / "missing_transformer.pkl",
            )
    finally:
        predictor.load_artifacts()
