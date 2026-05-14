"""Litestar app exposing /, /health, /predict."""

from __future__ import annotations

import logging

from litestar import Litestar, Request, get, post
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR

from src import predictor
from src.logging_setup import configure_logging
from src.schemas import HealthResponse, HomeResponse, PredictRequest, PredictResponse

logger = logging.getLogger("churn_api")


@get("/", sync_to_thread=False)
def home() -> HomeResponse:
    logger.info("home endpoint hit")
    return HomeResponse(
        service="churn-api",
        version="0.1.0",
        docs="/schema/swagger",
    )


@get("/health", sync_to_thread=False)
def health() -> HealthResponse:
    loaded = predictor.is_loaded()
    logger.info("health probe", extra={"model_loaded": loaded})
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@post("/predict", sync_to_thread=True)
def predict_endpoint(data: PredictRequest, request: Request) -> PredictResponse:
    logger.info(
        "predict request received",
        extra={"client": request.client.host if request.client else "unknown"},
    )
    try:
        return predictor.predict(data)
    except Exception as exc:
        logger.exception("prediction failed")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc


def _on_startup() -> None:
    configure_logging()
    try:
        predictor.load_artifacts()
    except RuntimeError as exc:
        logger.warning("startup: artifacts not loaded yet (%s)", exc)


logging_config = LoggingConfig(
    root={"level": "INFO", "handlers": ["queue_listener"]},
    formatters={
        "standard": {"format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"}
    },
    log_exceptions="always",
)


app = Litestar(
    route_handlers=[home, health, predict_endpoint],
    on_startup=[_on_startup],
    logging_config=logging_config,
)
