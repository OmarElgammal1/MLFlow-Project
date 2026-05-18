"""Litestar app exposing /, /health, /predict."""

from __future__ import annotations

import logging
import time

from litestar import Litestar, Request, get, post
from litestar.enums import ScopeType
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig
from litestar.middleware import ASGIMiddleware
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR
from litestar.types import ASGIApp, Receive, Scope, Send

from src import predictor
from src.logging_setup import configure_logging
from src.schemas import HealthResponse, HomeResponse, PredictRequest, PredictResponse

logger = logging.getLogger("churn_api")


class HttpEventMiddleware(ASGIMiddleware):
    """Emit one `http_request` event per HTTP call for Axiom dashboards."""

    scopes = (ScopeType.HTTP,)

    async def handle(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        next_app: ASGIApp,
    ) -> None:
        start = time.perf_counter()
        status_holder = {"code": 500}
        scene = "unknown"
        for key, value in scope.get("headers") or []:
            if key == b"x-scene":
                scene = value.decode("ascii", errors="ignore")
                break

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        try:
            await next_app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            client = scope.get("client")
            client_ip = client[0] if client else "unknown"
            logger.info(
                "http_request",
                extra={
                    "event": "http_request",
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "status_code": status_holder["code"],
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip,
                    "scene": scene,
                },
            )


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
    scene = request.headers.get("x-scene", "unknown")
    logger.info(
        "predict_request",
        extra={
            "event": "predict_request",
            "scene": scene,
            "client": request.client.host if request.client else "unknown",
            "credit_score": data.CreditScore,
            "geography": data.Geography,
            "gender": data.Gender,
            "age": data.Age,
            "tenure": data.Tenure,
            "balance": data.Balance,
            "num_of_products": data.NumOfProducts,
            "has_cr_card": data.HasCrCard,
            "is_active_member": data.IsActiveMember,
            "estimated_salary": data.EstimatedSalary,
        },
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
    log_exceptions="debug",
)


app = Litestar(
    route_handlers=[home, health, predict_endpoint],
    on_startup=[_on_startup],
    middleware=[HttpEventMiddleware()],
    logging_config=logging_config,
)
