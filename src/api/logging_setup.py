"""Configure Python logging and OpenTelemetry export to HyperDX."""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

DEFAULT_ENDPOINT = "https://in-otel.hyperdx.io"
_configured = False


def _build_stdout_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    return handler


def configure_logging() -> None:
    """Wire stdout logging and (optionally) ship logs to HyperDX via OTLP.

    Reads HYPERDX_API_KEY from environment / .env. Without the key, only
    stdout logging is configured so tests and local runs work offline.
    """
    global _configured
    if _configured:
        return

    load_dotenv(override=False)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(_build_stdout_handler())

    api_key = os.environ.get("HYPERDX_API_KEY", "").strip()
    if not api_key:
        logging.getLogger(__name__).info(
            "HYPERDX_API_KEY not set; OTLP exporter disabled, stdout only"
        )
        _configured = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logging.getLogger(__name__).warning(
            "OpenTelemetry SDK unavailable: %s", exc
        )
        _configured = True
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_ENDPOINT).rstrip(
        "/"
    )
    service_name = os.environ.get("OTEL_SERVICE_NAME", "churn-api")
    headers = {"authorization": api_key}

    resource = Resource.create({"service.name": service_name})

    log_provider = LoggerProvider(resource=resource)
    log_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=f"{endpoint}/v1/logs", headers=headers)
        )
    )
    set_logger_provider(log_provider)
    root.addHandler(LoggingHandler(level=logging.INFO, logger_provider=log_provider))

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", headers=headers)
        )
    )
    trace.set_tracer_provider(tracer_provider)

    logging.getLogger(__name__).info(
        "HyperDX OTLP exporters wired (service=%s, endpoint=%s)",
        service_name,
        endpoint,
    )
    _configured = True
