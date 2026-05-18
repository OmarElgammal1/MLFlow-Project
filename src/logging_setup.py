"""Configure Python logging and ship structured events to Axiom."""

from __future__ import annotations

import logging
import os
import sys
import threading

from dotenv import load_dotenv

_configured = False

_STANDARD_LOG_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
        "color_message",
    }
)


def _build_stdout_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    return handler


class AxiomBufferedHandler(logging.Handler):
    """Background-batched logging handler that ships records to Axiom.

    Each LogRecord becomes one event. `extra={...}` keys appear as top-level
    fields in the event so APL queries chart them directly without nested
    attribute paths.
    """

    def __init__(self, client, dataset: str, flush_interval: float = 2.0):
        super().__init__()
        self._client = client
        self._dataset = dataset
        self._flush_interval = flush_interval
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._flusher = threading.Thread(
            target=self._flush_loop, daemon=True, name="axiom-flusher"
        )
        self._flusher.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = self._record_to_event(record)
            with self._lock:
                self._buffer.append(event)
        except Exception:
            self.handleError(record)

    @staticmethod
    def _record_to_event(record: logging.LogRecord) -> dict:
        event: dict = {
            "_time": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_KEYS:
                continue
            event[key] = value
        return event

    def _flush_loop(self) -> None:
        while not self._stop.wait(self._flush_interval):
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return
        try:
            self._client.ingest_events(self._dataset, batch)
        except Exception as exc:
            sys.stderr.write(f"axiom ingest failed: {exc}\n")

    def close(self) -> None:
        self._stop.set()
        self._flush()
        super().close()


def configure_logging() -> None:
    """Wire stdout logging and (optionally) ship logs to Axiom.

    Reads AXIOM_TOKEN and AXIOM_DATASET from environment / .env. Without them,
    only stdout logging is configured so tests and local runs work offline.
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

    token = os.environ.get("AXIOM_TOKEN", "").strip()
    dataset = os.environ.get("AXIOM_DATASET", "").strip()

    if not token or not dataset:
        logging.getLogger(__name__).info(
            "AXIOM_TOKEN/AXIOM_DATASET not set; Axiom shipping disabled, stdout only"
        )
        _configured = True
        return

    try:
        import axiom_py
    except ImportError as exc:
        logging.getLogger(__name__).warning("axiom-py unavailable: %s", exc)
        _configured = True
        return

    client = axiom_py.Client(token=token)
    handler = AxiomBufferedHandler(client, dataset)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    logging.getLogger(__name__).info(
        "Axiom handler wired (dataset=%s)", dataset
    )
    _configured = True
