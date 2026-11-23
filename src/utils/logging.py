"""Structured JSON logging on stderr for the whole process."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_STRUCTURED_FIELDS = ("request_id", "request_ids", "tenant_id", "event_type", "error_code")
# SDK loggers can emit prompts and payloads at DEBUG, so they never go below WARNING. The
# OpenAI SDK sends requests through the "httpx2" and "httpcore2" transport loggers.
_PAYLOAD_LOGGERS = (
    "openai",
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "pymongo",
    "qdrant_client",
)
_SERVER_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _STRUCTURED_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    """Configure one structured stderr handler at process startup."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in _PAYLOAD_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # The uvicorn CLI installs its own plain-text handlers before the app is imported; send its
    # startup and access logs through the JSON handler instead.
    for name in _SERVER_LOGGERS:
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True
