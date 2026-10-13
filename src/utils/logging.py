"""Structured JSON logging on stderr for the whole process."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_STRUCTURED_FIELDS = ("request_id", "request_ids", "tenant_id", "event_type", "error_code")
# SDK loggers can emit prompts and payloads at DEBUG, so they never go below WARNING. The
# OpenAI SDK sends requests through the "httpx2" transport logger, not "httpx".
_PAYLOAD_LOGGERS = ("openai", "httpx", "httpx2", "httpcore", "pymongo", "qdrant_client")


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
