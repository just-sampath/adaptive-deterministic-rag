"""Structured JSON logging on stderr for the whole process."""

from __future__ import annotations

import json
import logging
import traceback
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
# Causes, contexts and group members described for one logged exception.
_MAX_DESCRIBED_EXCEPTIONS = 64


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
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = format_exception_safely(record.exc_info[1])
        return json.dumps(payload, default=str)


def format_exception_safely(error: BaseException) -> str:
    """Describe an exception by type and frame location only, with its causes and group members.

    Exception messages, source lines and locals are never included: a message can quote a
    rejected record, a provider payload or request text, and so can any cause a wrapper keeps.
    The outermost exception comes first, followed by each cause or context it carries.
    """

    lines: list[str] = []
    _describe(error, lines, set(), "")
    return "\n".join(lines)


def _describe(error: BaseException, lines: list[str], seen: set[int], indent: str) -> None:
    if id(error) in seen:
        lines.append(f"{indent}{_type_name(error)} (repeated)")
        return
    if len(seen) >= _MAX_DESCRIBED_EXCEPTIONS:
        # A formatter failure would make logging print the raw chain itself, so stay bounded.
        lines.append(f"{indent}{_type_name(error)} (further exceptions omitted)")
        return
    seen.add(id(error))
    lines.append(f"{indent}{_type_name(error)}")
    for frame, line_number in traceback.walk_tb(error.__traceback__):
        code = frame.f_code
        lines.append(f'{indent}  File "{code.co_filename}", line {line_number}, in {code.co_name}')
    if isinstance(error, BaseExceptionGroup):
        count = len(error.exceptions)
        for position, member in enumerate(error.exceptions, start=1):
            lines.append(f"{indent}  Group member {position} of {count}:")
            _describe(member, lines, seen, indent + "    ")
    if error.__cause__ is not None:
        lines.append(f"{indent}Caused by:")
        _describe(error.__cause__, lines, seen, indent)
    elif error.__context__ is not None and not error.__suppress_context__:
        lines.append(f"{indent}Raised while handling:")
        _describe(error.__context__, lines, seen, indent)


def _type_name(error: BaseException) -> str:
    kind = type(error)
    if kind.__module__ == "builtins":
        return kind.__qualname__
    return f"{kind.__module__}.{kind.__qualname__}"


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
