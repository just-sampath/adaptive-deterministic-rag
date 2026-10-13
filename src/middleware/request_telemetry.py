"""Concurrency-safe request IDs and request telemetry."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Protocol, cast
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from handlers.contracts import TransportTelemetry
from utils.request_context import bind_request, current_identity, reset_request

_REQUEST_ID_HEADER = b"x-request-id"


class _TelemetryHost(Protocol):
    @property
    def telemetry(self) -> TransportTelemetry: ...


class RequestTelemetryMiddleware:
    """Record request completion without mutable per-request instance state."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        tokens = bind_request(request_id)
        started = perf_counter()
        status_code = 500
        error_code: str | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = _with_request_id(message.get("headers", ()), request_id)
            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        except Exception as exc:
            error_code = type(exc).__name__
            raise
        finally:
            identity = current_identity()
            telemetry = cast(_TelemetryHost, scope["app"].state.container).telemetry
            telemetry.emit(
                "http.request.completed",
                request_id=request_id,
                tenant_id=identity.tenant_id if identity else None,
                user_id=identity.user_id if identity else None,
                attributes={
                    "method": scope.get("method", ""),
                    "path": _route_template(scope),
                    "status_code": status_code,
                    "latency_ms": round((perf_counter() - started) * 1000, 3),
                    "error_code": error_code,
                },
            )
            reset_request(tokens)


def _with_request_id(
    headers: Iterable[tuple[bytes, bytes]],
    request_id: str,
) -> list[tuple[bytes, bytes]]:
    kept = [(name, value) for name, value in headers if name.lower() != _REQUEST_ID_HEADER]
    kept.append((_REQUEST_ID_HEADER, request_id.encode()))
    return kept


def _route_template(scope: Scope) -> str:
    """Return the matched route template so events never carry raw path identifiers."""

    template = getattr(scope.get("route"), "path", None)
    return template if isinstance(template, str) and template else "<unmatched>"
