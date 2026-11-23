"""HTTP error mapping from application exceptions to stable JSON responses."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models.domain import Identity
from routes.dependencies import get_route_context
from utils.errors import (
    AuthorizationError,
    InfrastructureError,
    InvalidRequestError,
    NotFoundError,
    RequestDeadlineExceededError,
)
from utils.request_context import current_identity, current_request_id

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthorizationError)
    async def authorization_error(request: Request, exc: AuthorizationError) -> JSONResponse:
        return _error_response(request, 403, "FORBIDDEN", str(exc))

    @app.exception_handler(NotFoundError)
    async def not_found_error(request: Request, exc: NotFoundError) -> JSONResponse:
        return _error_response(request, 404, "NOT_FOUND", str(exc))

    @app.exception_handler(InvalidRequestError)
    async def invalid_request(request: Request, exc: InvalidRequestError) -> JSONResponse:
        return _error_response(request, 422, "INVALID_REQUEST", str(exc))

    @app.exception_handler(RequestValidationError)
    async def request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _json_response(
            request,
            422,
            {"error": "INVALID_REQUEST", "details": _validation_details(exc)},
        )

    @app.exception_handler(RequestDeadlineExceededError)
    async def deadline_error(request: Request, exc: RequestDeadlineExceededError) -> JSONResponse:
        _record_failure(request, exc)
        return _error_response(request, 504, "REQUEST_DEADLINE_EXCEEDED", str(exc))

    @app.exception_handler(TimeoutError)
    async def operation_timeout(request: Request, exc: TimeoutError) -> JSONResponse:
        _record_failure(request, exc)
        return _error_response(request, 504, "OPERATION_TIMEOUT", "An operation timed out")

    @app.exception_handler(InfrastructureError)
    async def infrastructure_error(request: Request, exc: InfrastructureError) -> JSONResponse:
        _record_failure(request, exc)
        return _error_response(request, 503, "SERVICE_UNAVAILABLE", "A required service failed")

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        return unexpected_error_response(request, exc)


def unexpected_error_response(request: Request, exc: Exception) -> JSONResponse:
    """Record an unexpected failure once and return the stable 500 response."""

    _record_failure(request, exc)
    return _error_response(request, 500, "INTERNAL_ERROR", "An unexpected error occurred")


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Expose location, message and type only; never echo the rejected input."""

    return [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]


def _record_failure(request: Request, exc: Exception) -> None:
    request_id = _request_id(request)
    identity = _identity(request)
    tenant_id = identity.tenant_id if identity else None
    error_code = type(exc).__name__
    logger.error(
        "Request failed",
        exc_info=None if _embeds_input(exc) else (type(exc), exc, exc.__traceback__),
        extra={"request_id": request_id, "tenant_id": tenant_id, "error_code": error_code},
    )
    get_route_context(request).telemetry.emit(
        "http.request.failed",
        request_id=request_id,
        tenant_id=tenant_id,
        user_id=identity.user_id if identity else None,
        attributes={"error_code": error_code},
    )


def _embeds_input(exc: Exception) -> bool:
    """Validation messages quote rejected input or model output, so they are logged by type only."""

    return isinstance(exc, ValidationError | RequestValidationError)


def _request_id(request: Request) -> str | None:
    request_id = getattr(request.state, "request_id", None)
    return request_id if isinstance(request_id, str) else current_request_id()


def _identity(request: Request) -> Identity | None:
    identity = getattr(request.state, "identity", None)
    return identity if isinstance(identity, Identity) else current_identity()


def _error_response(request: Request, status_code: int, error: str, detail: str) -> JSONResponse:
    return _json_response(request, status_code, {"error": error, "detail": detail})


def _json_response(request: Request, status_code: int, content: dict[str, Any]) -> JSONResponse:
    request_id = _request_id(request)
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(status_code=status_code, content=content, headers=headers)
