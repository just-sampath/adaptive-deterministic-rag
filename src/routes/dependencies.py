"""Shared FastAPI dependencies for trusted identity and the transport route context."""

from __future__ import annotations

import secrets
from typing import Annotated, Protocol, cast

from fastapi import Depends, Request
from pydantic import ValidationError

from handlers.contracts import (
    QueryHandler,
    ReadinessHandler,
    TraceQueryHandler,
    TransportTelemetry,
)
from models.domain import Identity
from utils.errors import AuthorizationError
from utils.request_context import bind_identity
from utils.settings import Settings


class RouteContext(Protocol):
    """Settings and handler ports that transport code is allowed to use."""

    @property
    def settings(self) -> Settings: ...

    @property
    def query_handler(self) -> QueryHandler: ...

    @property
    def trace_handler(self) -> TraceQueryHandler: ...

    @property
    def readiness_handler(self) -> ReadinessHandler: ...

    @property
    def telemetry(self) -> TransportTelemetry: ...


def get_route_context(request: Request) -> RouteContext:
    return cast(RouteContext, request.app.state.container)


Context = Annotated[RouteContext, Depends(get_route_context)]


async def resolve_identity(request: Request, context: Context) -> Identity:
    config = context.settings.auth
    expected_token = context.settings.external.auth_shared_token.get_secret_value()
    if not _secret_matches(request.headers.get(config.token_header, ""), expected_token):
        raise AuthorizationError("The trusted identity token is missing or invalid")
    tenant_id = request.headers.get(config.tenant_header, "").strip()
    user_id = request.headers.get(config.user_header, "").strip()
    roles = frozenset(
        role.strip()
        for role in request.headers.get(config.roles_header, "").split(",")
        if role.strip()
    )
    if not tenant_id or not user_id or not roles:
        raise AuthorizationError("Tenant, user, and at least one role are required")
    try:
        identity = Identity(tenant_id=tenant_id, user_id=user_id, roles=roles)
    except ValidationError as exc:
        raise AuthorizationError("Trusted identity headers are invalid") from exc
    bind_identity(identity)
    request.state.identity = identity
    return identity


RequestIdentity = Annotated[Identity, Depends(resolve_identity)]


def require_admin(identity: RequestIdentity, context: Context) -> Identity:
    if context.settings.auth.admin_role not in identity.roles:
        raise AuthorizationError("Administrator role required")
    return identity


AdminIdentity = Annotated[Identity, Depends(require_admin)]


def require_debug_access(
    request: Request,
    identity: RequestIdentity,
    context: Context,
) -> Identity:
    supplied = request.headers.get(context.settings.auth.debug_token_header, "")
    expected = context.settings.external.debug_token.get_secret_value()
    if not _secret_matches(supplied, expected):
        raise AuthorizationError("Debug access token is missing or invalid")
    return identity


DebugIdentity = Annotated[Identity, Depends(require_debug_access)]


def _secret_matches(supplied: str, expected: str) -> bool:
    """Compare UTF-8 bytes so non-ASCII header values fail closed instead of raising."""

    return bool(supplied) and secrets.compare_digest(supplied.encode(), expected.encode())
