from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request

from models.domain import Identity
from routes.dependencies import RouteContext, require_admin, require_debug_access, resolve_identity
from utils.errors import AuthorizationError
from utils.settings import Settings


def make_context(make_settings: Callable[..., Settings]) -> RouteContext:
    return cast(RouteContext, SimpleNamespace(settings=make_settings()))


def make_request(headers: dict[str, str]) -> Request:
    encoded = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": encoded})


async def test_trusted_headers_resolve_identity_and_admin_access(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(make_settings)
    request = make_request(
        {
            "X-Auth-Token": "test-auth-token",
            "X-Tenant-ID": "tenant-a",
            "X-User-ID": "user-a",
            "X-Roles": "reader, administrator",
            "X-Debug-Token": "test-debug-token",
        }
    )

    identity = await resolve_identity(request, context)

    assert identity.tenant_id == "tenant-a"
    assert identity.roles == frozenset({"reader", "administrator"})
    assert request.state.identity is identity
    assert require_admin(identity, context) is identity
    assert require_debug_access(request, identity, context) is identity


async def test_trusted_headers_reject_untrusted_identity_claims(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(make_settings)
    request = make_request(
        {
            "X-Auth-Token": "wrong-token",
            "X-Tenant-ID": "tenant-a",
            "X-User-ID": "user-a",
            "X-Roles": "reader",
        }
    )

    with pytest.raises(AuthorizationError, match="missing or invalid"):
        await resolve_identity(request, context)


async def test_trusted_headers_reject_invalid_identity_without_server_error(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(make_settings)
    request = make_request(
        {
            "X-Auth-Token": "test-auth-token",
            "X-Tenant-ID": "t" * 129,
            "X-User-ID": "user-a",
            "X-Roles": "reader",
        }
    )

    with pytest.raises(AuthorizationError, match="headers are invalid"):
        await resolve_identity(request, context)


async def test_non_ascii_identity_token_fails_closed(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(make_settings)
    request = make_request(
        {
            "X-Auth-Token": "tökén-ünïcode",
            "X-Tenant-ID": "tenant-a",
            "X-User-ID": "user-a",
            "X-Roles": "reader",
        }
    )

    with pytest.raises(AuthorizationError, match="missing or invalid"):
        await resolve_identity(request, context)


def test_non_ascii_debug_token_fails_closed(
    make_settings: Callable[..., Settings],
    identity: Identity,
) -> None:
    context = make_context(make_settings)
    request = make_request({"X-Debug-Token": "dëbug-tökén"})

    with pytest.raises(AuthorizationError, match="Debug access token"):
        require_debug_access(request, identity, context)


def test_admin_role_is_required_for_administrative_routes(
    make_settings: Callable[..., Settings],
    identity: Identity,
) -> None:
    context = make_context(make_settings)

    with pytest.raises(AuthorizationError, match="Administrator role required"):
        require_admin(identity, context)
