"""Task-local request and identity context for concurrent ASGI requests."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from models.domain import Identity

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_identity: ContextVar[Identity | None] = ContextVar("identity", default=None)


@dataclass(frozen=True, slots=True)
class RequestContextTokens:
    request_id: Token[str | None]
    identity: Token[Identity | None]


def bind_request(request_id: str) -> RequestContextTokens:
    return RequestContextTokens(
        request_id=_request_id.set(request_id),
        identity=_identity.set(None),
    )


def bind_identity(identity: Identity) -> None:
    _identity.set(identity)


def reset_request(tokens: RequestContextTokens) -> None:
    _identity.reset(tokens.identity)
    _request_id.reset(tokens.request_id)


def current_request_id() -> str | None:
    return _request_id.get()


def current_identity() -> Identity | None:
    return _identity.get()
