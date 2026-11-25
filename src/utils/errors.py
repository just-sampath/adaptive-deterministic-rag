"""Application error taxonomy."""

from __future__ import annotations


class RagError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(RagError):
    """Raised when a configured provider cannot be constructed."""


class InfrastructureError(RagError):
    """Raised when a required database or provider is unavailable."""


class ProviderError(InfrastructureError):
    """Raised when an external model or retrieval provider fails."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an unusable response."""


class NotFoundError(RagError):
    """Raised when an authorized resource does not exist."""


class InvalidRequestError(RagError):
    """Raised when input violates an application-level constraint."""


class AuthorizationError(RagError):
    """Raised when request identity cannot be trusted or authorized."""


class RequestDeadlineExceededError(InfrastructureError):
    """Raised when the global deterministic execution budget is exhausted."""


class OperationTimeoutError(TimeoutError):
    """Raised when an upstream client's own transport timeout expires.

    It is a ``TimeoutError`` and deliberately not an ``InfrastructureError``, so a provider's
    timeout is reported like the surrounding stage timer's rather than as an outage.
    """
