"""Handler interfaces exposed to transport controllers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from models.domain import Identity, QueryRequest, QueryResponse, RequestTrace


class QueryHandler(Protocol):
    async def query(self, identity: Identity, request: QueryRequest) -> QueryResponse: ...


class TraceQueryHandler(Protocol):
    async def get_trace(self, request_id: str, identity: Identity) -> RequestTrace: ...


class ReadinessHandler(Protocol):
    async def check(self) -> None: ...


TelemetryValue = str | int | float | bool | None


class TransportTelemetry(Protocol):
    """Event port for routes, middleware and HTTP error mapping (TelemetryService satisfies it)."""

    def emit(
        self,
        event_type: str,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        attributes: Mapping[str, TelemetryValue] | None = None,
    ) -> None: ...
