"""Tenant-scoped trace reads visible to the owner and to administrators."""

from __future__ import annotations

from models.domain import Identity, RequestTrace
from services.contracts import TraceReader
from utils.errors import NotFoundError


class TenantTraceHandler:
    """Return a trace only to its owner or to the administrator role within the tenant."""

    def __init__(self, *, reader: TraceReader, admin_role: str) -> None:
        self._reader = reader
        self._admin_role = admin_role

    async def get_trace(self, request_id: str, identity: Identity) -> RequestTrace:
        trace = await self._reader.get_trace(request_id, identity.tenant_id)
        if trace is None or not self._can_view(trace, identity):
            raise NotFoundError("Trace not found")
        return trace

    def _can_view(self, trace: RequestTrace, identity: Identity) -> bool:
        return trace.user_id == identity.user_id or self._admin_role in identity.roles
