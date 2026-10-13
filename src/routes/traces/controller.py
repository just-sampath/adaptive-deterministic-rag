from __future__ import annotations

from uuid import UUID

from routes.dependencies import Context, DebugIdentity
from routes.traces.validator import TraceResponse


async def get_trace(
    request_id: UUID,
    identity: DebugIdentity,
    context: Context,
) -> TraceResponse:
    return await context.trace_handler.get_trace(str(request_id), identity)
