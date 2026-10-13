"""MongoDB persistence adapter for telemetry events and RAG traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from managers.errors import mongo_errors
from models.domain import RequestTrace, TelemetryEvent


class MongoTelemetrySink:
    def __init__(
        self,
        database: AsyncDatabase[dict[str, Any]],
        *,
        trace_collection: str,
        event_collection: str,
    ) -> None:
        self._traces = database[trace_collection]
        self._events = database[event_collection]

    @mongo_errors
    async def write_events(self, events: Sequence[TelemetryEvent]) -> None:
        if events:
            await self._events.insert_many(
                [event.model_dump(mode="python") for event in events], ordered=False
            )

    @mongo_errors
    async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
        if traces:
            await self._traces.insert_many(
                [trace.model_dump(mode="python") for trace in traces], ordered=False
            )

    @mongo_errors
    async def get_trace(self, request_id: str, tenant_id: str) -> RequestTrace | None:
        item = await self._traces.find_one(
            {"tenant_id": tenant_id, "request_id": request_id}, {"_id": 0}
        )
        return RequestTrace.model_validate(item) if item else None
