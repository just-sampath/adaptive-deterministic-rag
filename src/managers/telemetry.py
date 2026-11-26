"""MongoDB persistence adapter for telemetry events and RAG traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bson import encode
from bson.errors import BSONError
from pymongo import ReadPreference, UpdateOne
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import BulkWriteError, ConnectionFailure, PyMongoError

from managers.errors import mongo_errors
from models.domain import RequestTrace, TelemetryEvent
from utils.errors import InfrastructureError, TraceConflictError, TransientTraceWriteError

_ACCOUNTING_FIELDS = (
    "operations",
    "usage_coverage",
    "handler_outcome",
    "returned_status",
    "returned_document_ids",
)

type TraceKey = tuple[str, str]
type Document = dict[str, Any]


def _key_filter(key: TraceKey) -> dict[str, str]:
    return {"tenant_id": key[0], "request_id": key[1]}


def _same_document(left: Document, right: Document) -> bool:
    # Compare the BSON each side encodes to, so a stored date (milliseconds) or array (list)
    # matches its in-memory form, but an int never matches a string or an extended-JSON-shaped
    # dict. Field order is not part of a trace's identity, so keys are sorted first.
    return encode(_sorted(left)) == encode(_sorted(right))


def _sorted(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sorted(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_sorted(item) for item in value]
    return value


def _duplicates_only(error: BulkWriteError) -> bool:
    errors = error.details.get("writeErrors", ())
    return (
        bool(errors)
        and not error.details.get("writeConcernErrors")
        and all(item.get("code") == 11000 for item in errors)
    )


def _transient(error: PyMongoError) -> bool:
    if isinstance(error, BulkWriteError):
        # BulkWriteError.timeout only inspects the last entry. Never replay a batch with a
        # permanent per-document error just because another operation happened to time out.
        entries = error.details.get("writeErrors", ())
        if any(
            item.get("code") != 50 and "RetryableWriteError" not in item.get("errorLabels", ())
            for item in entries
        ):
            return False
    return (
        isinstance(error, ConnectionFailure)
        or error.timeout
        or error.has_error_label("RetryableWriteError")
    )


class MongoTelemetrySink:
    def __init__(
        self,
        database: AsyncDatabase[dict[str, Any]],
        *,
        trace_collection: str,
        event_collection: str,
        write_query_intent_fields: bool = False,
        write_accounting_fields: bool = False,
    ) -> None:
        self._traces = database[trace_collection]
        self._events = database[event_collection]
        self._trace_exclude: Document = {}
        if not write_query_intent_fields:
            self._trace_exclude["contextualization"] = {"proposed_drift_score", "rewrite_decision"}
        if not write_accounting_fields:
            self._trace_exclude.update(dict.fromkeys(_ACCOUNTING_FIELDS, True))

    @mongo_errors
    async def write_events(self, events: Sequence[TelemetryEvent]) -> None:
        if events:
            await self._events.insert_many(
                [event.model_dump(mode="python") for event in events], ordered=False
            )

    async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
        documents: dict[TraceKey, Document] = {}
        conflicting = False
        try:
            for trace in traces:
                key = (trace.tenant_id, trace.request_id)
                document = self._document(trace)
                if key in documents:
                    # Encoding can reject a document here too, so it stays inside the boundary.
                    conflicting |= not _same_document(documents[key], document)
                else:
                    documents[key] = document
            if not documents:
                return
            await self._upsert_traces(documents)
        except PyMongoError as exc:
            if _transient(exc):
                raise TransientTraceWriteError(
                    "MongoDB trace write could not be confirmed"
                ) from exc
            raise InfrastructureError("MongoDB trace write failed") from exc
        except BSONError as exc:
            raise InfrastructureError("MongoDB trace document is invalid") from exc
        if conflicting:
            raise TraceConflictError("Trace key already identifies a different trace")

    async def _upsert_traces(self, documents: Mapping[TraceKey, Document]) -> None:
        updates = [
            UpdateOne(_key_filter(key), {"$setOnInsert": document}, upsert=True)
            for key, document in documents.items()
        ]
        try:
            result = await self._traces.bulk_write(updates, ordered=False)
        except BulkWriteError as exc:
            if not _duplicates_only(exc):
                raise
            # Concurrent writers may race on the unique index. Confirm each stored payload.
            replays = dict(documents)
        else:
            inserted = result.upserted_ids or {}
            replays = {
                key: document
                for index, (key, document) in enumerate(documents.items())
                if index not in inserted
            }
        if not replays:
            return
        primary = self._traces.with_options(read_preference=ReadPreference.PRIMARY)
        stored = await primary.find(
            {"$or": [_key_filter(key) for key in replays]}, {"_id": 0}
        ).to_list()
        for row in stored:
            if not _same_document(row, replays.pop((row["tenant_id"], row["request_id"]))):
                raise TraceConflictError("Trace key already identifies a different trace")
        if replays:
            raise TransientTraceWriteError("Previously matched trace could not be confirmed")

    def _document(self, trace: RequestTrace) -> dict[str, Any]:
        document = trace.model_dump(mode="python", exclude=self._trace_exclude)
        for attempt in document["attempts"]:
            if attempt["outcome"] is None:
                del attempt["outcome"]
        for row in document.get("operations") or ():
            # A first try keeps the row layout older readers accept; only a retry is marked.
            if row["retry_ordinal"] == 0:
                del row["retry_ordinal"]
        return document

    @mongo_errors
    async def get_trace(self, request_id: str, tenant_id: str) -> RequestTrace | None:
        item = await self._traces.find_one(
            {"tenant_id": tenant_id, "request_id": request_id}, {"_id": 0}
        )
        return RequestTrace.model_validate(item) if item else None
