"""Immutable trace upserts and the Mongo error boundary, without an external store."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, create_autospec

import pytest
from bson import BSON
from pymongo import ReadPreference, UpdateOne
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import AutoReconnect, BulkWriteError, OperationFailure
from pymongo.results import BulkWriteResult

from managers.telemetry import MongoTelemetrySink
from tests.test_telemetry import make_trace
from utils.errors import InfrastructureError, TraceConflictError, TransientTraceWriteError


class TraceStore:
    """Apply the upsert to BSON documents and expose the decoded state for assertions."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.fail_after_first_insert = False
        self.collection = create_autospec(AsyncCollection, instance=True)
        self.collection.bulk_write.side_effect = self.bulk_write
        self.collection.with_options.return_value.find.side_effect = self.find
        db = cast(AsyncDatabase[dict[str, Any]], {"traces": self.collection, "events": None})
        self.sink = MongoTelemetrySink(db, trace_collection="traces", event_collection="events")

    async def bulk_write(self, operations: list[UpdateOne], *, ordered: bool) -> BulkWriteResult:
        assert ordered is False
        inserted = []
        for index, op in enumerate(operations):
            assert isinstance(op, UpdateOne) and op._upsert is True
            assert set(op._filter) == {"tenant_id", "request_id"}
            assert isinstance(op._doc, dict) and set(op._doc) == {"$setOnInsert"}
            key = (op._filter["tenant_id"], op._filter["request_id"])
            if key in self.rows:
                continue
            self.rows[key] = BSON(BSON.encode(op._doc["$setOnInsert"])).decode()
            inserted.append({"index": index, "_id": index})
            if self.fail_after_first_insert:
                self.fail_after_first_insert = False
                raise AutoReconnect("ambiguous result")
        return BulkWriteResult({"upserted": inserted}, acknowledged=True)

    def find(self, query: dict[str, Any], projection: dict[str, int]) -> Any:
        assert projection == {"_id": 0}
        keys = [(item["tenant_id"], item["request_id"]) for item in query["$or"]]
        rows = [self.rows[key] for key in keys if key in self.rows]
        return type("Cursor", (), {"to_list": AsyncMock(return_value=rows)})()


async def test_partial_write_can_replay_without_changing_the_first_document_or_ttl() -> None:
    store = TraceStore()
    first, second = make_trace("first"), make_trace("second")
    store.fail_after_first_insert = True
    with pytest.raises(TransientTraceWriteError):
        await store.sink.write_traces([first, second])
    original = store.rows["tenant", "first"].copy()
    await store.sink.write_traces([first, second])
    await store.sink.write_traces([first, second])
    assert set(store.rows) == {("tenant", "first"), ("tenant", "second")}
    assert store.rows["tenant", "first"] == original
    store.collection.with_options.assert_called_with(read_preference=ReadPreference.PRIMARY)


@pytest.mark.parametrize("change", [{"user_id": "another"}, {"latency_ms": 99}])
async def test_trace_conflict_never_modifies_the_stored_payload(change: dict[str, Any]) -> None:
    store = TraceStore()
    original = make_trace("same")
    await store.sink.write_traces([original])
    saved = store.rows["tenant", "same"].copy()
    with pytest.raises(TraceConflictError):
        await store.sink.write_traces([original.model_copy(update=change)])
    assert store.rows["tenant", "same"] == saved


async def test_same_batch_key_conflict_keeps_first_owner_and_writes_independent_trace() -> None:
    store = TraceStore()
    first = make_trace("same")
    new_tenant = first.model_copy(update={"tenant_id": "another"})
    duplicate = first.model_copy(update={"user_id": "another-owner"})
    with pytest.raises(TraceConflictError):
        await store.sink.write_traces([first, duplicate, new_tenant])
    assert store.rows["tenant", "same"]["user_id"] == first.user_id
    assert store.rows["another", "same"]["user_id"] == first.user_id


async def test_replay_compares_bson_types_and_allows_different_dictionary_order() -> None:
    store = TraceStore()
    trace = make_trace("types")
    first = trace.model_copy(update={"literal_question_observation": {"n": 1, "mode": "full"}})
    await store.sink.write_traces([first])
    reordered = trace.model_copy(update={"literal_question_observation": {"mode": "full", "n": 1}})
    await store.sink.write_traces([reordered])
    changed = trace.model_copy(
        update={"literal_question_observation": {"mode": "full", "n": {"$numberInt": "1"}}}
    )
    with pytest.raises(TraceConflictError):
        await store.sink.write_traces([changed])


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (OperationFailure("stepdown", details={"errorLabels": ["RetryableWriteError"]}), True),
        (BulkWriteError({"writeErrors": [{"code": 121}]}), False),
        (OperationFailure("authorization denied", code=13), False),
    ],
)
async def test_only_transient_trace_writes_can_retry(error: Exception, transient: bool) -> None:
    store = TraceStore()
    store.collection.bulk_write.side_effect = error
    with pytest.raises(InfrastructureError) as caught:
        await store.sink.write_traces([make_trace("failed")])
    assert isinstance(caught.value, TransientTraceWriteError) is transient


@pytest.mark.parametrize("same_payload", [True, False], ids=["replay", "conflict"])
async def test_duplicate_key_race_confirms_each_stored_trace(same_payload: bool) -> None:
    """Another writer inserted the same key between our upsert's match and insert."""

    store = TraceStore()
    raced = make_trace("raced")
    other_writer = raced if same_payload else raced.model_copy(update={"user_id": "someone"})
    await store.sink.write_traces([other_writer])
    fresh = make_trace("fresh")

    async def race(operations: list[UpdateOne], *, ordered: bool) -> BulkWriteResult:
        await store.bulk_write([operations[1]], ordered=ordered)
        raise BulkWriteError(
            {"writeErrors": [{"index": 0, "code": 11000}], "writeConcernErrors": []}
        )

    store.collection.bulk_write.side_effect = race
    if same_payload:
        await store.sink.write_traces([raced, fresh])
    else:
        with pytest.raises(TraceConflictError):
            await store.sink.write_traces([raced, fresh])

    assert store.rows["tenant", "raced"]["user_id"] == other_writer.user_id
    assert ("tenant", "fresh") in store.rows
