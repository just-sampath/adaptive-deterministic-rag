"""MongoDB startup: a longer TTL applies in place; a shorter one or another conflict fails."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import pytest
from pymongo import AsyncMongoClient, IndexModel
from pymongo.errors import OperationFailure
from qdrant_client import AsyncQdrantClient

from db.initializer import DatabaseInitializer
from db.vector_index import build_vector_index_spec
from utils.errors import InfrastructureError
from utils.settings import MongoConfig, VectorStoreConfig

_DAY = 86_400
_TRACE_TTL = "ttl_traces_created_at"
_EVENT_TTL = "ttl_telemetry_created_at"


class _Cursor:
    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self._items = list(items)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[dict[str, Any]]:
        for item in self._items:
            yield item


class _Collection:
    """Keeps indexes by name and rejects a same-name index with other options, as MongoDB does."""

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}

    async def list_indexes(self) -> _Cursor:
        return _Cursor(list(self.indexes.values()))

    async def create_indexes(self, models: Sequence[IndexModel]) -> None:
        for model in models:
            wanted = dict(model.document)
            current = self.indexes.get(wanted["name"])
            if current is not None and current != wanted:
                raise OperationFailure("Index already exists with different options", code=85)
            self.indexes[wanted["name"]] = wanted


class _Database:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}
        self.collmods: list[tuple[str, dict[str, Any]]] = []

    def __getitem__(self, name: str) -> _Collection:
        return self.collections.setdefault(name, _Collection())

    async def create_collection(self, name: str) -> None:
        self[name]

    async def command(self, command: str, collection: str, *, index: dict[str, Any]) -> None:
        assert command == "collMod"
        self.collmods.append((collection, index))
        self[collection].indexes[index["name"]]["expireAfterSeconds"] = index["expireAfterSeconds"]


class _Admin:
    async def command(self, command: str) -> None:
        assert command == "ping"


class _Client:
    def __init__(self, database: _Database) -> None:
        self._database = database
        self.admin = _Admin()

    def __getitem__(self, name: str) -> _Database:
        return self._database


def _ttl(name: str, days: int, key: str = "created_at") -> dict[str, Any]:
    return {"name": name, "key": {key: 1}, "expireAfterSeconds": days * _DAY}


async def _start(database: _Database, *, trace_days: int, event_days: int) -> None:
    config = MongoConfig(trace_retention_days=trace_days, telemetry_retention_days=event_days)
    spec = build_vector_index_spec(
        VectorStoreConfig(),
        encoder_fingerprint="fake",
        dense_dimension=4,
        sparse_idf_modifier=True,
    )
    initializer = DatabaseInitializer(
        cast(AsyncMongoClient[dict[str, Any]], _Client(database)),
        cast(AsyncQdrantClient, object()),
        mongo=config,
        index=spec,
    )
    await initializer._ensure_mongo()


async def test_longer_retention_updates_the_ttl_indexes_in_place(
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = _Database()
    database["rag_request_traces"].indexes[_TRACE_TTL] = _ttl(_TRACE_TTL, 7)
    database["telemetry_events"].indexes[_EVENT_TTL] = _ttl(_EVENT_TTL, 14)

    with caplog.at_level(logging.WARNING, logger="db.initializer"):
        await _start(database, trace_days=30, event_days=30)

    assert database.collmods == [
        ("rag_request_traces", {"name": _TRACE_TTL, "expireAfterSeconds": 30 * _DAY}),
        ("telemetry_events", {"name": _EVENT_TTL, "expireAfterSeconds": 30 * _DAY}),
    ]
    assert database["rag_request_traces"].indexes[_TRACE_TTL]["expireAfterSeconds"] == 30 * _DAY
    assert database["telemetry_events"].indexes[_EVENT_TTL]["expireAfterSeconds"] == 30 * _DAY
    assert [record.levelno for record in caplog.records] == [logging.WARNING, logging.WARNING]


async def test_shorter_retention_fails_startup_without_deleting_records() -> None:
    database = _Database()
    database["rag_request_traces"].indexes[_TRACE_TTL] = _ttl(_TRACE_TTL, 30)

    with pytest.raises(InfrastructureError, match="Shortening retention deletes stored records"):
        await _start(database, trace_days=7, event_days=30)

    assert database.collmods == []
    assert database["rag_request_traces"].indexes[_TRACE_TTL] == _ttl(_TRACE_TTL, 30)


@pytest.mark.parametrize(
    "existing",
    [
        pytest.param(False, id="first-startup"),
        pytest.param(True, id="unchanged-retention"),
    ],
)
async def test_unchanged_or_new_ttl_indexes_need_no_update(existing: bool) -> None:
    database = _Database()
    if existing:
        database["rag_request_traces"].indexes[_TRACE_TTL] = _ttl(_TRACE_TTL, 30)
        database["telemetry_events"].indexes[_EVENT_TTL] = _ttl(_EVENT_TTL, 30)

    await _start(database, trace_days=30, event_days=30)

    assert database.collmods == []
    assert database["rag_request_traces"].indexes[_TRACE_TTL] == _ttl(_TRACE_TTL, 30)
    assert database["telemetry_events"].indexes[_EVENT_TTL] == _ttl(_EVENT_TTL, 30)


async def test_ttl_index_with_another_key_still_fails_startup() -> None:
    database = _Database()
    database["rag_request_traces"].indexes[_TRACE_TTL] = _ttl(_TRACE_TTL, 30, key="expires_at")

    with pytest.raises(InfrastructureError, match="MongoDB initialization failed"):
        await _start(database, trace_days=7, event_days=30)

    assert database.collmods == []
