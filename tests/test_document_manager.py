from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pymongo.asynchronous.database import AsyncDatabase

from managers.documents import MongoDocumentManager
from models.domain import Chunk, Identity
from tests.fakes import make_chunk
from utils.errors import ProviderResponseError


def apply_inclusion_projection(
    document: dict[str, Any],
    projection: dict[str, int],
) -> dict[str, Any]:
    """Keep only the fields a MongoDB inclusion projection selects."""

    included = {key for key, value in projection.items() if value == 1}
    return {key: value for key, value in document.items() if key in included}


class FakeCursor:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._iterator: AsyncIterator[dict[str, Any]] = self._iterate(items)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iterator

    @staticmethod
    async def _iterate(items: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        for item in items:
            yield item


class FakeCollection:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self.projections: list[dict[str, int]] = []

    def find(self, query: dict[str, Any], projection: dict[str, int]) -> FakeCursor:
        del query
        self.projections.append(projection)
        return FakeCursor([apply_inclusion_projection(item, projection) for item in self._items])


class FakeDatabase:
    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]] | None = None,
        documents: list[dict[str, Any]] | None = None,
    ) -> None:
        self.chunks = FakeCollection(chunks if chunks is not None else [{"id": "malformed"}])
        self.documents = FakeCollection(documents or [])

    def __getitem__(self, name: str) -> FakeCollection:
        return self.chunks if name == "chunks" else self.documents


def make_manager(database: FakeDatabase) -> MongoDocumentManager:
    return MongoDocumentManager(cast(AsyncDatabase[dict[str, Any]], database))


async def test_document_manager_rejects_malformed_mongo_chunk() -> None:
    manager = make_manager(FakeDatabase())
    identity = Identity(tenant_id="tenant", user_id="user", roles=frozenset({"reader"}))

    with pytest.raises(ProviderResponseError, match="invalid chunk"):
        await manager.hydrate_authorized_chunks(["malformed"], identity)


async def test_document_manager_rejects_malformed_current_document() -> None:
    chunk = make_chunk("00000000-0000-0000-0000-000000000001")
    manager = make_manager(
        FakeDatabase(
            chunks=[chunk.model_dump(mode="python")],
            documents=[{"id": chunk.document_id}],
        )
    )
    identity = Identity(tenant_id="tenant-a", user_id="user", roles=frozenset({"reader"}))

    with pytest.raises(ProviderResponseError, match="invalid document"):
        await manager.hydrate_authorized_chunks([chunk.id], identity)


async def test_document_manager_projects_only_chunk_model_fields() -> None:
    chunk = make_chunk("chunk-1")
    stored_chunk = {
        **chunk.model_dump(mode="python"),
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    database = FakeDatabase(
        chunks=[stored_chunk],
        documents=[
            {
                "id": chunk.document_id,
                "current_version_id": chunk.document_version_id,
                "title": "stored but not projected",
            }
        ],
    )
    manager = make_manager(database)
    identity = Identity(tenant_id="tenant-a", user_id="user", roles=frozenset({"reader"}))

    hydrated = await manager.hydrate_authorized_chunks([chunk.id], identity)

    assert hydrated == {chunk.id: chunk}
    (chunk_projection,) = database.chunks.projections
    assert set(chunk_projection) - {"_id"} == set(Chunk.model_fields)
    assert chunk_projection["_id"] == 0


async def test_document_manager_rechecks_tenant_when_store_returns_foreign_chunk() -> None:
    foreign = make_chunk("chunk-1", tenant_id="tenant-b")
    manager = make_manager(
        FakeDatabase(
            chunks=[foreign.model_dump(mode="python")],
            documents=[
                {
                    "id": foreign.document_id,
                    "current_version_id": foreign.document_version_id,
                }
            ],
        )
    )
    identity = Identity(tenant_id="tenant-a", user_id="user", roles=frozenset({"reader"}))

    assert await manager.hydrate_authorized_chunks([foreign.id], identity) == {}
