from __future__ import annotations

import math
import traceback
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from bson import Binary, Code, Int64
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
        self.queries: list[dict[str, Any]] = []
        self.projections: list[dict[str, int]] = []

    def find(self, query: dict[str, Any], projection: dict[str, int]) -> FakeCursor:
        self.queries.append(query)
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
    assert database.chunks.queries == [{"tenant_id": identity.tenant_id, "id": {"$in": [chunk.id]}}]
    assert database.documents.queries == [
        {
            "tenant_id": identity.tenant_id,
            "id": {"$in": [chunk.document_id]},
            "deleted_at": None,
            "current_version_id": {"$ne": None},
        }
    ]
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


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(250.0, 250), (Int64(2**53 + 1), 2**53 + 1)],
)
async def test_positive_whole_bson_token_counts_hydrate_as_exact_integers(
    identity: Identity, stored: int | float, expected: int
) -> None:
    chunk = make_chunk("page")
    record = {**chunk.model_dump(), "token_count": stored}
    current = {"id": chunk.document_id, "current_version_id": chunk.document_version_id}
    manager = make_manager(FakeDatabase(chunks=[record], documents=[current]))

    hydrated = await manager.hydrate_authorized_chunks([chunk.id], identity)

    count = hydrated[chunk.id].token_count
    assert type(count) is int
    assert count == expected


@pytest.mark.parametrize("count", [0, -1, 0.0, -2.0, True, 2.5, "2", math.nan, math.inf, -math.inf])
async def test_authoritative_token_count_must_be_a_positive_integer(
    identity: Identity, count: object
) -> None:
    chunk = make_chunk("page")
    manager = make_manager(FakeDatabase(chunks=[{**chunk.model_dump(), "token_count": count}]))
    with pytest.raises(ProviderResponseError, match="invalid chunk"):
        await manager.hydrate_authorized_chunks([chunk.id], identity)


async def test_metadata_preserves_bson_scalar_types(identity: Identity) -> None:
    chunk = make_chunk("fixture")
    current = [{"id": chunk.document_id, "current_version_id": chunk.document_version_id}]

    def with_value(value: object) -> MongoDocumentManager:
        record = {**chunk.model_dump(), "metadata": {"value": value}}
        return make_manager(FakeDatabase(chunks=[record], documents=current))

    hydrated = await with_value(Int64(3)).hydrate_authorized_chunks([chunk.id], identity)
    assert type(hydrated[chunk.id].metadata["value"]) is int
    sample = "sample-typed-field"
    for unsupported in (Binary(sample.encode()), Code(sample)):
        with pytest.raises(ProviderResponseError, match="invalid chunk") as failure:
            await with_value(unsupported).hydrate_authorized_chunks([chunk.id], identity)
        assert sample not in "".join(traceback.format_exception(failure.value))


@pytest.mark.parametrize("mismatch", ["role", "current_version"])
async def test_authoritative_rows_require_current_version_and_role(
    identity: Identity, mismatch: str
) -> None:
    roles = frozenset({"editor"}) if mismatch == "role" else identity.roles
    chunk = make_chunk("page", roles=roles)
    current = "revised-version" if mismatch == "current_version" else chunk.document_version_id
    manager = make_manager(
        FakeDatabase(
            chunks=[chunk.model_dump()],
            documents=[{"id": chunk.document_id, "current_version_id": current}],
        )
    )
    assert await manager.hydrate_authorized_chunks([chunk.id], identity) == {}
