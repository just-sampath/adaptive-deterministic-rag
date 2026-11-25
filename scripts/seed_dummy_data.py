"""Seed one fixed local-development document into MongoDB and Qdrant.

Seeding the same handbook again converges on one logical corpus revision. Missing rows or points
are restored without a new revision, and ``system_state.corpus_version`` advances once, when the
first seed of this fixture completes. A tenant row with ``local_seed`` records that completion.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError
from qdrant_client import AsyncQdrantClient, models

from db.collections import CHUNKS, DOCUMENT_VERSIONS, DOCUMENTS, SYSTEM_STATE
from db.initializer import DatabaseInitializer
from db.vector_index import VectorIndexSpec, build_vector_index_spec
from managers.vector import to_qdrant_sparse
from models.domain import Chunk, Identity
from providers.mongodb import build_mongo_client
from providers.openai import OpenAIEmbeddingProvider, build_openai_client
from providers.qdrant import build_qdrant_client
from providers.sparse import Bm25SparseEncoder
from services.contracts import EmbeddingProvider
from services.encoding import EmbeddingCache, HybridQueryEncoder
from utils.errors import ConfigurationError
from utils.settings import Environment, Settings, load_settings

_TITLE = "Adaptive RAG Local Handbook"
# Bump when the fixture's stored shape changes, so the fingerprint stops matching old records.
_FIXTURE_SCHEMA = 1
logger = logging.getLogger(__name__)
_TEXTS = (
    (
        ("Deterministic search",),
        "The backend tries a fixed, bounded sequence of retrieval strategies and abstains "
        "when evidence is insufficient.",
    ),
    (
        ("Observability",),
        "Request telemetry and detailed RAG traces are stored in MongoDB. Provider usage "
        "and request identifiers are retained when returned by the provider.",
    ),
)

Document = dict[str, Any]


class LegacyCounter(StrEnum):
    """How to treat a handbook an older script seeded without recording completion."""

    ADOPT = "adopt"  # it was already counted: keep corpus_version as it is
    COUNT = "count"  # it was interrupted before counting: advance corpus_version once


class SeedConflictError(ConfigurationError):
    """The fixed IDs or the tenant's seed record hold something other than this handbook."""


class SeedReconciliationRequiredError(ConfigurationError):
    """Existing handbook records and an unmarked counter cannot show whether it was counted."""


@dataclass(frozen=True, slots=True)
class SeedFixture:
    """The exact records one seed of the handbook writes for a tenant."""

    tenant_id: str
    document: Document
    version: Document
    chunks: tuple[Chunk, ...]
    fingerprint: str

    @property
    def document_id(self) -> str:
        return str(self.document["id"])

    @property
    def version_id(self) -> str:
        return str(self.version["id"])

    def chunk_row(self, chunk: Chunk) -> Document:
        row: Document = chunk.model_dump(mode="python")
        row["allowed_roles"] = sorted(chunk.allowed_roles)
        row["heading_path"] = list(chunk.heading_path)
        return row

    def payload(self, chunk: Chunk) -> Document:
        return {
            "tenant_id": chunk.tenant_id,
            "allowed_roles": sorted(chunk.allowed_roles),
            "document_id": chunk.document_id,
            "document_version_id": chunk.document_version_id,
            "section_id": chunk.section_id,
            "metadata": chunk.metadata,
            "active": True,
        }


@dataclass(frozen=True, slots=True)
class _Footprint:
    """Which of the fixture's fixed-ID records already exist."""

    document: bool
    version: bool
    chunks: frozenset[str]
    points: frozenset[str]

    @property
    def empty(self) -> bool:
        return not (self.document or self.version or self.chunks or self.points)

    def complete(self, fixture: SeedFixture) -> bool:
        ids = {chunk.id for chunk in fixture.chunks}
        return self.document and self.version and self.chunks == ids and self.points == ids


def build_fixture(tenant_id: str) -> SeedFixture:
    document_id = str(uuid5(NAMESPACE_URL, f"adaptive-rag:{tenant_id}:dummy"))
    version_id = str(uuid5(NAMESPACE_URL, f"{document_id}:version:1"))
    document = {
        "id": document_id,
        "tenant_id": tenant_id,
        "title": _TITLE,
        "source_type": "dummy",
        "source_uri": None,
        "metadata": {"seed": "local"},
        "current_version_id": version_id,
        "last_version_number": 1,
        "deleted_at": None,
    }
    version = {
        "id": version_id,
        "document_id": document_id,
        "tenant_id": tenant_id,
        "version": 1,
        "checksum": "local-dummy-v1",
        "active": True,
    }
    chunks = _chunks(tenant_id, document_id, version_id)
    unsigned = SeedFixture(tenant_id, document, version, chunks, fingerprint="")
    content = {
        "schema": _FIXTURE_SCHEMA,
        "document": document,
        "version": version,
        "chunks": [unsigned.chunk_row(chunk) for chunk in chunks],
        "points": [unsigned.payload(chunk) for chunk in chunks],
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return SeedFixture(tenant_id, document, version, chunks, fingerprint=fingerprint)


async def seed_dummy_data(
    settings: Settings,
    identity: Identity,
    *,
    legacy_counter: LegacyCounter | None = None,
) -> dict[str, str | int]:
    """Write the fixed dummy document; this is the repository's only vector writer."""

    if settings.external.app_env is not Environment.LOCAL:
        raise ConfigurationError("Dummy data can only be seeded in APP_ENV=local")
    if settings.ai.provider != "openai":
        raise ConfigurationError("The dummy seed script currently requires the OpenAI provider")
    assert settings.external.openai_api_key is not None

    mongo_client = build_mongo_client(
        uri=settings.external.mongodb_uri.get_secret_value(),
        server_selection_timeout_ms=settings.mongodb.server_selection_timeout_ms,
        operation_timeout_ms=settings.mongodb.operation_timeout_ms,
    )
    qdrant_client = build_qdrant_client(
        url=settings.external.qdrant_url,
        api_key=(
            settings.external.qdrant_api_key.get_secret_value()
            if settings.external.qdrant_api_key
            else None
        ),
        timeout_seconds=settings.timeouts.retrieval_seconds,
    )
    ai_client = build_openai_client(
        api_key=settings.external.openai_api_key.get_secret_value(),
        base_url=settings.external.openai_base_url,
        max_retries=settings.ai.max_retries,
        timeout_seconds=settings.timeouts.generation_seconds,
    )
    try:
        embeddings = OpenAIEmbeddingProvider(
            ai_client,
            model=settings.ai.embedding_model,
            dimension=settings.ai.embedding_dimension,
        )
        sparse = Bm25SparseEncoder(settings.vector_store.sparse)
        encoder = HybridQueryEncoder(
            dense=embeddings,
            sparse=sparse,
            cache=EmbeddingCache(0),
            embedding_timeout_seconds=settings.timeouts.embedding_seconds,
        )
        index = build_vector_index_spec(
            settings.vector_store,
            encoder_fingerprint=encoder.fingerprint,
            dense_dimension=encoder.dense_dimension,
            sparse_idf_modifier=encoder.requires_idf_modifier,
        )
        await DatabaseInitializer(
            mongo_client,
            qdrant_client,
            mongo=settings.mongodb,
            index=index,
        ).ensure_ready()
        return await seed_fixture(
            mongo_client[settings.mongodb.database],
            qdrant_client,
            index=index,
            embeddings=embeddings,
            sparse=sparse,
            fixture=build_fixture(identity.tenant_id),
            legacy_counter=legacy_counter,
        )
    finally:
        close_results = await asyncio.gather(
            ai_client.close(),
            qdrant_client.close(),
            mongo_client.close(),
            return_exceptions=True,
        )
        for result in close_results:
            if isinstance(result, BaseException):
                logger.error(
                    "Dummy seed resource shutdown failed",
                    exc_info=(type(result), result, result.__traceback__),
                )


async def seed_fixture(
    database: AsyncDatabase[Document],
    qdrant: AsyncQdrantClient,
    *,
    index: VectorIndexSpec,
    embeddings: EmbeddingProvider,
    sparse: Bm25SparseEncoder,
    fixture: SeedFixture,
    legacy_counter: LegacyCounter | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, str | int]:
    """Converge the stores on the fixture and record its completion once.

    Nothing is written until the tenant's seed record and every fixed-ID record have been checked.
    Rows are insert-only, so a rerun keeps their creation times, and only missing points are
    embedded and written. ``corpus_version`` moves in the same atomic update that marks the seed
    complete, so a retry after that update cannot count it twice.
    """

    state_rows = database[SYSTEM_STATE]
    # Records first, then the seed record: a concurrent run reserves before it writes, so any
    # records seen here were written under a reservation the state read will also see.
    footprint = await _footprint(database, qdrant, index, fixture)
    state = await state_rows.find_one({"tenant_id": fixture.tenant_id}, {"_id": 0})
    marker = _marker(state, fixture)
    if marker is None:
        counter_exists = state is not None and "corpus_version" in state
        if not footprint.empty and counter_exists:
            if legacy_counter is None:
                raise SeedReconciliationRequiredError(
                    "The local handbook is already partly or fully stored, but corpus_version "
                    "has no record of whether that seed was counted. Rerun with "
                    "--legacy-counter adopt if it was counted, or --legacy-counter count if it "
                    "was interrupted before counting."
                )
            count = legacy_counter is LegacyCounter.COUNT
            resolution: str | None = legacy_counter.value
        else:
            count, resolution = True, None
        marker = await _reserve(
            state_rows, fixture, count=count, resolution=resolution, now=clock()
        )

    now = clock()
    for collection, row in (
        (DOCUMENTS, fixture.document),
        (DOCUMENT_VERSIONS, fixture.version),
        *((CHUNKS, fixture.chunk_row(chunk)) for chunk in fixture.chunks),
    ):
        key = {"tenant_id": fixture.tenant_id, "id": row["id"]}
        inserted = {name: value for name, value in row.items() if name not in key}
        # The upsert copies the key from the filter. If a concurrent identical run inserts the
        # row first, the reread below compares whatever it stored.
        with suppress(DuplicateKeyError):
            await database[collection].update_one(
                key, {"$setOnInsert": {**inserted, "created_at": now}}, upsert=True
            )
    missing = [chunk for chunk in fixture.chunks if chunk.id not in footprint.points]
    if missing:
        await _write_points(qdrant, index, embeddings, sparse, fixture, missing)
    if not (await _footprint(database, qdrant, index, fixture)).complete(fixture):
        raise SeedConflictError("The local handbook could not be fully stored; nothing was counted")
    if marker["status"] != "completed":
        await _complete(state_rows, fixture, count=bool(marker["count_on_completion"]), now=now)
    return {
        "document_id": fixture.document_id,
        "document_version_id": fixture.version_id,
        "chunks_indexed": len(fixture.chunks),
    }


def _marker(state: Mapping[str, Any] | None, fixture: SeedFixture) -> Mapping[str, Any] | None:
    marker = state.get("local_seed") if state else None
    if marker is not None and marker.get("fingerprint") != fixture.fingerprint:
        raise SeedConflictError("This tenant's seed record belongs to different handbook content")
    return marker


async def _reserve(
    state_rows: AsyncCollection[Document],
    fixture: SeedFixture,
    *,
    count: bool,
    resolution: str | None,
    now: datetime,
) -> Mapping[str, Any]:
    """Claim the tenant for this fixture, or adopt an identical claim another run made first."""

    marker = {
        "fingerprint": fixture.fingerprint,
        "status": "reserved",
        "count_on_completion": count,
        "legacy_counter": resolution,
        "reserved_at": now,
    }
    # A DuplicateKeyError means another run created the tenant row first; its claim is read back.
    with suppress(DuplicateKeyError):
        await state_rows.update_one(
            {"tenant_id": fixture.tenant_id, "local_seed": {"$exists": False}},
            {"$set": {"local_seed": marker}, "$setOnInsert": {"tenant_id": fixture.tenant_id}},
            upsert=True,
        )
    state = await state_rows.find_one({"tenant_id": fixture.tenant_id}, {"_id": 0})
    stored = _marker(state, fixture)
    if stored is None:
        raise SeedConflictError("The tenant's seed record could not be reserved")
    return stored


async def _complete(
    state_rows: AsyncCollection[Document], fixture: SeedFixture, *, count: bool, now: datetime
) -> None:
    update: Document = {
        "$set": {
            "local_seed.status": "completed",
            "local_seed.completed_at": now,
            "updated_at": now,
        }
    }
    if count:
        update["$inc"] = {"corpus_version": 1}
    result = await state_rows.update_one(
        {
            "tenant_id": fixture.tenant_id,
            "local_seed.fingerprint": fixture.fingerprint,
            "local_seed.status": "reserved",
        },
        update,
    )
    if result.matched_count:
        return
    # A concurrent identical run, or this run before a lost acknowledgment, completed it already.
    state = await state_rows.find_one({"tenant_id": fixture.tenant_id}, {"_id": 0})
    marker = _marker(state, fixture)
    if marker is None or marker.get("status") != "completed":
        raise SeedConflictError("The tenant's seed record changed while the handbook was stored")


async def _footprint(
    database: AsyncDatabase[Document],
    qdrant: AsyncQdrantClient,
    index: VectorIndexSpec,
    fixture: SeedFixture,
) -> _Footprint:
    """Read the fixture's fixed-ID records and refuse any that hold other content."""

    tenant = fixture.tenant_id
    document = await database[DOCUMENTS].find_one({"tenant_id": tenant, "id": fixture.document_id})
    version = await database[DOCUMENT_VERSIONS].find_one(
        {"tenant_id": tenant, "id": fixture.version_id}
    )
    expected_chunks = {chunk.id: fixture.chunk_row(chunk) for chunk in fixture.chunks}
    stored_chunks = [
        row
        async for row in database[CHUNKS].find(
            {"tenant_id": tenant, "id": {"$in": list(expected_chunks)}}
        )
    ]
    _require_match(document, fixture.document, "document")
    _require_match(version, fixture.version, "document version")
    for row in stored_chunks:
        _require_match(row, expected_chunks[row["id"]], "chunk")
    points = await qdrant.retrieve(
        collection_name=index.physical_collection,
        ids=list(expected_chunks),
        with_payload=True,
        with_vectors=False,
    )
    payloads = {chunk.id: fixture.payload(chunk) for chunk in fixture.chunks}
    for point in points:
        if point.payload != payloads[str(point.id)]:
            raise SeedConflictError("A local handbook point holds different payload")
    return _Footprint(
        document=document is not None,
        version=version is not None,
        chunks=frozenset(row["id"] for row in stored_chunks),
        points=frozenset(str(point.id) for point in points),
    )


def _require_match(
    stored: Mapping[str, Any] | None, expected: Mapping[str, Any], kind: str
) -> None:
    if stored is not None and any(stored.get(key) != value for key, value in expected.items()):
        raise SeedConflictError(
            f"A local handbook {kind} with the fixed ID holds different content"
        )


async def _write_points(
    qdrant: AsyncQdrantClient,
    index: VectorIndexSpec,
    embeddings: EmbeddingProvider,
    sparse: Bm25SparseEncoder,
    fixture: SeedFixture,
    chunks: Sequence[Chunk],
) -> None:
    texts = [chunk.text for chunk in chunks]
    dense_results = await embeddings.embed_documents(texts)
    sparse_vectors = sparse.encode_documents(texts)
    await qdrant.upsert(
        collection_name=index.physical_collection,
        points=[
            models.PointStruct(
                id=chunk.id,
                vector={
                    index.dense_vector_name: dense.vector,
                    index.sparse_vector_name: to_qdrant_sparse(sparse_vector),
                },
                payload=fixture.payload(chunk),
            )
            for chunk, dense, sparse_vector in zip(
                chunks, dense_results, sparse_vectors, strict=True
            )
        ],
        wait=True,
    )


def _chunks(tenant_id: str, document_id: str, version_id: str) -> tuple[Chunk, ...]:
    allowed_roles = frozenset({"reader", "administrator"})
    chunk_ids = [
        str(uuid5(NAMESPACE_URL, f"{version_id}:chunk:{sequence}"))
        for sequence in range(len(_TEXTS))
    ]
    chunks: list[Chunk] = []
    for sequence, (heading, text) in enumerate(_TEXTS):
        section_id = str(uuid5(NAMESPACE_URL, f"{version_id}:section:{heading[0]}"))
        content = f"Document: {_TITLE}\nSection: {heading[0]}\nVersion: 1\n\n{text}"
        chunks.append(
            Chunk(
                id=chunk_ids[sequence],
                tenant_id=tenant_id,
                document_id=document_id,
                document_version_id=version_id,
                document_version=1,
                document_title=_TITLE,
                section_id=section_id,
                sequence=sequence,
                heading_path=heading,
                text=content,
                token_count=max(1, (len(content) + 3) // 4),
                previous_chunk_id=chunk_ids[sequence - 1] if sequence else None,
                next_chunk_id=(chunk_ids[sequence + 1] if sequence + 1 < len(chunk_ids) else None),
                allowed_roles=allowed_roles,
                metadata={"seed": "local"},
            )
        )
    return tuple(chunks)


async def _main(tenant_id: str, user_id: str, legacy_counter: LegacyCounter | None) -> None:
    settings = load_settings()
    result = await seed_dummy_data(
        settings,
        Identity(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=frozenset({"administrator"}),
        ),
        legacy_counter=legacy_counter,
    )
    print(result)  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", default="local-tenant")
    parser.add_argument("--user-id", default="local-admin")
    parser.add_argument(
        "--legacy-counter",
        choices=[item.value for item in LegacyCounter],
        default=None,
        help=(
            "Only needed when an older script already stored the handbook: 'adopt' keeps "
            "corpus_version as it is, 'count' advances it once."
        ),
    )
    args = parser.parse_args()
    legacy = LegacyCounter(args.legacy_counter) if args.legacy_counter else None
    asyncio.run(_main(args.tenant_id, args.user_id, legacy))


if __name__ == "__main__":
    main()
