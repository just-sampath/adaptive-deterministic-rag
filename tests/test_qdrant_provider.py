from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
import pytest
from pymongo import AsyncMongoClient
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from db.initializer import DatabaseInitializer
from db.vector_index import VectorIndexSpec, build_vector_index_spec
from managers.vector import QdrantVectorManager, to_qdrant_sparse
from models.domain import EncodedQuery, HybridHits, Identity, SparseEmbedding
from providers.sparse import Bm25SparseEncoder
from utils.errors import (
    InfrastructureError,
    InvalidRequestError,
    ProviderError,
    ProviderResponseError,
)
from utils.settings import MongoConfig, SparseConfig, VectorStoreConfig

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

_DENSE = "dense"
_SPARSE = "sparse"
_ENCODER = Bm25SparseEncoder(SparseConfig())
_FIRST = "00000000-0000-0000-0000-000000000001"
_SECOND = "00000000-0000-0000-0000-000000000005"
_THIRD = "00000000-0000-0000-0000-000000000006"


def _spec(*, dimension: int = 3, idf: bool = True) -> VectorIndexSpec:
    return build_vector_index_spec(
        VectorStoreConfig(collection_alias="test_chunks"),
        encoder_fingerprint="fake-encoder",
        dense_dimension=dimension,
        sparse_idf_modifier=idf,
    )


def _identity() -> Identity:
    return Identity(tenant_id="tenant-a", user_id="user-a", roles=frozenset({"reader"}))


def _query(sparse_text: str = "supported evidence") -> EncodedQuery:
    return EncodedQuery(
        text=sparse_text,
        dense=(1.0, 0.0, 0.0),
        sparse=_ENCODER.encode_query(sparse_text),
    )


def _point(
    point_id: str,
    *,
    roles: list[str],
    tenant_id: str = "tenant-a",
    dense: tuple[float, float, float] = (1, 0, 0),
) -> models.PointStruct:
    sparse = _ENCODER.encode_documents(["supported evidence"])[0]
    return models.PointStruct(
        id=point_id,
        vector={_DENSE: list(dense), _SPARSE: to_qdrant_sparse(sparse)},
        payload={
            "tenant_id": tenant_id,
            "allowed_roles": roles,
            "document_id": "document",
            "document_version_id": "version",
            "metadata": {"confidence": 1.5},
            "active": True,
        },
    )


def _manager(client: AsyncQdrantClient, spec: VectorIndexSpec) -> QdrantVectorManager:
    return QdrantVectorManager(
        client,
        collection_alias=spec.alias,
        dense_vector_name=spec.dense_vector_name,
        sparse_vector_name=spec.sparse_vector_name,
    )


async def test_manager_searches_through_the_alias_with_tenant_role_and_active_filters() -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _initializer(client, spec)._ensure_qdrant()
    await client.upsert(
        spec.physical_collection,
        points=[
            _point(_FIRST, roles=["reader"]),
            _point("00000000-0000-0000-0000-000000000002", roles=["administrator"]),
            _point(
                "00000000-0000-0000-0000-000000000003",
                roles=["reader"],
                tenant_id="tenant-b",
            ),
        ],
        wait=True,
    )

    hits = await _manager(client, spec).search(
        _query(),
        _identity(),
        {"confidence": 1.5},
        dense_k=10,
        sparse_k=10,
    )

    assert [item.chunk_id for item in hits.dense] == [_FIRST]
    assert [item.chunk_id for item in hits.sparse] == [_FIRST]
    await client.close()


async def test_manager_skips_empty_sparse_vectors_and_zero_budgets() -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _initializer(client, spec)._ensure_qdrant()
    await client.upsert(
        spec.physical_collection,
        points=[_point(_FIRST, roles=["reader"])],
        wait=True,
    )
    manager = _manager(client, spec)
    dense_only = EncodedQuery(
        text="dense only",
        dense=(1.0, 0.0, 0.0),
        sparse=SparseEmbedding(),
    )

    dense_result = await manager.search(dense_only, _identity(), {}, dense_k=10, sparse_k=10)
    empty = await manager.search(_query(), _identity(), {}, dense_k=0, sparse_k=0)

    assert [item.chunk_id for item in dense_result.dense] == [_FIRST]
    assert dense_result.sparse == ()
    assert empty == HybridHits()
    await client.close()


async def test_search_requests_identifiers_and_scores_without_payload_or_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _initializer(client, spec)._ensure_qdrant()
    await client.upsert(
        spec.physical_collection,
        points=[_point(_FIRST, roles=["reader"])],
        wait=True,
    )
    sent: list[models.QueryRequest] = []
    query_batch = client.query_batch_points

    async def recording_query_batch(**kwargs: Any) -> Any:
        sent.extend(kwargs["requests"])
        return await query_batch(**kwargs)

    monkeypatch.setattr(client, "query_batch_points", recording_query_batch)

    hits = await _manager(client, spec).search(_query(), _identity(), {}, dense_k=5, sparse_k=5)

    assert len(sent) == 2
    assert all(request.with_payload is False for request in sent)
    assert all(request.with_vector is False for request in sent)
    assert [item.chunk_id for item in hits.dense] == [_FIRST]
    await client.close()


async def test_hybrid_search_is_one_batched_call_with_one_based_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _initializer(client, spec)._ensure_qdrant()
    # Cosine similarity to the query (1, 0, 0) decreases in this order: FIRST, SECOND, THIRD.
    await client.upsert(
        spec.physical_collection,
        points=[
            _point(_THIRD, roles=["reader"], dense=(1, 1, 1)),
            _point(_FIRST, roles=["reader"], dense=(1, 0, 0)),
            _point(_SECOND, roles=["reader"], dense=(1, 1, 0)),
        ],
        wait=True,
    )
    batch_sizes: list[int] = []
    query_batch = client.query_batch_points

    async def counting_query_batch(**kwargs: Any) -> Any:
        batch_sizes.append(len(kwargs["requests"]))
        return await query_batch(**kwargs)

    monkeypatch.setattr(client, "query_batch_points", counting_query_batch)

    hits = await _manager(client, spec).search(_query(), _identity(), {}, dense_k=5, sparse_k=5)

    assert batch_sizes == [2]
    assert [item.chunk_id for item in hits.dense] == [_FIRST, _SECOND, _THIRD]
    assert [item.rank for item in hits.dense] == [1, 2, 3]
    assert sorted(item.rank for item in hits.sparse) == [1, 2, 3]
    await client.close()


async def test_health_check_probes_the_alias_and_translates_a_missing_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _initializer(client, spec)._ensure_qdrant()
    probed: list[str] = []
    get_collection = client.get_collection

    async def recording_get_collection(collection_name: str, **kwargs: Any) -> Any:
        probed.append(collection_name)
        if collection_name != spec.alias:
            # A server answers an unknown name with HTTP 404; local mode raises ValueError instead.
            raise UnexpectedResponse(404, "Not Found", b"", httpx.Headers())
        return await get_collection(collection_name, **kwargs)

    monkeypatch.setattr(client, "get_collection", recording_get_collection)

    await _manager(client, spec).health_check()
    assert probed == [spec.alias]

    missing = QdrantVectorManager(
        client,
        collection_alias="absent_alias",
        dense_vector_name=_DENSE,
        sparse_vector_name=_SPARSE,
    )
    with pytest.raises(ProviderError, match="Qdrant operation failed"):
        await missing.health_check()
    assert probed == [spec.alias, "absent_alias"]
    await client.close()


async def test_ensure_ready_cancels_qdrant_setup_when_mongo_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    initializer = _initializer(client, _spec())
    qdrant_cancelled: list[bool] = []

    async def failing_mongo() -> None:
        raise InfrastructureError("MongoDB initialization failed")

    async def slow_qdrant() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            qdrant_cancelled.append(True)
            raise

    monkeypatch.setattr(initializer, "_ensure_mongo", failing_mongo)
    monkeypatch.setattr(initializer, "_ensure_qdrant", slow_qdrant)

    with pytest.raises(InfrastructureError, match="MongoDB initialization failed"):
        await initializer.ensure_ready()
    assert qdrant_cancelled == [True]
    await client.close()


def test_manager_rejects_invalid_metadata_key() -> None:
    with pytest.raises(InvalidRequestError, match="Invalid metadata"):
        QdrantVectorManager._filter(_identity(), {"$unsafe": "value"})


async def test_initializer_creates_physical_collection_alias_and_idf_indexes() -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    initializer = _initializer(client, spec)

    await initializer._ensure_qdrant()
    await initializer._ensure_qdrant()

    aliases = await client.get_aliases()
    assert [(item.alias_name, item.collection_name) for item in aliases.aliases] == [
        (spec.alias, spec.physical_collection)
    ]
    info = await client.get_collection(spec.alias)
    assert info.config.params.sparse_vectors is not None
    assert info.config.params.sparse_vectors[_SPARSE].modifier is models.Modifier.IDF
    await client.close()


async def test_initializer_rejects_alias_that_points_at_another_collection() -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _create_collection(client, "other_collection", dimension=3, idf=True)
    await client.update_collection_aliases([_alias("other_collection", spec.alias)])

    with pytest.raises(ProviderResponseError, match="points to other_collection"):
        await _initializer(client, spec)._ensure_qdrant()
    await client.close()


async def test_initializer_rejects_legacy_collection_named_like_the_alias() -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _create_collection(client, spec.alias, dimension=3, idf=True)

    with pytest.raises(ProviderResponseError, match="legacy layout"):
        await _initializer(client, spec)._ensure_qdrant()
    await client.close()


async def test_initializer_rejects_wrong_dimension_in_the_physical_collection() -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec(dimension=3)
    await _create_collection(client, spec.physical_collection, dimension=2, idf=True)

    with pytest.raises(ProviderResponseError, match="vector schema"):
        await _initializer(client, spec)._ensure_qdrant()
    await client.close()


async def test_initializer_rejects_sparse_modifier_that_disagrees_with_the_encoder() -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec(idf=True)
    await _create_collection(client, spec.physical_collection, dimension=3, idf=False)

    with pytest.raises(ProviderResponseError, match="vector schema"):
        await _initializer(client, spec)._ensure_qdrant()
    await client.close()


async def test_initializer_indexes_tenant_role_document_version_section_and_active_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    create_index = client.create_payload_index
    indexed: dict[str, models.PayloadSchemaType] = {}

    async def record(**kwargs: Any) -> Any:
        assert kwargs["collection_name"] == spec.physical_collection
        indexed[kwargs["field_name"]] = kwargs["field_schema"]
        return await create_index(**kwargs)

    monkeypatch.setattr(client, "create_payload_index", record)
    await _initializer(client, spec)._ensure_qdrant()

    assert indexed == {
        "tenant_id": models.PayloadSchemaType.KEYWORD,
        "allowed_roles": models.PayloadSchemaType.KEYWORD,
        "document_id": models.PayloadSchemaType.KEYWORD,
        "document_version_id": models.PayloadSchemaType.KEYWORD,
        "section_id": models.PayloadSchemaType.KEYWORD,
        "active": models.PayloadSchemaType.BOOL,
    }
    await client.close()


def _conflict() -> UnexpectedResponse:
    return UnexpectedResponse(409, "Conflict", b"", httpx.Headers())


async def test_concurrent_creation_of_the_same_physical_collection_is_tolerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    create = client.create_collection

    async def lose_the_race(**kwargs: Any) -> bool:
        await create(**kwargs)  # the concurrent startup creates the collection first...
        raise _conflict()  # ...so this startup receives HTTP 409 for the same name

    monkeypatch.setattr(client, "create_collection", lose_the_race)
    await _initializer(client, spec)._ensure_qdrant()

    aliases = await client.get_aliases()
    assert [(item.alias_name, item.collection_name) for item in aliases.aliases] == [
        (spec.alias, spec.physical_collection)
    ]
    await client.close()


async def test_alias_race_that_lands_on_the_same_physical_collection_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _create_collection(client, spec.physical_collection, dimension=3, idf=True)
    attach = client.update_collection_aliases

    async def lose_the_race(operations: Any) -> bool:
        await attach(operations)
        raise _conflict()

    monkeypatch.setattr(client, "update_collection_aliases", lose_the_race)
    await _initializer(client, spec)._ensure_qdrant()

    aliases = await client.get_aliases()
    assert [(item.alias_name, item.collection_name) for item in aliases.aliases] == [
        (spec.alias, spec.physical_collection)
    ]
    await client.close()


async def test_alias_race_that_lands_on_another_collection_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    spec = _spec()
    await _create_collection(client, "other_collection", dimension=3, idf=True)
    attach = client.update_collection_aliases

    async def lose_the_race(operations: Any) -> bool:
        del operations
        await attach([_alias("other_collection", spec.alias)])
        raise _conflict()

    monkeypatch.setattr(client, "update_collection_aliases", lose_the_race)

    with pytest.raises(ProviderResponseError, match="created concurrently"):
        await _initializer(client, spec)._ensure_qdrant()
    await client.close()


def _physical_collection(
    *,
    fingerprint: str = "fake-encoder",
    dimension: int = 3,
    idf: bool = True,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
) -> str:
    config = VectorStoreConfig(
        collection_alias="test_chunks",
        dense_vector_name=dense_name,
        sparse_vector_name=sparse_name,
    )
    return build_vector_index_spec(
        config,
        encoder_fingerprint=fingerprint,
        dense_dimension=dimension,
        sparse_idf_modifier=idf,
    ).physical_collection


def test_physical_collection_name_is_stable_for_an_identical_layout() -> None:
    assert _physical_collection() == _physical_collection()


@pytest.mark.parametrize(
    "change",
    [
        {"fingerprint": "other-encoder"},
        {"dimension": 4},
        {"idf": False},
        {"dense_name": "dense_v2"},
        {"sparse_name": "sparse_v2"},
    ],
    ids=["encoder", "dimension", "idf", "dense-name", "sparse-name"],
)
def test_physical_collection_name_changes_with_each_layout_input(change: dict[str, Any]) -> None:
    assert _physical_collection(**change) != _physical_collection()


def _initializer(client: AsyncQdrantClient, spec: VectorIndexSpec) -> DatabaseInitializer:
    mongo = cast(AsyncMongoClient[dict[str, Any]], object())
    return DatabaseInitializer(mongo, client, mongo=MongoConfig(), index=spec)


async def _create_collection(
    client: AsyncQdrantClient,
    name: str,
    *,
    dimension: int,
    idf: bool,
) -> None:
    await client.create_collection(
        name,
        vectors_config={
            _DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            _SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF if idf else None,
            )
        },
    )


def _alias(collection: str, alias: str) -> models.CreateAliasOperation:
    return models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
    )
