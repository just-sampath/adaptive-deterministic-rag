"""Hybrid Qdrant search through the stable collection alias with immutable filters."""

from __future__ import annotations

import re
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from managers.errors import qdrant_errors
from models.domain import (
    FILTER_KEY_PATTERN,
    EncodedQuery,
    HybridHits,
    Identity,
    SearchHit,
    SparseEmbedding,
)
from utils.errors import InvalidRequestError

_METADATA_KEY = re.compile(FILTER_KEY_PATTERN)


class QdrantVectorManager:
    """Search Qdrant with immutable tenant, role, and active-version filters."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        collection_alias: str,
        dense_vector_name: str,
        sparse_vector_name: str,
    ) -> None:
        self._client = client
        self._collection_alias = collection_alias
        self._dense_vector_name = dense_vector_name
        self._sparse_vector_name = sparse_vector_name

    @qdrant_errors
    async def search(
        self,
        query: EncodedQuery,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        *,
        dense_k: int,
        sparse_k: int,
    ) -> HybridHits:
        query_filter = self._filter(identity, filters)
        dense_request = self._dense_request(query, query_filter, dense_k) if dense_k > 0 else None
        sparse_request = (
            self._sparse_request(query, query_filter, sparse_k)
            if sparse_k > 0 and query.sparse.indices
            else None
        )
        requests = [request for request in (dense_request, sparse_request) if request is not None]
        if not requests:
            return HybridHits()
        responses = iter(
            await self._client.query_batch_points(
                collection_name=self._collection_alias,
                requests=requests,
            )
        )
        # Responses follow request order: dense first, then sparse.
        return HybridHits(
            dense=_hits(next(responses)) if dense_request is not None else (),
            sparse=_hits(next(responses)) if sparse_request is not None else (),
        )

    @qdrant_errors
    async def health_check(self) -> None:
        await self._client.get_collection(self._collection_alias)

    def _dense_request(
        self,
        query: EncodedQuery,
        query_filter: models.Filter,
        limit: int,
    ) -> models.QueryRequest:
        return models.QueryRequest(
            query=list(query.dense),
            using=self._dense_vector_name,
            filter=query_filter,
            limit=limit,
            with_payload=False,
            with_vector=False,
        )

    def _sparse_request(
        self,
        query: EncodedQuery,
        query_filter: models.Filter,
        limit: int,
    ) -> models.QueryRequest:
        return models.QueryRequest(
            query=to_qdrant_sparse(query.sparse),
            using=self._sparse_vector_name,
            filter=query_filter,
            limit=limit,
            with_payload=False,
            with_vector=False,
        )

    @staticmethod
    def _filter(
        identity: Identity,
        filters: dict[str, str | int | float | bool],
    ) -> models.Filter:
        conditions: list[Any] = [
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=identity.tenant_id)
            ),
            models.FieldCondition(
                key="allowed_roles", match=models.MatchAny(any=sorted(identity.roles))
            ),
            models.FieldCondition(key="active", match=models.MatchValue(value=True)),
        ]
        for key, value in filters.items():
            if not _METADATA_KEY.fullmatch(key):
                raise InvalidRequestError(f"Invalid metadata filter key: {key!r}")
            payload_key = f"metadata.{key}"
            if isinstance(value, float):
                condition = models.FieldCondition(
                    key=payload_key, range=models.Range(gte=value, lte=value)
                )
            else:
                condition = models.FieldCondition(
                    key=payload_key, match=models.MatchValue(value=value)
                )
            conditions.append(condition)
        return models.Filter(must=conditions)


def to_qdrant_sparse(embedding: SparseEmbedding) -> models.SparseVector:
    """Translate the provider-neutral sparse vector into the Qdrant wire model."""

    return models.SparseVector(indices=list(embedding.indices), values=list(embedding.values))


def _hits(response: models.QueryResponse) -> tuple[SearchHit, ...]:
    return tuple(
        SearchHit(chunk_id=str(point.id), score=point.score, rank=rank)
        for rank, point in enumerate(response.points, start=1)
    )
