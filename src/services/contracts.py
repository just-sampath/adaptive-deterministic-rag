"""Small ports consumed by application services."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel

from models.domain import (
    Chunk,
    ConversationMessage,
    EmbeddingResult,
    EncodedQuery,
    FollowupDecision,
    HybridHits,
    Identity,
    ModelResult,
    RankedChunk,
    RequestTrace,
    SparseEmbedding,
    TelemetryEvent,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
TelemetryValue = str | int | float | bool | None


class AsyncCloseable(Protocol):
    async def close(self) -> None: ...


class TextModel(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]: ...


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    async def embed_query(self, text: str) -> EmbeddingResult: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingResult]: ...


class SparseEncoder(Protocol):
    @property
    def fingerprint(self) -> str: ...

    @property
    def requires_idf_modifier(self) -> bool: ...

    def encode_query(self, text: str) -> SparseEmbedding: ...

    def encode_documents(self, texts: Sequence[str]) -> list[SparseEmbedding]: ...


class QueryEncoder(Protocol):
    """Encode one query into dense + sparse vectors (BGE-M3 can implement this in one pass)."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def dense_dimension(self) -> int: ...

    @property
    def requires_idf_modifier(self) -> bool: ...

    async def encode(self, text: str) -> EncodedQuery: ...


class VectorSearch(Protocol):
    async def search(
        self,
        query: EncodedQuery,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        *,
        dense_k: int,
        sparse_k: int,
    ) -> HybridHits: ...


class Reranker(Protocol):
    @property
    def version(self) -> str: ...

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RankedChunk],
        top_n: int,
    ) -> list[RankedChunk]: ...


class PassageTokenCounter(Protocol):
    """Measure the content of one passage in the configured packing units."""

    def __call__(self, text: str, /) -> int: ...


class DocumentReader(Protocol):
    async def hydrate_authorized_chunks(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> dict[str, Chunk]: ...

    async def get_neighbors(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> dict[str, Chunk]: ...


class ConversationStore(Protocol):
    async def get_messages(
        self,
        identity: Identity,
        conversation_id: str,
        limit: int,
    ) -> tuple[ConversationMessage, ...]: ...

    async def append_messages(
        self,
        identity: Identity,
        conversation_id: str,
        messages: Sequence[ConversationMessage],
    ) -> None: ...


class TelemetrySink(Protocol):
    async def write_events(self, events: Sequence[TelemetryEvent]) -> None: ...

    async def write_traces(self, traces: Sequence[RequestTrace]) -> None: ...

    async def get_trace(self, request_id: str, tenant_id: str) -> RequestTrace | None: ...


class FollowupDetector(Protocol):
    @property
    def name(self) -> str: ...

    async def detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> FollowupDecision: ...


class TraceReader(Protocol):
    async def get_trace(self, request_id: str, tenant_id: str) -> RequestTrace | None: ...


class HealthCheck(Protocol):
    async def health_check(self) -> None: ...
