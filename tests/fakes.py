"""Faithful external-boundary fakes shared by tests."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import TypeVar, cast

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
    ModelUsage,
    RankedChunk,
    RequestTrace,
    SearchHit,
    SparseEmbedding,
    TelemetryEvent,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeTextModel:
    def __init__(self) -> None:
        self.responses: dict[type[BaseModel], deque[BaseModel]] = defaultdict(deque)
        self.calls: list[type[BaseModel]] = []
        self.output_prompts: list[tuple[type[BaseModel], str]] = []

    def queue(self, output_type: type[SchemaT], *values: SchemaT) -> None:
        self.responses[output_type].extend(values)

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]:
        del system_prompt, max_output_tokens
        self.calls.append(output_type)
        self.output_prompts.append((output_type, user_prompt))
        if not self.responses[output_type]:
            raise AssertionError(f"No queued fake response for {output_type.__name__}")
        value = cast(SchemaT, self.responses[output_type].popleft())
        return ModelResult(
            value=value,
            model=model,
            provider="fake",
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            request_id="fake-request",
        )


class StalledTextModel:
    """Never answers in time, so every stage that awaits it must enforce its own budget."""

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]:
        del model, system_prompt, user_prompt, output_type, max_output_tokens
        await asyncio.sleep(60)
        raise AssertionError("A stalled stage was not cancelled by its timeout")


class FakeEmbeddings:
    model = "fake-embedding"
    dimension = 3
    fingerprint = "fake-embedding:3"

    def __init__(self) -> None:
        self.query_calls = 0

    async def embed_query(self, text: str) -> EmbeddingResult:
        self.query_calls += 1
        return EmbeddingResult(
            vector=[float(len(text)), 0, 1],
            model=self.model,
            provider="fake",
            usage=ModelUsage(input_tokens=2, total_tokens=2),
            request_id=f"embedding-{self.query_calls}",
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        return [
            EmbeddingResult(
                vector=[float(len(text)), 0, 1],
                model=self.model,
                provider="fake",
                usage=ModelUsage(input_tokens=2, total_tokens=2),
                request_id=f"document-embedding-{index}",
            )
            for index, text in enumerate(texts, start=1)
        ]


class FakeQueryEncoder:
    fingerprint = "fake-encoder"
    dense_dimension = 3
    requires_idf_modifier = True

    def __init__(self) -> None:
        self.encode_calls = 0

    async def encode(self, text: str) -> EncodedQuery:
        self.encode_calls += 1
        return EncodedQuery(
            text=text,
            dense=(float(len(text)), 0.0, 1.0),
            sparse=SparseEmbedding(),
        )


class FakeVectorSearch:
    def __init__(self) -> None:
        self.dense_hits: list[SearchHit] = []
        self.sparse_hits: list[SearchHit] = []
        self.last_query: EncodedQuery | None = None
        self.search_calls = 0

    async def search(
        self,
        query: EncodedQuery,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        *,
        dense_k: int,
        sparse_k: int,
    ) -> HybridHits:
        del identity, filters
        self.search_calls += 1
        self.last_query = query
        return HybridHits(
            dense=tuple(self.dense_hits[:dense_k]),
            sparse=tuple(self.sparse_hits[:sparse_k]),
        )


class FakeFollowupDetector:
    name = "fake"

    def __init__(self, probability: float = 1.0, scores: tuple[float, ...] | None = None) -> None:
        self._probability = probability
        self._scores = scores
        self.detect_calls = 0

    async def detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> FollowupDecision:
        del question
        self.detect_calls += 1
        message_scores = self._scores if self._scores is not None else tuple(1.0 for _ in history)
        return FollowupDecision(
            detector=self.name,
            is_followup_probability=self._probability,
            message_scores=message_scores,
        )


class FakeReranker:
    version = "fake-reranker"

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RankedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        del query
        return [
            item.model_copy(update={"reranker_score": item.fusion_score, "reranker_rank": rank})
            for rank, item in enumerate(chunks[:top_n], start=1)
        ]


class FakeRepository:
    def __init__(self, chunks: Sequence[Chunk] = ()) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.current_versions = {chunk.document_id: chunk.document_version_id for chunk in chunks}
        self.traces: dict[str, RequestTrace] = {}
        self.events: list[TelemetryEvent] = []
        self.messages: dict[str, list[ConversationMessage]] = defaultdict(list)
        self.hydrate_calls: list[tuple[str, ...]] = []

    async def hydrate_authorized_chunks(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> dict[str, Chunk]:
        self.hydrate_calls.append(tuple(chunk_ids))
        return {
            chunk_id: chunk
            for chunk_id in chunk_ids
            if (chunk := self.chunks.get(chunk_id))
            and chunk.tenant_id == identity.tenant_id
            and not chunk.allowed_roles.isdisjoint(identity.roles)
            and self.current_versions.get(chunk.document_id) == chunk.document_version_id
        }

    async def get_neighbors(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> dict[str, Chunk]:
        return await self.hydrate_authorized_chunks(chunk_ids, identity)

    async def get_messages(
        self,
        identity: Identity,
        conversation_id: str,
        limit: int,
    ) -> tuple[ConversationMessage, ...]:
        del identity
        return tuple(self.messages[conversation_id][-limit:])

    async def append_messages(
        self,
        identity: Identity,
        conversation_id: str,
        messages: Sequence[ConversationMessage],
    ) -> None:
        del identity
        self.messages[conversation_id].extend(messages)

    async def write_events(self, events: Sequence[TelemetryEvent]) -> None:
        self.events.extend(events)

    async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
        self.traces.update((trace.request_id, trace) for trace in traces)

    async def get_trace(self, request_id: str, tenant_id: str) -> RequestTrace | None:
        trace = self.traces.get(request_id)
        return trace if trace and trace.tenant_id == tenant_id else None


def make_chunk(
    chunk_id: str,
    *,
    tenant_id: str = "tenant-a",
    roles: frozenset[str] = frozenset({"reader"}),
    document_id: str = "doc-a",
    version_id: str = "00000000-0000-0000-0000-000000000010",
    section_id: str = "section-a",
    sequence: int = 0,
    text: str = "Document: Guide\nSection: Test\nVersion: 1\n\nSupported evidence.",
    previous_id: str | None = None,
    next_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        tenant_id=tenant_id,
        document_id=document_id,
        document_version_id=version_id,
        document_version=1,
        document_title="Guide",
        section_id=section_id,
        sequence=sequence,
        heading_path=("Test",),
        text=text,
        token_count=20,
        previous_chunk_id=previous_id,
        next_chunk_id=next_id,
        allowed_roles=roles,
    )
