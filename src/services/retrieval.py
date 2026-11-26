"""Hybrid retrieval, fusion, and controlled query recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from models.domain import (
    Identity,
    ModelCallTrace,
    ModelResult,
    ModelUsage,
    QueryRecoveryOutput,
    QueryUnderstanding,
    RankedChunk,
    RetrievalPlan,
    SearchHit,
)
from services.contracts import DocumentReader, QueryEncoder, TextModel, VectorSearch
from services.understanding import query_drift_score
from utils.concurrency import run_all
from utils.prompts import QUERY_RECOVERY_SYSTEM, query_recovery_user_prompt
from utils.request_context import observed_call, stage_timeout
from utils.retry import NO_RETRY, retry_transient
from utils.settings import FusionStrategy, RetrievalConfig, RetryConfig


@dataclass(frozen=True, slots=True)
class QueryVariants:
    values: tuple[str, ...]
    usage: ModelUsage
    model_calls: tuple[ModelCallTrace, ...]


@dataclass(frozen=True, slots=True)
class FusedHit:
    chunk_id: str
    dense_score: float | None
    dense_rank: int | None
    sparse_score: float | None
    sparse_rank: int | None
    fusion_score: float
    fusion_rank: int


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    queries: tuple[str, ...]
    ranked_chunks: tuple[RankedChunk, ...]
    dense_result_count: int
    sparse_result_count: int
    fused_result_count: int
    candidate_k: int
    embedding_usage: ModelUsage
    query_recovery_usage: ModelUsage
    model_calls: tuple[ModelCallTrace, ...] = ()
    # Hydrated hits whose MongoDB metadata failed request filters that Qdrant had accepted.
    metadata_mismatch_count: int = 0


class QueryRecoveryService:
    """Build a bounded set of retrieval variants without model-controlled search loops."""

    def __init__(
        self,
        *,
        model: TextModel,
        model_name: str,
        max_output_tokens: int,
        timeout_seconds: float,
        max_variants: int,
        max_decomposed_queries: int,
        retry: RetryConfig = NO_RETRY,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._max_variants = max_variants
        self._max_decomposed_queries = max_decomposed_queries
        self._retry = retry

    async def build(self, query: QueryUnderstanding) -> QueryVariants:
        user_prompt = query_recovery_user_prompt(query)
        async with stage_timeout(self._timeout_seconds):
            result = await retry_transient(
                "query_recovery",
                lambda retry: observed_call(
                    "query_recovery",
                    self._model_name,
                    self._model.complete(
                        model=self._model_name,
                        system_prompt=QUERY_RECOVERY_SYSTEM,
                        user_prompt=user_prompt,
                        output_type=QueryRecoveryOutput,
                        max_output_tokens=self._max_output_tokens,
                    ),
                    retry_ordinal=retry,
                ),
                policy=self._retry,
            )
        # A rewritten follow-up whose words all survived the rewrite ("what about the second
        # one?") adds nothing as a search of its own. When the rewrite dropped some of the user's
        # words, the literal text is the one variant that still carries them.
        literal_adds_words = query_drift_score(query.literal_question, query.canonical_question) > 0
        literal = query.literal_question if not query.rewritten or literal_adds_words else None
        primary = _distinct((query.canonical_question, literal, result.value.keyword_query))[
            : self._max_variants
        ]
        decomposed = _distinct(result.value.decomposed_queries, exclude=primary)[
            : self._max_decomposed_queries
        ]
        return QueryVariants(
            values=primary + decomposed,
            usage=result.usage,
            model_calls=(_model_call("query_recovery", result),),
        )


class HybridRetriever:
    """Run one encoded query through dense and sparse search, then hydrate authorized chunks."""

    def __init__(
        self,
        *,
        encoder: QueryEncoder,
        vector_store: VectorSearch,
        repository: DocumentReader,
        config: RetrievalConfig,
        retrieval_timeout_seconds: float,
        retry: RetryConfig = NO_RETRY,
    ) -> None:
        self._encoder = encoder
        self._vector_store = vector_store
        self._repository = repository
        self._config = config
        self._retrieval_timeout_seconds = retrieval_timeout_seconds
        self._retry = retry

    async def retrieve(
        self,
        *,
        query: str,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        plan: RetrievalPlan,
    ) -> RetrievalOutcome:
        encoded = await self._encoder.encode(query)
        async with stage_timeout(self._retrieval_timeout_seconds):
            # Qdrant search is not a model call, so its tries have no operation rows.
            hits = await retry_transient(
                "retrieval",
                lambda _retry: self._vector_store.search(
                    encoded,
                    identity,
                    filters,
                    dense_k=plan.dense_k,
                    sparse_k=plan.sparse_k,
                ),
                policy=self._retry,
            )
        fused = fuse_results(
            hits.dense,
            hits.sparse,
            strategy=self._config.fusion_strategy,
            dense_weight=self._config.dense_weight,
            sparse_weight=self._config.sparse_weight,
            rrf_k=self._config.rrf_k,
            limit=None,
        )
        hydrated = await self._repository.hydrate_authorized_chunks(
            [item.chunk_id for item in fused], identity
        )
        present = [item for item in fused if item.chunk_id in hydrated]
        authorized = [item for item in present if hydrated[item.chunk_id].matches_filters(filters)]
        ranked = tuple(
            RankedChunk(
                chunk=hydrated[item.chunk_id],
                dense_score=item.dense_score,
                dense_rank=item.dense_rank,
                sparse_score=item.sparse_score,
                sparse_rank=item.sparse_rank,
                fusion_score=item.fusion_score,
                fusion_rank=rank,
            )
            for rank, item in enumerate(authorized[: plan.candidate_k], start=1)
        )
        return RetrievalOutcome(
            queries=(query,),
            ranked_chunks=ranked,
            dense_result_count=len(hits.dense),
            sparse_result_count=len(hits.sparse),
            fused_result_count=len(ranked),
            candidate_k=plan.candidate_k,
            embedding_usage=encoded.usage,
            query_recovery_usage=ModelUsage(),
            model_calls=encoded.model_calls,
            metadata_mismatch_count=len(present) - len(authorized),
        )


class StrategyRetriever:
    """Translate one retrieval plan into one or more bounded hybrid retrievals."""

    def __init__(
        self,
        *,
        hybrid: HybridRetriever,
        query_recovery: QueryRecoveryService,
        config: RetrievalConfig,
    ) -> None:
        self._hybrid = hybrid
        self._query_recovery = query_recovery
        self._config = config

    async def retrieve(
        self,
        *,
        plan: RetrievalPlan,
        query: QueryUnderstanding,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
    ) -> RetrievalOutcome:
        if not plan.recover_queries:
            return await self._hybrid.retrieve(
                query=query.canonical_question,
                identity=identity,
                filters=filters,
                plan=plan,
            )

        variants = await self._query_recovery.build(query)
        semaphore = asyncio.Semaphore(self._config.query_variant_concurrency)

        async def retrieve_variant(value: str) -> RetrievalOutcome:
            async with semaphore:
                return await self._hybrid.retrieve(
                    query=value,
                    identity=identity,
                    filters=filters,
                    plan=plan,
                )

        outcomes = await run_all(retrieve_variant(value) for value in variants.values)
        merged = merge_query_outcomes(
            outcomes,
            rrf_k=self._config.rrf_k,
            limit=plan.candidate_k,
        )
        return RetrievalOutcome(
            queries=variants.values,
            ranked_chunks=merged,
            dense_result_count=sum(item.dense_result_count for item in outcomes),
            sparse_result_count=sum(item.sparse_result_count for item in outcomes),
            fused_result_count=len(merged),
            candidate_k=plan.candidate_k,
            embedding_usage=sum((item.embedding_usage for item in outcomes), ModelUsage()),
            query_recovery_usage=variants.usage,
            model_calls=variants.model_calls
            + tuple(call for outcome in outcomes for call in outcome.model_calls),
            metadata_mismatch_count=sum(item.metadata_mismatch_count for item in outcomes),
        )


def _model_call[SchemaT: BaseModel](operation: str, result: ModelResult[SchemaT]) -> ModelCallTrace:
    return ModelCallTrace(
        operation=operation,
        provider=result.provider,
        model=result.model,
        request_id=result.request_id,
        usage=result.usage,
    )


def _distinct(
    values: Iterable[str | None],
    *,
    exclude: Collection[str] = (),
) -> tuple[str, ...]:
    """Strip values, drop blanks and excluded entries, and keep first occurrences in order."""

    cleaned = (value.strip() for value in values if value is not None)
    return tuple(dict.fromkeys(item for item in cleaned if item and item not in exclude))


def fuse_results(
    dense: Sequence[SearchHit],
    sparse: Sequence[SearchHit],
    *,
    strategy: FusionStrategy,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
    limit: int | None,
) -> list[FusedHit]:
    """Deduplicate dense and sparse results and produce one inspectable rank; None keeps all."""

    dense_by_id = {item.chunk_id: item for item in dense}
    sparse_by_id = {item.chunk_id: item for item in sparse}
    dense_normalized = _normalize_scores(dense)
    sparse_normalized = _normalize_scores(sparse)
    scores: dict[str, float] = {}
    for chunk_id in dense_by_id.keys() | sparse_by_id.keys():
        dense_hit = dense_by_id.get(chunk_id)
        sparse_hit = sparse_by_id.get(chunk_id)
        if strategy is FusionStrategy.RRF:
            score = (1 / (rrf_k + dense_hit.rank) if dense_hit else 0) + (
                1 / (rrf_k + sparse_hit.rank) if sparse_hit else 0
            )
        else:
            total_weight = dense_weight + sparse_weight
            score = (
                dense_weight * dense_normalized.get(chunk_id, 0)
                + sparse_weight * sparse_normalized.get(chunk_id, 0)
            ) / total_weight
        scores[chunk_id] = score
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        FusedHit(
            chunk_id=chunk_id,
            dense_score=dense_by_id[chunk_id].score if chunk_id in dense_by_id else None,
            dense_rank=dense_by_id[chunk_id].rank if chunk_id in dense_by_id else None,
            sparse_score=sparse_by_id[chunk_id].score if chunk_id in sparse_by_id else None,
            sparse_rank=sparse_by_id[chunk_id].rank if chunk_id in sparse_by_id else None,
            fusion_score=scores[chunk_id],
            fusion_rank=rank,
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]


def merge_query_outcomes(
    outcomes: Sequence[RetrievalOutcome],
    *,
    rrf_k: int,
    limit: int,
) -> tuple[RankedChunk, ...]:
    """Fuse bounded query-variant results while retaining each chunk's best provenance."""

    scores: dict[str, float] = {}
    best: dict[str, RankedChunk] = {}
    for outcome in outcomes:
        for rank, item in enumerate(outcome.ranked_chunks, start=1):
            scores[item.chunk.id] = scores.get(item.chunk.id, 0) + 1 / (rrf_k + rank)
            current = best.get(item.chunk.id)
            if current is None or item.fusion_rank < current.fusion_rank:
                best[item.chunk.id] = item
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return tuple(
        best[chunk_id].model_copy(update={"fusion_score": scores[chunk_id], "fusion_rank": rank})
        for rank, chunk_id in enumerate(ordered, start=1)
    )


def _normalize_scores(hits: Sequence[SearchHit]) -> dict[str, float]:
    if not hits:
        return {}
    low = min(item.score for item in hits)
    high = max(item.score for item in hits)
    if high == low:
        return {item.chunk_id: 1 for item in hits}
    return {item.chunk_id: (item.score - low) / (high - low) for item in hits}
