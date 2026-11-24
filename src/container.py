"""Composition root that resolves configured providers exactly once."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pymongo import AsyncMongoClient
from qdrant_client import AsyncQdrantClient

from db.initializer import DatabaseInitializer
from db.vector_index import VectorIndexSpec, build_vector_index_spec
from handlers.contracts import QueryHandler, ReadinessHandler, TraceQueryHandler
from handlers.health import DependencyReadinessHandler
from handlers.policy import DecisionPolicy
from handlers.search import SearchHandler
from handlers.search_strategies.registry import build_strategy_registry
from handlers.traces import TenantTraceHandler
from managers.conversations import MongoConversationManager
from managers.documents import MongoDocumentManager
from managers.health import MongoHealthManager
from managers.telemetry import MongoTelemetrySink
from managers.vector import QdrantVectorManager
from providers.cohere import CohereReranker
from providers.mongodb import build_mongo_client
from providers.openai import OpenAIEmbeddingProvider, OpenAITextModel, build_openai_client
from providers.qdrant import build_qdrant_client
from providers.sparse import Bm25SparseEncoder
from providers.tokenizer import load_o200k_counter
from services.citations import CitationRenderer
from services.context import ContextBuilder, ContextPacker, StructuralCompleter, utf8_byte_count
from services.contracts import (
    AsyncCloseable,
    ConversationStore,
    DocumentReader,
    FollowupDetector,
    PassageTokenCounter,
    QueryEncoder,
    Reranker,
    TextModel,
    VectorSearch,
)
from services.encoding import EmbeddingCache, HybridQueryEncoder
from services.evaluation import GenerationService, OnlineEvaluator
from services.followup import (
    AlwaysFollowupDetector,
    HeuristicFollowupDetector,
    ModelFollowupDetector,
)
from services.retrieval import HybridRetriever, QueryRecoveryService, StrategyRetriever
from services.telemetry import TelemetryService
from services.understanding import QueryUnderstandingService
from utils.errors import ConfigurationError
from utils.settings import FollowupDetectorName, Settings, SparseEncoderName, TokenizerEncoding

logger = logging.getLogger(__name__)
_RESOURCE_CLOSE_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class AppContainer:
    """Own the resolved graph and shared asynchronous clients."""

    settings: Settings
    query_handler: QueryHandler
    trace_handler: TraceQueryHandler
    readiness_handler: ReadinessHandler
    telemetry: TelemetryService
    initializer: DatabaseInitializer
    external_resources: tuple[AsyncCloseable, ...]

    async def start(self) -> None:
        await self.initializer.ensure_ready()
        await self.telemetry.start()

    async def close(self) -> None:
        try:
            await self.telemetry.close()
        finally:
            await _close_all(self.external_resources)


@dataclass(frozen=True, slots=True)
class _Clients:
    openai: AsyncOpenAI
    mongo: AsyncMongoClient[dict[str, Any]]
    qdrant: AsyncQdrantClient
    reranker: CohereReranker


async def create_container(settings: Settings) -> AppContainer:
    """Resolve provider implementations from validated startup settings."""

    _validate_implementations(settings)
    count_tokens: PassageTokenCounter = utf8_byte_count
    if settings.context.tokenizer_encoding is TokenizerEncoding.O200K_BASE:
        count_tokens = await asyncio.to_thread(load_o200k_counter)
    resources: list[AsyncCloseable] = []
    try:
        return _compose(settings, resources, count_tokens)
    except BaseException:
        await _close_all(resources)
        raise


def _validate_implementations(settings: Settings) -> None:
    """Reject unknown implementation names before any external client is constructed."""

    _require_supported("AI provider", settings.ai.provider, ("openai",))
    _require_supported("database provider", settings.mongodb.provider, ("mongodb",))
    _require_supported("vector database provider", settings.vector_store.provider, ("qdrant",))
    _require_supported("reranker provider", settings.reranker.provider, ("cohere",))
    _require_supported(
        "sparse encoder", settings.vector_store.sparse.encoder, {SparseEncoderName.BM25}
    )
    _require_supported(
        "follow-up detector", settings.contextualization.detector, FollowupDetectorName
    )


def _require_supported(label: str, configured: str, supported: Iterable[str]) -> None:
    if configured not in set(supported):
        raise ConfigurationError(f"Unsupported {label}: {configured}")


def _compose(
    settings: Settings, resources: list[AsyncCloseable], count_tokens: PassageTokenCounter
) -> AppContainer:
    """Wire the process graph in dependency order; every client is tracked when created."""

    clients = _build_clients(settings, resources)
    text_model = OpenAITextModel(clients.openai, reasoning_effort=settings.ai.reasoning_effort)
    embeddings = OpenAIEmbeddingProvider(
        clients.openai,
        model=settings.ai.embedding_model,
        dimension=settings.ai.embedding_dimension,
    )
    encoder = HybridQueryEncoder(
        dense=embeddings,
        sparse=Bm25SparseEncoder(settings.vector_store.sparse),
        cache=EmbeddingCache(settings.retrieval.embedding_cache_entries),
        embedding_timeout_seconds=settings.timeouts.embedding_seconds,
    )
    index = build_vector_index_spec(
        settings.vector_store,
        encoder_fingerprint=encoder.fingerprint,
        dense_dimension=encoder.dense_dimension,
        sparse_idf_modifier=encoder.requires_idf_modifier,
    )
    database = clients.mongo[settings.mongodb.database]
    documents = MongoDocumentManager(database)
    telemetry = TelemetryService(
        MongoTelemetrySink(
            database,
            trace_collection=settings.mongodb.trace_collection,
            event_collection=settings.mongodb.telemetry_collection,
        ),
        settings.telemetry,
        hash_key=settings.external.telemetry_hash_key.get_secret_value().encode(),
    )
    vector_manager = QdrantVectorManager(
        clients.qdrant,
        collection_alias=settings.vector_store.collection_alias,
        dense_vector_name=settings.vector_store.dense_vector_name,
        sparse_vector_name=settings.vector_store.sparse_vector_name,
    )
    detector = _build_detector(settings, text_model)
    mongo_health = MongoHealthManager(clients.mongo)
    return AppContainer(
        settings=settings,
        query_handler=_build_query_handler(
            settings,
            count_tokens=count_tokens,
            text_model=text_model,
            detector=detector,
            encoder=encoder,
            index=index,
            vector_manager=vector_manager,
            documents=documents,
            conversations=MongoConversationManager(
                database,
                write_provenance=settings.contextualization.write_history_provenance,
                max_history_refs=settings.history_ref_limit,
            ),
            reranker=clients.reranker,
            telemetry=telemetry,
        ),
        trace_handler=TenantTraceHandler(reader=telemetry, admin_role=settings.auth.admin_role),
        readiness_handler=DependencyReadinessHandler((mongo_health, vector_manager)),
        telemetry=telemetry,
        initializer=DatabaseInitializer(
            clients.mongo,
            clients.qdrant,
            mongo=settings.mongodb,
            index=index,
        ),
        external_resources=tuple(resources),
    )


def _build_clients(settings: Settings, resources: list[AsyncCloseable]) -> _Clients:
    """Create each external client and track it before the next one can fail."""

    assert settings.external.openai_api_key is not None
    assert settings.external.cohere_api_key is not None
    openai_client = _track(
        build_openai_client(
            api_key=settings.external.openai_api_key.get_secret_value(),
            base_url=settings.external.openai_base_url,
            max_retries=settings.ai.max_retries,
            timeout_seconds=settings.timeouts.generation_seconds,
        ),
        resources,
    )
    mongo_client = _track(
        build_mongo_client(
            uri=settings.external.mongodb_uri.get_secret_value(),
            server_selection_timeout_ms=settings.mongodb.server_selection_timeout_ms,
            operation_timeout_ms=settings.mongodb.operation_timeout_ms,
        ),
        resources,
    )
    qdrant_client = _track(
        build_qdrant_client(
            url=settings.external.qdrant_url,
            api_key=(
                settings.external.qdrant_api_key.get_secret_value()
                if settings.external.qdrant_api_key
                else None
            ),
            timeout_seconds=settings.timeouts.retrieval_seconds,
        ),
        resources,
    )
    reranker = _track(
        CohereReranker(
            api_key=settings.external.cohere_api_key.get_secret_value(),
            model=settings.reranker.model,
            base_url=settings.external.cohere_base_url,
            timeout_seconds=settings.timeouts.reranking_seconds,
        ),
        resources,
    )
    return _Clients(
        openai=openai_client,
        mongo=mongo_client,
        qdrant=qdrant_client,
        reranker=reranker,
    )


def _track[T: AsyncCloseable](resource: T, resources: list[AsyncCloseable]) -> T:
    resources.append(resource)
    return resource


async def _close_all(
    resources: Iterable[AsyncCloseable],
    *,
    timeout_seconds: float = _RESOURCE_CLOSE_TIMEOUT_SECONDS,
) -> None:
    """Close every resource concurrently; a close that hangs is abandoned after its timeout."""

    results = await asyncio.gather(
        *(_bounded_close(resource, timeout_seconds) for resource in resources),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            logger.error(
                "Resource shutdown failed",
                exc_info=(type(result), result, result.__traceback__),
            )


async def _bounded_close(resource: AsyncCloseable, timeout_seconds: float) -> None:
    async with asyncio.timeout(timeout_seconds):
        await resource.close()


def _build_detector(settings: Settings, text_model: TextModel) -> FollowupDetector:
    contextualization = settings.contextualization
    match contextualization.detector:
        case FollowupDetectorName.MODEL:
            return ModelFollowupDetector(
                model=text_model,
                model_name=contextualization.detector_model,
                max_output_tokens=settings.ai.max_output_tokens,
            )
        case FollowupDetectorName.HEURISTIC:
            return HeuristicFollowupDetector()
        case FollowupDetectorName.ALWAYS:
            return AlwaysFollowupDetector()


def _build_query_handler(
    settings: Settings,
    *,
    count_tokens: PassageTokenCounter,
    text_model: TextModel,
    detector: FollowupDetector,
    encoder: QueryEncoder,
    index: VectorIndexSpec,
    vector_manager: VectorSearch,
    documents: DocumentReader,
    conversations: ConversationStore,
    reranker: Reranker,
    telemetry: TelemetryService,
) -> SearchHandler:
    return SearchHandler(
        query_understanding=QueryUnderstandingService(
            detector=detector,
            model=text_model,
            model_name=settings.ai.contextualization_model,
            max_output_tokens=settings.ai.max_output_tokens,
            config=settings.contextualization,
            detection_timeout_seconds=settings.timeouts.followup_detection_seconds,
            contextualization_timeout_seconds=settings.timeouts.contextualization_seconds,
        ),
        retriever=_build_retriever(
            settings,
            text_model=text_model,
            encoder=encoder,
            vector_manager=vector_manager,
            documents=documents,
        ),
        context_builder=ContextBuilder(
            reranker=reranker,
            completer=StructuralCompleter(
                repository=documents,
                max_neighbors_per_seed=settings.context.max_neighbors_per_seed,
                short_chunk_tokens=settings.context.short_chunk_tokens,
            ),
            packer=ContextPacker(settings.context, count_tokens),
            generation_k=settings.retrieval.generation_k,
            reranking_timeout_seconds=settings.timeouts.reranking_seconds,
        ),
        generator=GenerationService(
            model=text_model,
            model_name=settings.ai.generation_model,
            max_output_tokens=settings.ai.max_output_tokens,
            timeout_seconds=settings.timeouts.generation_seconds,
        ),
        evaluator=OnlineEvaluator(
            model=text_model,
            model_name=settings.ai.evaluation_model,
            max_output_tokens=settings.ai.max_output_tokens,
            timeout_seconds=settings.timeouts.evaluation_seconds,
        ),
        decision=DecisionPolicy(policy=settings.policy, retrieval=settings.retrieval),
        strategies=build_strategy_registry(settings.retrieval),
        documents=documents,
        conversations=conversations,
        telemetry=telemetry,
        citations=CitationRenderer(inline=settings.answers.inline_citations),
        policy=settings.policy,
        retrieval=settings.retrieval,
        contextualization=settings.contextualization,
        history_ref_limit=settings.history_ref_limit,
        timeouts=settings.timeouts,
        runtime_versions=_runtime_versions(
            settings,
            detector=detector,
            encoder=encoder,
            index=index,
            reranker=reranker,
        ),
        configuration_fingerprint=hashlib.sha256(settings.model_dump_json().encode()).hexdigest(),
    )


def _build_retriever(
    settings: Settings,
    *,
    text_model: TextModel,
    encoder: QueryEncoder,
    vector_manager: VectorSearch,
    documents: DocumentReader,
) -> StrategyRetriever:
    hybrid = HybridRetriever(
        encoder=encoder,
        vector_store=vector_manager,
        repository=documents,
        config=settings.retrieval,
        retrieval_timeout_seconds=settings.timeouts.retrieval_seconds,
    )
    query_recovery = QueryRecoveryService(
        model=text_model,
        model_name=settings.ai.query_recovery_model,
        max_output_tokens=settings.ai.max_output_tokens,
        timeout_seconds=settings.timeouts.query_recovery_seconds,
        max_variants=settings.retrieval.max_query_variants,
        max_decomposed_queries=settings.retrieval.max_decomposed_queries,
    )
    return StrategyRetriever(
        hybrid=hybrid,
        query_recovery=query_recovery,
        config=settings.retrieval,
    )


def _runtime_versions(
    settings: Settings,
    *,
    detector: FollowupDetector,
    encoder: QueryEncoder,
    index: VectorIndexSpec,
    reranker: Reranker,
) -> dict[str, str]:
    return {
        "ai_provider": settings.ai.provider,
        "generation_model": settings.ai.generation_model,
        "contextualization_model": settings.ai.contextualization_model,
        "query_recovery_model": settings.ai.query_recovery_model,
        "evaluation_model": settings.ai.evaluation_model,
        "embedding_model": settings.ai.embedding_model,
        "query_encoder": encoder.fingerprint,
        "followup_detector": detector.name,
        "vector_index": index.physical_collection,
        "reranker": reranker.version,
        "vector_store_provider": settings.vector_store.provider,
        "database_provider": settings.mongodb.provider,
    }
