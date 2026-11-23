from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

import pytest

from handlers.policy import DecisionPolicy
from handlers.search import SearchHandler
from handlers.search_strategies.registry import build_strategy_registry
from models.domain import (
    Citation,
    CompletenessEvaluation,
    CompletenessStatus,
    ContextualizationOutput,
    ContextualizationTrace,
    ConversationMessage,
    EncodedQuery,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    GenerationDraft,
    HybridHits,
    Identity,
    QueryRecoveryOutput,
    QueryRequest,
    RagFailure,
    RequestTrace,
    RequirementEvaluation,
    RequirementStatus,
    ResultStatus,
    RetrievalStrategy,
    RetryDecision,
    RetryHypothesis,
    SearchHit,
    Source,
)
from providers.sparse import Bm25SparseEncoder
from services.citations import CitationRenderer
from services.context import ContextBuilder, ContextPacker, StructuralCompleter
from services.encoding import EmbeddingCache, HybridQueryEncoder
from services.evaluation import GenerationService, OnlineEvaluator
from services.retrieval import HybridRetriever, QueryRecoveryService, StrategyRetriever
from services.telemetry import TelemetryService
from services.understanding import QueryUnderstandingService
from tests.fakes import (
    FakeEmbeddings,
    FakeFollowupDetector,
    FakeRepository,
    FakeReranker,
    FakeTextModel,
    FakeVectorSearch,
    make_chunk,
)
from utils.errors import ProviderError, RequestDeadlineExceededError
from utils.settings import (
    ContentMode,
    ContextConfig,
    ContextualizationConfig,
    PolicyConfig,
    RetrievalConfig,
    SparseConfig,
    TelemetryConfig,
    TimeoutConfig,
)

CHUNK_ID = "00000000-0000-0000-0000-000000000001"
STAGE_SECONDS = 10
CITATION = Citation(number=1, document_id="doc-a", title="Guide", section="Test", version=1)
HAPPY = RetrievalStrategy.HAPPY_PATH
RECALL = RetrievalStrategy.RECALL_EXPANSION
RECOVERY = RetrievalStrategy.QUERY_RECOVERY
CONTEXT_ISOLATION = RetrievalStrategy.CONTEXT_ISOLATION
SHORT_BUDGET = TimeoutConfig(request_seconds=2, finalization_reserve_seconds=1)


class ScriptedVectorSearch(FakeVectorSearch):
    """Fake vector search whose calls from ``fail_from`` (1-based) run ``action`` first."""

    def __init__(self, *, fail_from: int, action: Callable[[], Awaitable[None]]) -> None:
        super().__init__()
        self._fail_from = fail_from
        self._action = action
        self._calls = 0

    async def search(
        self,
        query: EncodedQuery,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        *,
        dense_k: int,
        sparse_k: int,
    ) -> HybridHits:
        self._calls += 1
        if self._calls >= self._fail_from:
            await self._action()
        return await super().search(query, identity, filters, dense_k=dense_k, sparse_k=sparse_k)


async def _stall() -> None:
    await asyncio.sleep(60)


async def _time_out() -> None:
    raise TimeoutError("vector search timed out")


async def _unavailable() -> None:
    raise ProviderError("vector store unavailable")


def hit_store() -> FakeVectorSearch:
    store = FakeVectorSearch()
    store.dense_hits = [SearchHit(chunk_id=CHUNK_ID, score=1, rank=1)]
    store.sparse_hits = [SearchHit(chunk_id=CHUNK_ID, score=1, rank=1)]
    return store


def scripted_store(
    *, fail_from: int, action: Callable[[], Awaitable[None]]
) -> ScriptedVectorSearch:
    store = ScriptedVectorSearch(fail_from=fail_from, action=action)
    store.dense_hits = [SearchHit(chunk_id=CHUNK_ID, score=1, rank=1)]
    store.sparse_hits = [SearchHit(chunk_id=CHUNK_ID, score=1, rank=1)]
    return store


def build_orchestrator(
    *,
    model: FakeTextModel,
    repository: FakeRepository,
    vector_store: FakeVectorSearch,
    timeouts: TimeoutConfig | None = None,
    detector: FakeFollowupDetector | None = None,
    inline_citations: bool = True,
) -> SearchHandler:
    retrieval = RetrievalConfig(
        happy_dense_k=4,
        happy_sparse_k=4,
        happy_candidate_k=4,
        recall_dense_k=8,
        recall_sparse_k=8,
        recall_candidate_k=8,
        generation_k=2,
    )
    context = ContextConfig(
        max_tokens=512,
        max_seed_chunks=2,
        max_document_share=1,
        max_neighbors_per_seed=0,
    )
    contextualization = ContextualizationConfig()
    policy = PolicyConfig()
    encoder = HybridQueryEncoder(
        dense=FakeEmbeddings(),
        sparse=Bm25SparseEncoder(SparseConfig()),
        cache=EmbeddingCache(0),
        embedding_timeout_seconds=STAGE_SECONDS,
    )
    hybrid = HybridRetriever(
        encoder=encoder,
        vector_store=vector_store,
        repository=repository,
        config=retrieval,
        retrieval_timeout_seconds=STAGE_SECONDS,
    )
    return SearchHandler(
        query_understanding=QueryUnderstandingService(
            detector=detector if detector is not None else FakeFollowupDetector(probability=0),
            model=model,
            model_name="context-model",
            max_output_tokens=256,
            config=contextualization,
            detection_timeout_seconds=STAGE_SECONDS,
            contextualization_timeout_seconds=STAGE_SECONDS,
        ),
        retriever=StrategyRetriever(
            hybrid=hybrid,
            query_recovery=QueryRecoveryService(
                model=model,
                model_name="query-model",
                max_output_tokens=256,
                timeout_seconds=STAGE_SECONDS,
                max_variants=4,
                max_decomposed_queries=3,
            ),
            config=retrieval,
        ),
        context_builder=ContextBuilder(
            reranker=FakeReranker(),
            completer=StructuralCompleter(
                repository=repository,
                max_neighbors_per_seed=0,
                short_chunk_tokens=context.short_chunk_tokens,
            ),
            packer=ContextPacker(context),
            generation_k=2,
            reranking_timeout_seconds=STAGE_SECONDS,
        ),
        generator=GenerationService(
            model=model,
            model_name="generation-model",
            max_output_tokens=256,
            timeout_seconds=STAGE_SECONDS,
        ),
        evaluator=OnlineEvaluator(
            model=model,
            model_name="evaluation-model",
            max_output_tokens=256,
            timeout_seconds=STAGE_SECONDS,
        ),
        decision=DecisionPolicy(policy=policy, retrieval=retrieval),
        strategies=build_strategy_registry(retrieval),
        documents=repository,
        conversations=repository,
        telemetry=TelemetryService(
            repository,
            TelemetryConfig(enabled=True, content_mode=ContentMode.METADATA_ONLY),
            hash_key=b"test-telemetry-key",
        ),
        citations=CitationRenderer(inline=inline_citations),
        policy=policy,
        retrieval=retrieval,
        contextualization=contextualization,
        timeouts=timeouts if timeouts is not None else TimeoutConfig(),
        runtime_versions={"ai_provider": "fake", "embedding_model": "fake-embedding"},
        configuration_fingerprint="test-configuration",
    )


def _supported() -> FaithfulnessEvaluation:
    return FaithfulnessEvaluation(
        passed=True,
        failure_type=FaithfulnessStatus.SUPPORTED,
        verified_sources=("E1",),
        confidence=1,
    )


def _complete_evaluation() -> CompletenessEvaluation:
    return CompletenessEvaluation(
        passed=True,
        failure_type=CompletenessStatus.COMPLETE,
        requirements=(
            RequirementEvaluation(
                requirement="advantages",
                status=RequirementStatus.ANSWERED,
                available_in_context=True,
                supporting_chunks=("E1",),
            ),
        ),
        confidence=1,
    )


def _partial_evaluation() -> CompletenessEvaluation:
    """Answer one requirement from evidence and leave another absent from the corpus."""

    return CompletenessEvaluation(
        passed=False,
        failure_type=CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE,
        requirements=(
            RequirementEvaluation(
                requirement="advantages",
                status=RequirementStatus.ANSWERED,
                available_in_context=True,
                supporting_chunks=("E1",),
            ),
            RequirementEvaluation(
                requirement="limitations",
                status=RequirementStatus.MISSING,
                available_in_context=False,
            ),
        ),
        confidence=1,
    )


def queue_complete_answer(model: FakeTextModel, answer: str) -> None:
    model.queue(GenerationDraft, GenerationDraft(answer=answer))
    model.queue(FaithfulnessEvaluation, _supported())
    model.queue(CompletenessEvaluation, _complete_evaluation())


def queue_partial_rounds(model: FakeTextModel, rounds: int) -> None:
    model.queue(
        GenerationDraft,
        *(GenerationDraft(answer="Advantages [E1].") for _ in range(rounds)),
    )
    model.queue(FaithfulnessEvaluation, *(_supported() for _ in range(rounds)))
    model.queue(CompletenessEvaluation, *(_partial_evaluation() for _ in range(rounds)))


def _decisions(trace: RequestTrace) -> list[RetryDecision]:
    decisions = [attempt.decision for attempt in trace.attempts]
    assert all(decision is not None for decision in decisions)
    return [decision for decision in decisions if decision is not None]


def _unsupported(*, correcting: bool = False) -> FaithfulnessEvaluation:
    return FaithfulnessEvaluation(
        passed=False,
        failure_type=FaithfulnessStatus.UNSUPPORTED_CLAIMS,
        correcting_evidence_available=correcting,
        correcting_chunks=("E1",) if correcting else (),
        confidence=1,
    )


def _generation_payloads(model: FakeTextModel) -> list[dict[str, object]]:
    return [
        json.loads(prompt) for output, prompt in model.output_prompts if output is GenerationDraft
    ]


class _SourcesUnavailableRepository(FakeRepository):
    """Authorized retrieval works, but the final source lookup fails."""

    async def get_sources(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> tuple[Source, ...]:
        del chunk_ids, identity
        raise ProviderError("document store unavailable")


async def test_orchestrator_repairs_once_and_re_evaluates_entire_answer(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Advantages only [E1]."),
        GenerationDraft(answer="Advantages and limitations [E1]."),
    )
    model.queue(FaithfulnessEvaluation, _supported(), _supported())
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=False,
            failure_type=CompletenessStatus.MISSING_ANSWER_CONTENT,
            requirements=(
                RequirementEvaluation(
                    requirement="limitations",
                    status=RequirementStatus.MISSING,
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
            missing_information_available=True,
            confidence=1,
        ),
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            requirements=(
                RequirementEvaluation(
                    requirement="limitations",
                    status=RequirementStatus.ANSWERED,
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
            confidence=1,
        ),
    )
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.COMPLETE
    assert response.answer == "Advantages and limitations [1]."
    assert response.citations == (CITATION,)
    assert response.metadata.generation_repaired
    assert response.metadata.attempts == 2
    trace = repository.traces[response.request_id]
    assert trace.attempted_strategies == (HAPPY,)
    assert trace.generation_repair_used
    assert [attempt.generation_repair_attempted for attempt in trace.attempts] == [False, True]
    assert trace.runtime_versions["ai_provider"] == "fake"
    assert trace.configuration_fingerprint == "test-configuration"
    assert trace.token_usage.input_tokens > 0
    assert {call.provider for call in trace.model_calls} == {"fake"}
    assert "repair" in {call.operation for call in trace.model_calls}


async def test_complete_answer_renders_verified_citations_and_strips_invalid_markers(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages [E1] and limitations [E9].")
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.COMPLETE
    assert response.answer == "Advantages [1] and limitations."
    assert response.citations == (CITATION,)
    assert response.sources[0].document_id == "doc-a"


async def test_inline_citations_disabled_returns_plain_answer(identity: Identity) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages [E1] and limitations.")
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        inline_citations=False,
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.answer == "Advantages and limitations."
    assert response.citations == ()


async def test_orchestrator_exhausts_each_retrieval_strategy_once_then_abstains(
    identity: Identity,
) -> None:
    repository = FakeRepository()
    vector_store = FakeVectorSearch()
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput(keyword_query="exact identifier"))
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=vector_store)

    response = await orchestrator.query(identity, QueryRequest(message="Unknown fact"))

    assert response.status is ResultStatus.INSUFFICIENT_EVIDENCE
    assert response.metadata.attempts == 3
    trace = repository.traces[response.request_id]
    assert len(trace.attempted_strategies) == len(set(trace.attempted_strategies)) == 3
    assert not trace.generation_repair_used
    decisions = _decisions(trace)
    assert [decision.hypothesis for decision in decisions] == [RetryHypothesis.LOW_RECALL] * 3
    assert [decision.next_strategy for decision in decisions] == [RECALL, RECOVERY, None]


async def test_partial_answer_after_exhausting_strategies_discloses_unanswered_requirements(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    queue_partial_rounds(model, rounds=3)
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.answer == (
        "Advantages [1].\n\nI could not verify from the available sources: limitations."
    )
    assert response.unanswered_requirements == ("limitations",)
    assert response.metadata.degraded is False
    trace = repository.traces[response.request_id]
    assert trace.attempted_strategies == (HAPPY, RECOVERY, RECALL)
    assert trace.final_status is ResultStatus.VERIFIED_PARTIAL
    assert trace.degraded is False
    assert trace.error_code is None


async def test_deadline_salvages_best_verified_answer_and_keeps_the_retry_decision(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    queue_partial_rounds(model, rounds=1)
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=scripted_store(fail_from=2, action=_stall),
        timeouts=SHORT_BUDGET,
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.metadata.degraded is True
    assert "I could not verify from the available sources: limitations." in response.answer
    trace = repository.traces[response.request_id]
    assert trace.error_code == "REQUEST_DEADLINE_EXCEEDED"
    assert trace.degraded is True
    assert trace.final_status is ResultStatus.VERIFIED_PARTIAL
    assert trace.attempted_strategies == (HAPPY, RECOVERY)
    decision = _decisions(trace)[0]
    assert decision.hypothesis is RetryHypothesis.QUERY_DRIFT
    assert decision.next_strategy is RECOVERY
    assert RagFailure.QUERY_DRIFT in trace.attempts[0].failure_types


async def test_deadline_without_a_verified_answer_raises_request_deadline_error(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    orchestrator = build_orchestrator(
        model=FakeTextModel(),
        repository=repository,
        vector_store=scripted_store(fail_from=1, action=_stall),
        timeouts=SHORT_BUDGET,
    )

    with pytest.raises(RequestDeadlineExceededError):
        await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    trace = next(iter(repository.traces.values()))
    assert trace.error_code == "REQUEST_DEADLINE_EXCEEDED"
    assert trace.final_status is None
    assert trace.degraded is False


async def test_stage_timeout_after_a_verified_answer_is_salvaged_and_traced(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    queue_partial_rounds(model, rounds=1)
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=scripted_store(fail_from=2, action=_time_out),
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.metadata.degraded is True
    assert "limitations" in response.answer
    trace = repository.traces[response.request_id]
    assert trace.error_code == "TimeoutError"
    assert trace.degraded is True


async def test_infrastructure_failure_after_a_verified_answer_is_salvaged(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    queue_partial_rounds(model, rounds=1)
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=scripted_store(fail_from=2, action=_unavailable),
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.metadata.degraded is True
    trace = repository.traces[response.request_id]
    assert trace.error_code == "ProviderError"


async def test_infrastructure_failure_without_a_verified_answer_is_reraised(
    identity: Identity,
) -> None:
    repository = FakeRepository()
    orchestrator = build_orchestrator(
        model=FakeTextModel(),
        repository=repository,
        vector_store=scripted_store(fail_from=1, action=_unavailable),
    )

    with pytest.raises(ProviderError):
        await orchestrator.query(identity, QueryRequest(message="Question"))

    trace = next(iter(repository.traces.values()))
    assert trace.error_code == "ProviderError"


async def test_provider_timeout_is_not_mislabeled_as_global_deadline(
    identity: Identity,
) -> None:
    repository = FakeRepository()
    orchestrator = build_orchestrator(
        model=FakeTextModel(),
        repository=repository,
        vector_store=scripted_store(fail_from=1, action=_time_out),
    )

    with pytest.raises(TimeoutError):
        await orchestrator.query(identity, QueryRequest(message="Question"))

    trace = next(iter(repository.traces.values()))
    assert trace.error_code == "TimeoutError"


async def test_conversation_history_is_contextualized_recorded_and_extended(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    repository.messages["conversation-1"].extend(
        [
            ConversationMessage(role="user", content="Tell me about migration"),
            ConversationMessage(role="assistant", content="Migration is staged."),
            ConversationMessage(role="user", content="Tell me more"),
        ]
    )
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="Explain the feature limitations"),
    )
    queue_complete_answer(model, "Advantages [E1].")
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        detector=FakeFollowupDetector(probability=1),
    )

    response = await orchestrator.query(
        identity,
        QueryRequest(message="Explain the feature", conversation_id="conversation-1"),
    )

    assert response.status is ResultStatus.COMPLETE
    trace = repository.traces[response.request_id]
    assert trace.contextualization == ContextualizationTrace(
        enabled=True,
        detector="fake",
        followup_probability=1,
        fail_open=False,
        history_messages=3,
        selected_messages=3,
        rewritten=True,
    )
    assert repository.messages["conversation-1"][-2].content == "Explain the feature"
    assert repository.messages["conversation-1"][-1].content == "Advantages."
    assert response.answer == "Advantages [1]."


async def test_later_turns_never_see_citation_markers_from_earlier_answers(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="Explain the feature limitations"),
    )
    queue_complete_answer(model, "Advantages [E1].")
    queue_complete_answer(model, "Limitations [E1].")
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        detector=FakeFollowupDetector(probability=1),
    )

    first = await orchestrator.query(
        identity,
        QueryRequest(message="Explain the feature", conversation_id="conversation-1"),
    )
    await orchestrator.query(
        identity,
        QueryRequest(message="Tell me the limitations", conversation_id="conversation-1"),
    )

    assert first.answer == "Advantages [1]."
    (contextualizer_prompt,) = [
        prompt for output, prompt in model.output_prompts if output is ContextualizationOutput
    ]
    assert "Advantages." in contextualizer_prompt
    assert "[1]" not in contextualizer_prompt
    assert "[E1]" not in contextualizer_prompt


async def test_contaminated_generation_is_retried_in_isolation_without_repeating_retrieval(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    repository.messages["conversation-1"].extend(
        [
            ConversationMessage(role="user", content="Tell me about migration"),
            ConversationMessage(role="assistant", content="Migration is staged."),
        ]
    )
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(
            standalone_question="Explain the feature",
            relevant_conversation_context=("The user asked about migration earlier.",),
        ),
    )
    # The unsupported claim is repaired first; the repair still fails, so isolation runs.
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Advantages from the earlier turn [E1]."),
        GenerationDraft(answer="Advantages, still from the earlier turn [E1]."),
        GenerationDraft(answer="Advantages [E1]."),
    )
    model.queue(FaithfulnessEvaluation, _unsupported(), _unsupported(), _supported())
    model.queue(
        CompletenessEvaluation,
        _complete_evaluation(),
        _complete_evaluation(),
        _complete_evaluation(),
    )
    vector_store = hit_store()
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=vector_store,
        detector=FakeFollowupDetector(probability=1),
    )

    response = await orchestrator.query(
        identity,
        QueryRequest(message="Explain the feature", conversation_id="conversation-1"),
    )

    assert response.status is ResultStatus.COMPLETE
    trace = repository.traces[response.request_id]
    assert trace.attempted_strategies == (HAPPY, CONTEXT_ISOLATION)
    assert [attempt.generation_repair_attempted for attempt in trace.attempts] == [
        False,
        True,
        False,
    ]
    repair_decision = trace.attempts[1].decision
    assert repair_decision is not None
    assert repair_decision.hypothesis is RetryHypothesis.CONTEXT_CONTAMINATION
    assert vector_store.search_calls == 1
    first, repair, isolated = _generation_payloads(model)
    assert first["relevant_conversation_facts"] == ["The user asked about migration earlier."]
    assert "relevant_conversation_facts" not in repair
    assert isolated["relevant_conversation_facts"] == []


async def test_unsupported_claim_without_correcting_evidence_is_repaired_instead_of_abstaining(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Advantages [E1]. It is also free."),
        GenerationDraft(answer="Advantages [E1]."),
    )
    model.queue(FaithfulnessEvaluation, _unsupported(), _supported())
    model.queue(CompletenessEvaluation, _complete_evaluation(), _complete_evaluation())
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.COMPLETE
    assert response.metadata.generation_repaired
    assert response.answer == "Advantages [1]."
    trace = repository.traces[response.request_id]
    assert trace.attempted_strategies == (HAPPY,)


async def test_hallucinated_answer_to_unanswerable_question_keeps_the_repair_for_later(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Invented details [E1]."),
        GenerationDraft(answer="Advantages [E1]."),
        GenerationDraft(answer="Advantages and limits [E1]."),
    )
    not_answerable = CompletenessEvaluation(
        passed=False, failure_type=CompletenessStatus.QUESTION_NOT_ANSWERABLE, confidence=1
    )
    missing_content = CompletenessEvaluation(
        passed=False,
        failure_type=CompletenessStatus.MISSING_ANSWER_CONTENT,
        requirements=(
            RequirementEvaluation(
                requirement="limits",
                status=RequirementStatus.MISSING,
                available_in_context=True,
                supporting_chunks=("E1",),
            ),
        ),
        confidence=1,
    )
    model.queue(FaithfulnessEvaluation, _unsupported(), _supported(), _supported())
    model.queue(CompletenessEvaluation, not_answerable, missing_content, _complete_evaluation())
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.COMPLETE
    assert response.metadata.generation_repaired
    trace = repository.traces[response.request_id]
    assert trace.attempted_strategies == (HAPPY, RECALL)
    assert [attempt.generation_repair_attempted for attempt in trace.attempts] == [
        False,
        False,
        True,
    ]


async def test_judge_flag_that_disagrees_with_its_status_does_not_fail_the_request(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Advantages [E1]. It is also free."),
        GenerationDraft(answer="Advantages [E1]."),
    )
    # The judge says it passed while reporting unsupported claims: read as a failure, then repaired.
    disagreeing = FaithfulnessEvaluation.model_validate_json(
        '{"passed": true, "failure_type": "UNSUPPORTED_CLAIMS",'
        ' "verified_sources": ["E1"], "confidence": 0.9}'
    )
    model.queue(FaithfulnessEvaluation, disagreeing, _supported())
    model.queue(CompletenessEvaluation, _complete_evaluation(), _complete_evaluation())
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.COMPLETE
    assert response.metadata.generation_repaired


async def test_failed_repair_is_evaluated_and_its_attempt_carries_the_retry_decision(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Unsupported claim [E1]."),
        GenerationDraft(answer="Still unsupported [E1]."),
    )
    model.queue(FaithfulnessEvaluation, _unsupported(correcting=True), _unsupported())
    model.queue(CompletenessEvaluation, _complete_evaluation(), _complete_evaluation())
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.INSUFFICIENT_EVIDENCE
    trace = repository.traces[response.request_id]
    assert [attempt.generation_repair_attempted for attempt in trace.attempts] == [False, True]
    assert trace.generation_repair_used
    assert trace.attempts[0].decision is None
    last = trace.attempts[-1]
    assert last.decision is not None
    assert last.decision.hypothesis is RetryHypothesis.GENERATION_ONLY
    assert last.decision.next_strategy is None
    assert RagFailure.GENERATION_HALLUCINATION in last.failure_types


async def test_trace_keeps_complete_terminal_when_sources_fail_after_it(identity: Identity) -> None:
    repository = _SourcesUnavailableRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages [E1].")
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    with pytest.raises(ProviderError):
        await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    trace = next(iter(repository.traces.values()))
    assert trace.final_status is ResultStatus.COMPLETE
    assert trace.final_source_ids == (CHUNK_ID,)
    assert trace.degraded is False
    assert trace.error_code == "ProviderError"


async def test_trace_keeps_salvaged_partial_as_degraded_when_sources_fail(
    identity: Identity,
) -> None:
    repository = _SourcesUnavailableRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    queue_partial_rounds(model, rounds=1)
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=scripted_store(fail_from=2, action=_time_out),
    )

    with pytest.raises(ProviderError):
        await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    trace = next(iter(repository.traces.values()))
    assert trace.final_status is ResultStatus.VERIFIED_PARTIAL
    assert trace.final_source_ids == (CHUNK_ID,)
    assert trace.degraded is True
    assert trace.error_code == "ProviderError"


async def test_recorded_candidates_keep_per_list_ranks_with_none_for_single_list_hits(
    identity: Identity,
) -> None:
    both = CHUNK_ID
    dense_only = "00000000-0000-0000-0000-000000000002"
    sparse_only = "00000000-0000-0000-0000-000000000003"
    repository = FakeRepository((make_chunk(both), make_chunk(dense_only), make_chunk(sparse_only)))
    store = FakeVectorSearch()
    store.dense_hits = [
        SearchHit(chunk_id=dense_only, score=0.9, rank=1),
        SearchHit(chunk_id=both, score=0.5, rank=2),
    ]
    store.sparse_hits = [
        SearchHit(chunk_id=both, score=3.0, rank=1),
        SearchHit(chunk_id=sparse_only, score=1.0, rank=2),
    ]
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages [E1].")
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=store)

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.COMPLETE
    trace = repository.traces[response.request_id]
    candidates = {item.chunk_id: item for item in trace.attempts[0].candidates}
    assert set(candidates) == {both, dense_only, sparse_only}
    assert (candidates[both].dense_rank, candidates[both].sparse_rank) == (2, 1)
    assert (candidates[dense_only].dense_rank, candidates[dense_only].sparse_rank) == (1, None)
    assert (candidates[sparse_only].dense_rank, candidates[sparse_only].sparse_rank) == (None, 2)
