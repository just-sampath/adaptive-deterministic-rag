from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from handlers.policy import DecisionPolicy
from handlers.search import SearchHandler
from handlers.search_strategies.registry import build_strategy_registry
from models.domain import (
    MAX_CITATION_OCCURRENCES,
    AttemptOutcome,
    AttributedFaithfulness,
    Chunk,
    Citation,
    CompletenessEvaluation,
    CompletenessStatus,
    ContextualizationOutput,
    ContextualizationTrace,
    ConversationMessage,
    EncodedQuery,
    EvidenceRef,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    FollowupClassification,
    GenerationDraft,
    HybridHits,
    Identity,
    ModelUsage,
    QueryRecoveryOutput,
    QueryRequest,
    RagFailure,
    RankedChunk,
    RequestTrace,
    RequirementEvaluation,
    RequirementStatus,
    ResultStatus,
    RetrievalStrategy,
    RetryDecision,
    RetryHypothesis,
    RewriteDecision,
    SearchHit,
)
from providers.sparse import Bm25SparseEncoder
from services.citations import CitationRenderer
from services.context import ContextBuilder, ContextPacker, StructuralCompleter, utf8_byte_count
from services.contracts import FollowupDetector
from services.encoding import EmbeddingCache, HybridQueryEncoder
from services.evaluation import GenerationService, OnlineEvaluator
from services.followup import ModelFollowupDetector
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
from tests.test_transport import TRUSTED_HEADERS, make_app, make_context
from utils.errors import OperationTimeoutError, ProviderError, RequestDeadlineExceededError
from utils.prompts import STANDARD_ABSTENTION
from utils.settings import (
    ContentMode,
    ContextConfig,
    ContextualizationConfig,
    PolicyConfig,
    RetrievalConfig,
    Settings,
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


def _telemetry(repository: FakeRepository) -> TelemetryService:
    return TelemetryService(
        repository,
        TelemetryConfig(
            enabled=True,
            content_mode=ContentMode.METADATA_ONLY,
            write_query_intent_fields=True,
            write_accounting_fields=True,
        ),
        hash_key=b"test-telemetry-key",
    )


class _RecordingTelemetry(TelemetryService):
    """Keep emitted events in memory; the queue worker is not started in these tests."""

    def __init__(self, repository: FakeRepository) -> None:
        super().__init__(
            repository,
            TelemetryConfig(
                enabled=True, content_mode=ContentMode.METADATA_ONLY, write_accounting_fields=True
            ),
            hash_key=b"test-telemetry-key",
        )
        self.emitted: list[tuple[str, dict[str, object]]] = []

    def emit(self, event_type: str, **kwargs: Any) -> None:
        self.emitted.append((event_type, dict(kwargs.get("attributes") or {})))
        super().emit(event_type, **kwargs)

    def named(self, event_type: str) -> list[dict[str, object]]:
        return [attributes for name, attributes in self.emitted if name == event_type]


@pytest.mark.parametrize("returned", [FaithfulnessEvaluation, CompletenessEvaluation])
async def test_interrupted_judges_preserve_returned_calls(
    identity: Identity, returned: type
) -> None:
    finished = asyncio.Event()

    class InterruptedJudges(FakeTextModel):
        async def complete(self, **kwargs: Any) -> Any:
            schema = kwargs["output_type"]
            if (
                schema in (FaithfulnessEvaluation, CompletenessEvaluation)
                and schema is not returned
            ):
                await finished.wait()
                raise ProviderError("judge unavailable")
            result = await super().complete(**kwargs)
            if schema is returned:
                asyncio.get_running_loop().call_soon(finished.set)
            return replace(result, request_id=schema.__name__)

    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = InterruptedJudges()
    queue_complete_answer(model, "Answer [E1].")
    handler = build_orchestrator(model=model, repository=repository, vector_store=hit_store())
    with pytest.raises(ProviderError):
        await handler.query(identity, QueryRequest(message="What is supported?"))
    (trace,) = repository.traces.values()
    assert trace.attempted_strategies == (HAPPY,) and not trace.attempts
    assert trace.token_usage.total_tokens == 2 + 15 + 15
    assert {item.request_id for item in trace.model_calls} == {
        "embedding-1",
        "GenerationDraft",
        returned.__name__,
    }
    assert trace.usage_coverage == "partial"
    assert trace.operations is not None
    assert [item.slot for item in trace.operations] == list(range(1, len(trace.operations) + 1))
    assert all(item.strategy == HAPPY and item.strategy_try == 1 for item in trace.operations)
    assert sum(item.status == "failed" and item.usage is None for item in trace.operations) == 1


async def test_judge_cancelled_before_its_reply_has_unknown_usage(identity: Identity) -> None:
    class CancelledJudge(FakeTextModel):
        async def complete(self, **kwargs: Any) -> Any:
            if kwargs["output_type"] is FaithfulnessEvaluation:
                await asyncio.Event().wait()  # cancelled by the failing sibling
            if kwargs["output_type"] is CompletenessEvaluation:
                raise ProviderError("judge unavailable")
            return await super().complete(**kwargs)

    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = CancelledJudge()
    queue_complete_answer(model, "Answer [E1].")
    handler = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    with pytest.raises(ProviderError):
        await handler.query(identity, QueryRequest(message="What is supported?"))

    (trace,) = repository.traces.values()
    assert trace.operations is not None
    judges = {item.operation: item for item in trace.operations if "evaluation" in item.operation}
    assert judges["faithfulness_evaluation"].status == "cancelled"
    assert judges["faithfulness_evaluation"].usage is None
    assert judges["completeness_evaluation"].status == "failed"
    assert trace.token_usage.total_tokens == 2 + 15
    assert trace.usage_coverage == "partial"


async def test_repair_calls_carry_their_ordinal_and_totals_match_the_attempts(
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
    missing = CompletenessEvaluation(
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
    )
    model.queue(CompletenessEvaluation, missing, _complete_evaluation())
    handler = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    response = await handler.query(identity, QueryRequest(message="Explain the feature"))

    trace = repository.traces[response.request_id]
    assert trace.operations is not None
    stages = [(item.operation, item.repair_ordinal) for item in trace.operations]
    assert stages == [
        ("embedding", 0),
        ("rerank", 0),
        ("generation", 0),
        ("faithfulness_evaluation", 0),
        ("completeness_evaluation", 0),
        ("repair", 1),
        ("faithfulness_evaluation", 1),
        ("completeness_evaluation", 1),
    ]
    # The fake reranker reports no usage, so the request total is only partly observed.
    assert trace.usage_coverage == "partial"
    assert trace.token_usage.total_tokens == 2 + 6 * 15
    assert trace.token_usage == sum(
        (attempt.token_usage for attempt in trace.attempts), ModelUsage()
    )


async def test_concurrent_requests_keep_separate_operation_records(identity: Identity) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "First [E1].")
    queue_complete_answer(model, "Second [E1].")
    handler = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    responses = await asyncio.gather(
        handler.query(identity, QueryRequest(message="First question")),
        handler.query(identity, QueryRequest(message="Second question")),
    )

    for response in responses:
        trace = repository.traces[response.request_id]
        assert trace.operations is not None
        assert [item.slot for item in trace.operations] == [1, 2, 3, 4, 5]
        assert trace.token_usage.total_tokens == 2 + 3 * 15


def build_orchestrator(
    *,
    model: FakeTextModel,
    repository: FakeRepository,
    vector_store: FakeVectorSearch,
    timeouts: TimeoutConfig | None = None,
    detector: FollowupDetector | None = None,
    inline_citations: bool = True,
    reranker: FakeReranker | None = None,
    max_neighbors: int = 0,
    contextualization: ContextualizationConfig | None = None,
    history_ref_limit: int = 8,
    telemetry: TelemetryService | None = None,
    policy: PolicyConfig | None = None,
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
        max_neighbors_per_seed=max_neighbors,
    )
    contextualization = contextualization or ContextualizationConfig()
    # Most cases here queue unattributed verdicts; verified answers are opted into per test.
    policy = policy or PolicyConfig(verify_answers=False)
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
            query_drift_threshold=retrieval.query_drift_threshold,
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
            reranker=reranker if reranker is not None else FakeReranker(),
            completer=StructuralCompleter(
                repository=repository,
                max_neighbors_per_seed=max_neighbors,
                short_chunk_tokens=context.short_chunk_tokens,
            ),
            packer=ContextPacker(context, utf8_byte_count),
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
        telemetry=telemetry if telemetry is not None else _telemetry(repository),
        citations=CitationRenderer(inline=inline_citations),
        policy=policy,
        retrieval=retrieval,
        contextualization=contextualization,
        history_ref_limit=history_ref_limit,
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


def _attributed(
    *groups: tuple[str, ...], verified: tuple[str, ...] = ("E1",)
) -> AttributedFaithfulness:
    return AttributedFaithfulness.model_validate(
        {
            **_supported().model_dump(),
            "verified_sources": verified,
            "has_uncited_material_claims": False,
            "occurrence_assessments": [
                {
                    "occurrence_id": index,
                    "supported_labels": labels,
                    "attribution_status": "supported" if labels else "unsupported",
                }
                for index, labels in enumerate(groups, 1)
            ],
        }
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


def _assert_answer_targets(model: FakeTextModel, literal: str, accepted: str) -> None:
    for kind, prompt in model.output_prompts:
        if kind in (GenerationDraft, FaithfulnessEvaluation, CompletenessEvaluation):
            payload = json.loads(prompt)
            assert payload["literal_question"] == literal
            assert payload["accepted_question"] == accepted


class _StoppingJudge(FakeTextModel):
    """Judge stopping and trace storage independently, even when both facts are supported."""

    async def complete(self, **kwargs: Any) -> Any:
        if kwargs["output_type"] is CompletenessEvaluation:
            payload = json.loads(kwargs["user_prompt"])
            target = payload["literal_question"]
            term = "stop" if "stop" in target else "MongoDB"
            answered = term in payload["answer"]
            self.queue(
                CompletenessEvaluation,
                CompletenessEvaluation(
                    passed=answered,
                    failure_type=CompletenessStatus.COMPLETE
                    if answered
                    else CompletenessStatus.MISSING_ANSWER_CONTENT,
                    requirements=(
                        RequirementEvaluation(
                            requirement=target,
                            status=RequirementStatus.ANSWERED
                            if answered
                            else RequirementStatus.MISSING,
                            available_in_context=True,
                            supporting_chunks=("E1",),
                        ),
                    ),
                    missing_information_available=not answered,
                    confidence=1,
                ),
            )
        return await super().complete(**kwargs)


@pytest.mark.parametrize(
    ("proposed", "off_topic_draft"),
    [
        pytest.param("Where are request traces stored?", False, id="wrong-subject"),
        pytest.param("Where are request traces stored?", True, id="grounded-but-off-topic-draft"),
        pytest.param("What terminates the search process?", False, id="low-overlap-paraphrase"),
    ],
)
def test_literal_fallback_answers_the_callers_question(
    identity: Identity, make_settings: Callable[..., Settings], proposed: str, off_topic_draft: bool
) -> None:
    literal = "When does retrieval stop?"
    chunk = make_chunk(
        CHUNK_ID, text="Retrieval stops at the attempt limit. MongoDB stores traces."
    )
    repository = FakeRepository((chunk,))
    repository.messages[identity.tenant_id, identity.user_id, "guide"].extend(
        (
            ConversationMessage(role="user", content="Explain the search service"),
            ConversationMessage(
                role="assistant",
                content="It retrieves passages.",
                evidence_refs=(EvidenceRef.from_chunk(chunk),),
            ),
        )
    )
    store = hit_store()
    model = _StoppingJudge()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(
            standalone_question=proposed,
            relevant_conversation_context=("Find the MongoDB collection.",),
        ),
    )
    correct = GenerationDraft(answer="Retrieval stops at the attempt limit [E1].")
    if off_topic_draft:
        model.queue(GenerationDraft, GenerationDraft(answer="MongoDB stores traces [E1]."))
        model.queue(FaithfulnessEvaluation, _supported())
    model.queue(GenerationDraft, correct)
    model.queue(FaithfulnessEvaluation, _supported())
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=store,
        detector=FakeFollowupDetector(probability=1),
    )
    with TestClient(
        make_app(make_context(make_settings(), query=handler, repository=repository))
    ) as client:
        response = client.post(
            "/v1/rag/query",
            headers=TRUSTED_HEADERS,
            json={"message": literal, "conversation_id": "guide"},
        )
    assert response.status_code == 200
    answer = response.json()
    assert answer["status"] == ResultStatus.COMPLETE.value
    assert answer["answer"] == "Retrieval stops at the attempt limit [1]."
    assert answer["metadata"]["generation_repaired"] is off_topic_draft
    trace = repository.traces[answer["request_id"]].contextualization
    assert trace is not None and trace.rewrite_decision is RewriteDecision.REJECTED_DRIFT
    assert trace.proposed_drift_score == 1
    assert store.search_calls == 1 and store.last_query is not None
    assert store.last_query.text == literal
    assert _generation_payloads(model)[0]["relevant_conversation_facts"] == []
    assert model.calls.count(CompletenessEvaluation) == 1 + int(off_topic_draft)
    _assert_answer_targets(model, literal, literal)


@pytest.mark.parametrize(
    ("literal", "altered", "reordered"),
    [
        pytest.param(
            "Which records have retry_count <= 3 and batch_size <= 10?",
            "Which archive records have retry_count <= 10 and batch_size <= 3?",
            "Which archive records have batch_size <= 10 and retry_count <= 3?",
            id="field-limits",
        ),
        pytest.param(
            "Which records have cost <= $3 and price <= $10?",
            "Which archive records have cost <= $10 and price <= $3?",
            "Which archive records have $10 >= price and $3 >= cost?",
            id="currency-bounds",
        ),
        pytest.param(
            "Does it cost $50 per month?",
            "Does the archive plan cost $50 per year?",
            "Does the archive plan cost $50/month?",
            id="currency-rate-period",
        ),
        pytest.param(
            "Which plans have fee <= $50/month and seat <= $10/month?",
            "Which archive plans have seat <= $50 per month and fee <= $10 per month?",
            "Which archive plans have seat <= $10 per month and fee <= $50 per month?",
            id="currency-rate-bounds",
        ),
        pytest.param(
            "Which records have start < 2025-07-21 and end < 2026-07-22?",
            "Which archive records have start < 2026-07-22 and end < 2025-07-21?",
            "Which archive records have end < 2026-07-22 and start < 2025-07-21?",
            id="iso-date-bounds",
        ),
        pytest.param(
            'Which records have "from" <= 3 and "to" <= 10?',
            'Which archive records have "from" <= 10 and "to" <= 3?',
            'Which archive records have 10 >= "to" and 3 >= "from"?',
            id="quoted-fields",
        ),
        pytest.param(
            "Compare the archive deadlines July 21st, 2025 and July 22nd, 2026.",
            "Compare the migration archive deadlines July 21st, 2026 and July 22nd, 2025.",
            "Compare the migration archive deadlines July 22nd, 2026 and July 21st, 2025.",
            id="calendar-years",
        ),
        pytest.param(
            "What do the JSON keys `Tag7` and `OwnerID` mean in that schema?",
            "What do the JSON keys `Tag7` and `ownerId` mean in the archive schema?",
            "What do the JSON keys `OwnerID` and `Tag7` mean in the archive schema?",
            id="json-key-list",
        ),
        pytest.param(
            "Compare JSON keys Tag7, name and OwnerID in that archive.",
            "Compare JSON keys Tag7, name and ownerid in the migration archive.",
            "Compare JSON keys OwnerID, name and Tag7 in the migration archive.",
            id="label-as-key",
        ),
        pytest.param(
            'Explain the JSON keys "Tag7" and "OwnerID" and describe that archive.',
            'Describe the migration archive and explain the JSON keys "ownerid" and "Tag7".',
            'Describe the migration archive and explain the JSON keys "OwnerID" and "Tag7".',
            id="key-list-with-second-clause",
        ),
        pytest.param(
            "Explain the Tag7, name, and OwnerID JSON keys in that archive.",
            "Explain the Tag7, OwnerID, and Name JSON keys in the migration archive.",
            "Explain the Tag7, OwnerID, and name JSON keys in the migration archive.",
            id="label-as-preposed-key",
        ),
        pytest.param(
            "Are JSON keys Tag7 and name the required keys in that schema?",
            "Are JSON keys Tag7 and Name the required keys in the archive schema?",
            "Are JSON keys name and Tag7 the required keys in the archive schema?",
            id="key-before-predicate",
        ),
        pytest.param(
            "Explain JSON keys Tag7 and OwnerID and show examples from that archive.",
            "Show examples from the archive and explain JSON keys ownerid and Tag7.",
            "Show examples from the archive and explain JSON keys OwnerID and Tag7.",
            id="clause-after-complete-list",
        ),
        pytest.param(
            "Compare JSON keys Tag7, from, and OwnerID in that archive.",
            "Compare JSON keys from, Tag7, and ownerid in the migration archive.",
            "Compare JSON keys from, Tag7, and OwnerID in the migration archive.",
            id="stopword-as-list-item",
        ),
    ],
)
@pytest.mark.parametrize("preserved", [False, True], ids=["altered", "reordered"])
async def test_query_sent_to_retrieval_preserves_structured_terms(
    identity: Identity, literal: str, altered: str, reordered: str, preserved: bool
) -> None:
    chunk = make_chunk(CHUNK_ID, text="The guide covers the migration archive.")
    repository = FakeRepository((chunk,))
    repository.messages[identity.tenant_id, identity.user_id, "guide"].extend(
        (
            ConversationMessage(role="user", content="Which archive does the guide cover?"),
            ConversationMessage(
                role="assistant",
                content="The migration archive.",
                evidence_refs=(EvidenceRef.from_chunk(chunk),),
            ),
        )
    )
    store = hit_store()
    model = FakeTextModel()
    proposal = reordered if preserved else altered
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question=proposal))
    queue_complete_answer(model, "The guide covers the migration archive [E1].")
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=store,
        detector=FakeFollowupDetector(probability=1),
    )

    response = await handler.query(identity, QueryRequest(message=literal, conversation_id="guide"))

    expected_query = reordered if preserved else literal
    assert store.search_calls == 1 and store.last_query is not None
    assert store.last_query.text == expected_query
    trace = repository.traces[response.request_id].contextualization
    assert trace is not None
    assert trace.rewrite_decision is (
        RewriteDecision.ACCEPTED if preserved else RewriteDecision.REJECTED_ANCHOR
    )
    _assert_answer_targets(model, literal, expected_query)


async def test_changed_filing_date_cannot_hide_available_evidence(identity: Identity) -> None:
    class FilingDateIndex(FakeVectorSearch):
        async def search(
            self,
            query: EncodedQuery,
            identity: Identity,
            filters: dict[str, str | int | float | bool],
            *,
            dense_k: int,
            sparse_k: int,
        ) -> HybridHits:
            if "July 21st" in query.text:
                self.dense_hits = [SearchHit(chunk_id=CHUNK_ID, score=1, rank=1)]
            else:
                self.dense_hits = []
            return await super().search(
                query, identity, filters, dense_k=dense_k, sparse_k=sparse_k
            )

    literal = "For that archive, is the filing deadline July 21st?"
    chunk = make_chunk(CHUNK_ID, text="The migration archive filing deadline is July 21st.")
    repository = FakeRepository((chunk,))
    repository.messages[identity.tenant_id, identity.user_id, "archive"].extend(
        (
            ConversationMessage(role="user", content="Which archive?"),
            ConversationMessage(
                role="assistant",
                content="The migration archive.",
                evidence_refs=(EvidenceRef.from_chunk(chunk),),
            ),
        )
    )
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(
            standalone_question="For the migration archive, is the filing deadline July 22nd?"
        ),
    )
    model.queue(
        QueryRecoveryOutput, QueryRecoveryOutput(keyword_query="migration filing July 22nd")
    )
    queue_complete_answer(model, "The filing deadline is July 21st [E1].")
    store = FilingDateIndex()
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=store,
        detector=FakeFollowupDetector(probability=1),
    )

    response = await handler.query(
        identity, QueryRequest(message=literal, conversation_id="archive")
    )

    assert response.status is ResultStatus.COMPLETE
    assert response.answer == "The filing deadline is July 21st [1]."
    assert store.search_calls == 1 and store.last_query is not None
    assert store.last_query.text == literal
    _assert_answer_targets(model, literal, literal)


class _SourcesUnavailableRepository(FakeRepository):
    """Authorized retrieval works, but the final source lookup fails."""

    async def hydrate_authorized_chunks(
        self, chunk_ids: Sequence[str], identity: Identity
    ) -> dict[str, Chunk]:
        if self.hydrate_calls:
            raise ProviderError("document store unavailable")
        return await super().hydrate_authorized_chunks(chunk_ids, identity)


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


@pytest.mark.parametrize("draft", ["", "   ", "[E1].", "[E1, E2] … !"])
async def test_invalid_draft_skips_judges_and_abstains(identity: Identity, draft: str) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, draft)
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        policy=PolicyConfig(max_strategies=1, verify_answers=True),
    )
    response = await handler.query(identity, QueryRequest(message="Explain the feature"))
    assert response.status is ResultStatus.INSUFFICIENT_EVIDENCE
    assert model.calls == [GenerationDraft]
    (attempt,) = repository.traces[response.request_id].attempts
    assert attempt.faithfulness is None and attempt.completeness is None
    assert attempt.token_usage.total_tokens > 0
    assert attempt.decision and attempt.decision.hypothesis is RetryHypothesis.UNCLASSIFIED


def test_malformed_complete_verdict_returns_503(
    identity: Identity, make_settings: Callable[..., Settings]
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(GenerationDraft, GenerationDraft(answer="No [E1]."))
    model.queue(AttributedFaithfulness, _attributed(("E1",)))
    model.queue(
        CompletenessEvaluation, _complete_evaluation().model_copy(update={"requirements": ()})
    )
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        policy=PolicyConfig(verify_answers=True),
    )
    with TestClient(
        make_app(make_context(make_settings(), query=handler, repository=repository))
    ) as client:
        response = client.post(
            "/v1/rag/query", headers=TRUSTED_HEADERS, json={"message": "Is it supported?"}
        )
    assert response.status_code == 503


@pytest.mark.parametrize(
    ("repaired", "expected"),
    [
        ("No [E1].", ResultStatus.COMPLETE),
        ("0 [E1]", ResultStatus.COMPLETE),
        ("[E1].", ResultStatus.INSUFFICIENT_EVIDENCE),
        ("No " + "[E1]" * (MAX_CITATION_OCCURRENCES + 1), ResultStatus.INSUFFICIENT_EVIDENCE),
    ],
)
async def test_swapped_citations_need_a_new_verified_draft(
    identity: Identity, repaired: str, expected: ResultStatus
) -> None:
    second = "00000000-0000-0000-0000-000000000002"
    repository = FakeRepository(
        (make_chunk(CHUNK_ID, text="Tea"), make_chunk(second, text="Coffee", document_id="doc-b"))
    )
    store = hit_store()
    store.dense_hits.append(SearchHit(chunk_id=second, score=0.8, rank=2))
    model = FakeTextModel()
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Tea [E2]. Coffee [E1]."),
        GenerationDraft(answer=repaired),
    )
    model.queue(
        AttributedFaithfulness, _attributed((), (), verified=("E1", "E2")), _attributed(("E1",))
    )
    model.queue(CompletenessEvaluation, _complete_evaluation(), _complete_evaluation())
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=store,
        policy=PolicyConfig(verify_answers=True, max_strategies=1),
    )
    response = await handler.query(identity, QueryRequest(message="What is served?"))
    assert response.status is expected
    trace = repository.traces[response.request_id]
    original, repair = trace.attempts
    assert original.faithfulness and not original.faithfulness.passed
    assert not original.faithfulness.verified_sources
    assert trace.generation_repair_used and repair.generation_repair_attempted
    faith_prompts = [
        json.loads(p) for typ, p in model.output_prompts if typ is AttributedFaithfulness
    ]
    assert [row["claimed_labels"] for row in faith_prompts[0]["citation_occurrences"]] == [
        ["E2"],
        ["E1"],
    ]
    assert [row["evidence_label"] for row in faith_prompts[0]["evidence"]] == ["E1", "E2"]
    if expected is ResultStatus.COMPLETE:
        assert trace.final_source_ids == (CHUNK_ID,)
        assert response.citations == (CITATION,)
        assert [row["occurrence_id"] for row in faith_prompts[1]["citation_occurrences"]] == [1]
    else:
        assert (
            len(faith_prompts) == 1 and repair.faithfulness is None and repair.completeness is None
        )
        assert repair.outcome in {AttemptOutcome.EMPTY_DRAFT, AttemptOutcome.CITATION_LIMIT}
        assert response.sources == () and response.citations == ()


@pytest.mark.parametrize("verify", [False, True])
async def test_orchestrator_exhausts_each_retrieval_strategy_once_then_abstains(
    identity: Identity,
    verify: bool,
) -> None:
    repository = FakeRepository()
    vector_store = FakeVectorSearch()
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput(keyword_query="exact identifier"))
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=vector_store,
        policy=PolicyConfig(verify_answers=verify),
    )

    response = await orchestrator.query(identity, QueryRequest(message="Unknown fact"))

    assert response.status is ResultStatus.INSUFFICIENT_EVIDENCE
    assert response.metadata.attempts == 3
    trace = repository.traces[response.request_id]
    assert len(trace.attempted_strategies) == len(set(trace.attempted_strategies)) == 3
    assert not trace.generation_repair_used
    decisions = _decisions(trace)
    assert [decision.hypothesis for decision in decisions] == [RetryHypothesis.LOW_RECALL] * 3
    assert [decision.next_strategy for decision in decisions] == [RECALL, RECOVERY, None]
    assert trace.attempted_strategies == (HAPPY, RECALL, RECOVERY)
    assert model.calls == [QueryRecoveryOutput]
    assert [item.failure_types for item in trace.attempts] == [(RagFailure.LOW_RECALL,)] * 3
    if verify:
        assert all(
            item.outcome is AttemptOutcome.EMPTY_EVIDENCE
            and item.faithfulness is None
            and item.completeness is None
            for item in trace.attempts
        )


class _HandbookReranker(FakeReranker):
    async def rerank(
        self, query: str, chunks: Sequence[RankedChunk], top_n: int
    ) -> list[RankedChunk]:
        assert [r.chunk.id for r in chunks] == [CHUNK_ID]
        return await super().rerank(query, chunks, top_n)


def _bilingual_handbook() -> tuple[FakeRepository, FakeVectorSearch]:
    seed = make_chunk(CHUNK_ID, text="English summary:", next_id="following").model_copy(
        update={"metadata": {"lang": "en"}}
    )
    other = make_chunk("other", text="French summary").model_copy(
        update={"metadata": {"lang": "fr"}}
    )
    following = other.model_copy(update={"id": "following", "text": "French continuation"})
    store = hit_store()
    store.dense_hits.append(SearchHit(chunk_id=other.id, score=0.5, rank=2))
    return FakeRepository((seed, other, following)), store


def _assert_english_evidence(model: FakeTextModel) -> None:
    for kind, prompt in model.output_prompts:
        if kind in (GenerationDraft, FaithfulnessEvaluation, CompletenessEvaluation):
            assert [item["content"] for item in json.loads(prompt)["evidence"]] == [
                "English summary:"
            ]


async def test_partial_answer_after_exhausting_strategies_discloses_unanswered_requirements(
    identity: Identity,
) -> None:
    repository, vector_store = _bilingual_handbook()
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput(keyword_query="handbook capabilities"))
    queue_partial_rounds(model, rounds=3)
    telemetry = _RecordingTelemetry(repository)
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=vector_store,
        reranker=_HandbookReranker(),
        max_neighbors=1,
        telemetry=telemetry,
    )

    response = await orchestrator.query(
        identity, QueryRequest(filters={"lang": "en"}, message="Explain the feature")
    )

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.answer == (
        "Advantages [1].\n\nI could not verify from the available sources: limitations."
    )
    assert response.unanswered_requirements == ("limitations",)
    assert response.metadata.degraded is False
    _assert_english_evidence(model)
    # The fake vector store ignores filters, so the French hit stands in for store drift.
    mismatches = telemetry.named("rag.retrieval.metadata_mismatch")
    assert [event["strategy"] for event in mismatches] == [
        HAPPY.value,
        RECOVERY.value,
        RECALL.value,
    ]
    # Recovery ran both the accepted query and its distinct keyword variant.
    assert [event["count"] for event in mismatches] == [1, 2, 1]
    trace = repository.traces[response.request_id]
    assert trace.attempted_strategies == (HAPPY, RECOVERY, RECALL)
    _assert_answer_targets(model, "Explain the feature", "Explain the feature")
    assert trace.final_status is ResultStatus.VERIFIED_PARTIAL
    assert trace.degraded is False
    assert trace.error_code is None
    assert trace.handler_outcome == "returned"
    assert trace.returned_status is ResultStatus.VERIFIED_PARTIAL
    assert trace.returned_document_ids == tuple(source.document_id for source in response.sources)


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


async def _transport_time_out() -> None:
    raise OperationTimeoutError("Qdrant operation timed out")


async def test_adapter_timeout_without_an_answer_is_an_operation_timeout_not_a_deadline(
    identity: Identity,
) -> None:
    repository = FakeRepository()
    orchestrator = build_orchestrator(
        model=FakeTextModel(),
        repository=repository,
        vector_store=scripted_store(fail_from=1, action=_transport_time_out),
    )

    with pytest.raises(OperationTimeoutError):
        await orchestrator.query(identity, QueryRequest(message="Question"))

    trace = next(iter(repository.traces.values()))
    assert trace.error_code == "OperationTimeoutError"
    assert trace.final_status is None


async def test_adapter_timeout_after_a_verified_answer_is_salvaged_like_a_stage_timeout(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    queue_partial_rounds(model, rounds=1)
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=scripted_store(fail_from=2, action=_transport_time_out),
    )

    response = await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.metadata.degraded is True
    trace = repository.traces[response.request_id]
    assert trace.error_code == "OperationTimeoutError"
    assert trace.degraded is True


class _FinalReadTimesOutRepository(FakeRepository):
    """Authorized retrieval works, but the final evidence read times out at the adapter."""

    async def hydrate_authorized_chunks(
        self, chunk_ids: Sequence[str], identity: Identity
    ) -> dict[str, Chunk]:
        if self.hydrate_calls:
            raise OperationTimeoutError("MongoDB operation timed out")
        return await super().hydrate_authorized_chunks(chunk_ids, identity)


async def test_adapter_timeout_in_the_final_evidence_read_prevents_delivery(
    identity: Identity,
) -> None:
    repository = _FinalReadTimesOutRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages and limitations [E1].")
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    with pytest.raises(OperationTimeoutError):
        await orchestrator.query(identity, QueryRequest(message="Explain the feature"))

    trace = next(iter(repository.traces.values()))
    assert trace.final_status is ResultStatus.COMPLETE
    assert trace.error_code == "OperationTimeoutError"


@pytest.mark.parametrize("inline", [True, False])
def test_partial_disclosure_survives_http_storage_and_the_next_model_prompts(
    make_settings: Callable[..., Settings], inline: bool
) -> None:
    chunk = make_chunk(CHUNK_ID)
    repository = FakeRepository((chunk,))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    draft = "Read `items[0]` and note [42] [E1]."
    for _ in range(3):
        model.queue(GenerationDraft, GenerationDraft(answer=draft))
        model.queue(FaithfulnessEvaluation, _supported())
        model.queue(CompletenessEvaluation, _partial_evaluation())
    model.queue(FollowupClassification, FollowupClassification(is_followup_probability=1))
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question="Which note?"))
    queue_complete_answer(model, "The note applies [E1].")
    handler = build_orchestrator(
        repository=repository,
        model=model,
        vector_store=hit_store(),
        inline_citations=inline,
        detector=ModelFollowupDetector(model=model, model_name="detector", max_output_tokens=128),
    )
    context = make_context(make_settings(), query=handler, repository=repository)
    request = {"message": "Explain the feature", "conversation_id": "notes"}
    with TestClient(make_app(context)) as client:
        first = client.post("/v1/rag/query", headers=TRUSTED_HEADERS, json=request)
        assert first.status_code == 200
        answer = first.json()
        assert answer["status"] == "VERIFIED_PARTIAL"
        assert answer["unanswered_requirements"] == ["limitations"]
        saved = repository.messages[("tenant-a", "user-a", "notes")][-1]
        plain = "Read `items[0]` and note [42]."
        suffix = "\n\nI could not verify from the available sources: limitations."
        assert saved.content == plain + suffix
        assert saved.evidence_refs == (EvidenceRef.from_chunk(chunk),)
        displayed = "Read `items[0]` and note [42] [1]." if inline else plain
        assert answer["answer"] == displayed + suffix
        second = client.post(
            "/v1/rag/query", headers=TRUSTED_HEADERS, json={**request, "message": "Which note?"}
        )
        assert second.status_code == 200
    payloads = {kind: json.loads(prompt) for kind, prompt in model.output_prompts}
    detector_lines = [
        item["message"] for item in payloads[FollowupClassification]["conversation_history"]
    ]
    assert detector_lines == payloads[ContextualizationOutput]["conversation_history"]
    assert detector_lines == ["user: Explain the feature", f"assistant: {saved.content}"]


async def test_conversation_history_is_contextualized_recorded_and_extended(
    identity: Identity,
) -> None:
    repository = FakeRepository(
        (make_chunk(CHUNK_ID, text="Discovery comes first. Migration proceeds in stages."),)
    )
    key = (identity.tenant_id, identity.user_id, "conversation-1")
    repository.messages[key].extend(
        [
            ConversationMessage(role="user", content="Which two processes are documented?"),
            ConversationMessage(
                role="assistant",
                content="Discovery comes first and migration comes second.",
                evidence_refs=(EvidenceRef.from_chunk(repository.chunks[CHUNK_ID]),),
            ),
        ]
    )
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="How does migration work?"),
    )
    queue_complete_answer(model, "Migration proceeds in stages [E1].")
    store = hit_store()
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=store,
        detector=FakeFollowupDetector(probability=1),
    )

    response = await orchestrator.query(
        identity,
        QueryRequest(message="What about the second one?", conversation_id="conversation-1"),
    )

    assert response.status is ResultStatus.COMPLETE
    trace = repository.traces[response.request_id]
    assert trace.contextualization == ContextualizationTrace(
        enabled=True,
        detector="fake",
        followup_probability=1,
        fail_open=False,
        history_messages=2,
        selected_messages=2,
        rewritten=True,
        proposed_drift_score=0,
        rewrite_decision=RewriteDecision.ACCEPTED,
    )
    assert repository.messages[key][-2].content == "What about the second one?"
    assert repository.messages[key][-1].content == "Migration proceeds in stages."
    assert response.answer == "Migration proceeds in stages [1]."
    assert store.last_query is not None and store.last_query.text == "How does migration work?"
    _assert_answer_targets(model, "What about the second one?", "How does migration work?")


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


@pytest.mark.parametrize("case", ["current", "changed", "too_many"])
async def test_final_read_and_saved_turn_include_dependencies_from_history(
    identity: Identity, case: str
) -> None:
    previous, current = make_chunk("earlier", document_id="earlier-doc"), make_chunk(CHUNK_ID)

    class Catalog(FakeRepository):
        async def hydrate_authorized_chunks(
            self, chunk_ids: Sequence[str], identity: Identity
        ) -> dict[str, Chunk]:
            if case == "changed" and {current.id, previous.id} == set(chunk_ids):
                self.chunks[previous.id] = previous.model_copy(update={"text": "Revised edition."})
            return await super().hydrate_authorized_chunks(chunk_ids, identity)

    repository = Catalog((previous, current))
    key = (identity.tenant_id, identity.user_id, "guide")
    repository.messages[key].extend(
        (
            ConversationMessage(role="user", content="Which edition?"),
            ConversationMessage(
                role="assistant",
                content="Previous edition.",
                evidence_refs=(EvidenceRef.from_chunk(previous),),
            ),
        )
    )
    model = FakeTextModel()
    queue_complete_answer(model, "Current edition [E1].")
    detector = FakeFollowupDetector(probability=0)
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        detector=detector,
        history_ref_limit=1 if case == "too_many" else 2,
    )
    request = QueryRequest(message="Next edition?", conversation_id="guide")
    response = await handler.query(identity, request)
    assert detector.detect_calls == 1
    assert set(repository.hydrate_calls[-1]) == {current.id, previous.id}
    saved = repository.messages[key][-1]
    if case == "changed":
        assert response.status is ResultStatus.INSUFFICIENT_EVIDENCE
        assert response.unanswered_requirements == (request.message,)
        assert response.sources == ()
        assert response.citations == ()
        assert saved.content == STANDARD_ABSTENTION
        assert saved.evidence_refs == ()
    else:
        assert response.status is ResultStatus.COMPLETE
        assert [source.document_id for source in response.sources] == [current.document_id]
        if case == "too_many":
            assert saved.evidence_refs is None
            queue_complete_answer(model, "Current edition [E1].")
            await handler.query(identity, request)
            assert detector.detect_calls == 1
        else:
            assert saved.evidence_refs is not None
            assert set(saved.evidence_refs) == {
                EvidenceRef.from_chunk(current),
                EvidenceRef.from_chunk(previous),
            }


async def test_disabled_contextualization_skips_history_but_still_writes_the_turn(
    identity: Identity,
) -> None:
    class WriteOnly(FakeRepository):
        async def get_messages(
            self, identity: Identity, conversation_id: str, limit: int
        ) -> tuple[ConversationMessage, ...]:
            pytest.fail("Disabled contextualization must skip the read")

    current = make_chunk(CHUNK_ID)
    repository = WriteOnly((current,))
    model = FakeTextModel()
    queue_complete_answer(model, "Current edition [E1].")
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        contextualization=ContextualizationConfig(enabled=False),
    )
    await handler.query(identity, QueryRequest(message="Which edition?", conversation_id="guide"))
    user, assistant = repository.messages[identity.tenant_id, identity.user_id, "guide"]
    assert user.content == "Which edition?"
    assert assistant.content == "Current edition."
    assert assistant.evidence_refs == (EvidenceRef.from_chunk(current),)


async def test_concurrent_users_reusing_a_conversation_id_get_separate_history(
    identity: Identity,
) -> None:
    current = make_chunk(CHUNK_ID)
    users = (identity, identity.model_copy(update={"user_id": "other-user"}))
    history_chunks = [make_chunk(user.user_id, document_id=user.user_id) for user in users]
    repository = FakeRepository((current, *history_chunks))
    model = FakeTextModel()
    for user, history_chunk in zip(users, history_chunks, strict=True):
        repository.messages[user.tenant_id, user.user_id, "shared"].extend(
            (
                ConversationMessage(role="user", content="Which guide?"),
                ConversationMessage(
                    role="assistant",
                    content=user.user_id,
                    evidence_refs=(EvidenceRef.from_chunk(history_chunk),),
                ),
            )
        )
        model.queue(FollowupClassification, FollowupClassification(is_followup_probability=0))
        queue_complete_answer(model, "Current edition [E1].")
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        detector=ModelFollowupDetector(model=model, model_name="detector", max_output_tokens=128),
    )
    await asyncio.gather(
        *(
            handler.query(
                user, QueryRequest(message=f"Guide for {user.user_id}?", conversation_id="shared")
            )
            for user in users
        )
    )
    prompts = [
        json.loads(body) for kind, body in model.output_prompts if kind is FollowupClassification
    ]
    assert len(prompts) == 2
    for user, history_chunk in zip(users, history_chunks, strict=True):
        (prompt,) = [p for p in prompts if p["current_message"] == f"Guide for {user.user_id}?"]
        assert [row["message"] for row in prompt["conversation_history"]] == [
            "user: Which guide?",
            f"assistant: {user.user_id}",
        ]
        messages = repository.messages[user.tenant_id, user.user_id, "shared"]
        assert len(messages) == 4
        refs = messages[-1].evidence_refs
        assert refs is not None
        assert set(refs) == {EvidenceRef.from_chunk(current), EvidenceRef.from_chunk(history_chunk)}


async def test_contaminated_generation_is_retried_in_isolation_without_repeating_retrieval(
    identity: Identity,
) -> None:
    repository, vector_store = _bilingual_handbook()
    repository.messages[identity.tenant_id, identity.user_id, "conversation-1"].extend(
        [
            ConversationMessage(role="user", content="Tell me about migration"),
            ConversationMessage(
                role="assistant",
                content="Migration is staged.",
                evidence_refs=(EvidenceRef.from_chunk(repository.chunks[CHUNK_ID]),),
            ),
        ]
    )
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(
            standalone_question="Explain the migration feature",
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
    orchestrator = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=vector_store,
        reranker=_HandbookReranker(),
        max_neighbors=1,
        detector=FakeFollowupDetector(probability=1),
    )

    response = await orchestrator.query(
        identity,
        QueryRequest(
            filters={"lang": "en"}, message="Explain the feature", conversation_id="conversation-1"
        ),
    )

    assert response.status is ResultStatus.COMPLETE
    _assert_english_evidence(model)
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
    _assert_answer_targets(model, "Explain the feature", "Explain the migration feature")
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


@pytest.mark.parametrize("drop_events", [False, True], ids=["transport-event", "events-dropped"])
def test_trace_keeps_complete_terminal_when_sources_fail_after_it(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, drop_events: bool
) -> None:
    repository = _SourcesUnavailableRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages [E1].")
    orchestrator = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    context = make_context(make_settings(), repository=repository, query=orchestrator)
    if drop_events:
        monkeypatch.setattr(context.telemetry, "emit", lambda *args, **kwargs: None)
    with TestClient(make_app(context)) as client:
        response = client.post(
            "/v1/rag/query", headers=TRUSTED_HEADERS, json={"message": "Explain the feature"}
        )
    assert response.status_code == 503
    trace = repository.traces[response.headers["x-request-id"]]
    events = context.telemetry.named("http.request.completed")
    if drop_events:
        assert not events
    else:
        (event,) = events
        assert event.request_id == trace.request_id
        assert event.attributes["status_code"] == 503
        assert event.attributes["response_completed"] is True
    assert trace.final_status is ResultStatus.COMPLETE
    assert trace.final_source_ids == (CHUNK_ID,)
    assert trace.handler_outcome == "failed"
    assert trace.returned_status is None
    assert trace.returned_document_ids is None
    assert trace.degraded is False
    assert trace.error_code == "ProviderError"


async def test_trace_records_cancelled_history_write(identity: Identity) -> None:
    writing = asyncio.Event()

    class PausedHistory(FakeRepository):
        async def append_messages(
            self, identity: Identity, conversation_id: str, messages: Sequence[ConversationMessage]
        ) -> None:
            writing.set()
            await asyncio.Event().wait()

    repository = PausedHistory((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages [E1].")
    handler = build_orchestrator(model=model, repository=repository, vector_store=hit_store())
    task = asyncio.create_task(
        handler.query(
            identity, QueryRequest(message="Explain the feature", conversation_id="guide")
        )
    )
    try:
        await asyncio.wait_for(writing.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    (trace,) = repository.traces.values()
    assert trace.handler_outcome == "cancelled"
    assert trace.final_status is ResultStatus.COMPLETE
    assert trace.returned_status is None and trace.returned_document_ids is None
    assert trace.error_code == "CancelledError"


async def test_response_construction_failure_is_in_the_saved_trace(
    identity: Identity, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    queue_complete_answer(model, "Advantages [E1].")
    handler = build_orchestrator(model=model, repository=repository, vector_store=hit_store())

    def fail_response(*args: object) -> None:
        raise ValueError("invalid response")

    monkeypatch.setattr(handler, "_response", fail_response)
    with pytest.raises(ValueError, match="invalid response"):
        await handler.query(identity, QueryRequest(message="Explain the feature"))
    (trace,) = repository.traces.values()
    assert trace.final_status is ResultStatus.COMPLETE
    assert trace.handler_outcome == "failed"
    assert trace.returned_status is None and trace.returned_document_ids is None
    assert trace.error_code == "ValueError"


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


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "repaired",
        "text",
        "heading",
        "version",
        "metadata",
        "still_eligible",
        "unchanged",
    ],
)
async def test_final_evidence_uses_every_packed_passage(identity: Identity, change: str) -> None:
    first = make_chunk(CHUNK_ID).model_copy(update={"metadata": {"lang": "en"}})
    sibling = first.model_copy(update={"id": "sibling", "heading_path": ("Details",)})

    class UpdatedCatalog(FakeRepository):
        async def hydrate_authorized_chunks(
            self, chunk_ids: Sequence[str], identity: Identity
        ) -> dict[str, Chunk]:
            if self.hydrate_calls:
                updates: dict[str, dict[str, object]] = {
                    "text": {"text": "Revised edition."},
                    "heading": {"heading_path": ("Appendix",)},
                    "version": {"document_version_id": "v2"},
                    "metadata": {"metadata": {"lang": "fr"}},
                    "still_eligible": {
                        "metadata": {"lang": "en", "edition": 2},
                        "allowed_roles": identity.roles | {"editor"},
                    },
                }
                if change in {"missing", "repaired"}:
                    self.chunks.pop(sibling.id, None)
                elif change in updates:
                    self.chunks[sibling.id] = sibling.model_copy(update=updates[change])
            return await super().hydrate_authorized_chunks(chunk_ids, identity)

    repository = UpdatedCatalog((first, sibling))
    key = (identity.tenant_id, identity.user_id, "guide")
    repository.messages[key].extend(
        (
            ConversationMessage(role="user", content="Handbooks"),
            ConversationMessage(role="assistant", content=STANDARD_ABSTENTION, evidence_refs=()),
        )
    )
    store = hit_store()
    store.dense_hits.append(SearchHit(chunk_id=sibling.id, score=0.5, rank=2))
    model = FakeTextModel()
    cited: tuple[str, ...] = (first.id,)
    if change == "repaired":
        model.queue(GenerationDraft, GenerationDraft(answer="First draft."))
        model.queue(FaithfulnessEvaluation, _unsupported(correcting=True))
        model.queue(CompletenessEvaluation, _complete_evaluation())
    if change == "missing":
        model.queue(GenerationDraft, GenerationDraft(answer="The handbook applies [E1, E2]."))
        model.queue(
            FaithfulnessEvaluation,
            _supported().model_copy(update={"verified_sources": ("E1", "E2")}),
        )
        model.queue(CompletenessEvaluation, _complete_evaluation())
        cited = (first.id, sibling.id)
    else:
        queue_complete_answer(model, "The handbook applies [E1].")
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="Which handbook applies to the new edition?"),
    )
    telemetry = _RecordingTelemetry(repository)
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=store,
        detector=FakeFollowupDetector(probability=1),
        telemetry=telemetry,
    )
    request = QueryRequest(
        message="Which handbook applies?", conversation_id="guide", filters={"lang": "en"}
    )
    response = await handler.query(identity, request)
    trace = repository.traces[response.request_id]
    assert trace.final_status is ResultStatus.COMPLETE
    _assert_answer_targets(model, request.message, "Which handbook applies to the new edition?")
    assert trace.final_source_ids == cited
    assert trace.handler_outcome == "returned"
    assert trace.returned_status is response.status
    assert trace.returned_document_ids == tuple(source.document_id for source in response.sources)
    assert response.metadata.generation_repaired is (change == "repaired")
    assert len(_generation_payloads(model)) == (2 if change == "repaired" else 1)
    assert repository.hydrate_calls == [(first.id, sibling.id), (first.id, sibling.id)]
    if change in {"unchanged", "still_eligible"}:
        assert response.status is ResultStatus.COMPLETE
        assert response.citations == (CITATION,)
        assert [s.document_id for s in response.sources] == [first.document_id]
        assert telemetry.named("rag.evidence.withdrawn") == []
    else:
        # A superseded version is no longer current, so hydration omits it like a deletion.
        unavailable = ("missing", "repaired", "version")
        reason = dict.fromkeys(unavailable, "unavailable") | {"metadata": "metadata"}
        assert telemetry.named("rag.evidence.withdrawn") == [
            {
                "reason": reason.get(change, "changed"),
                "selected_status": "COMPLETE",
                "strategy": HAPPY.value,
                "packed_passages": 2,
            }
        ]
        assert response.status is ResultStatus.INSUFFICIENT_EVIDENCE
        assert response.unanswered_requirements == (request.message,)
        assert response.sources == ()
        assert response.citations == ()
        assert repository.messages[key][-1].content == response.answer
        assert repository.messages[key][-1].evidence_refs == ()
        assert "handbook applies" not in response.answer


@pytest.mark.parametrize("degraded", [False, True])
async def test_selected_partial_rechecks_its_own_evidence(
    identity: Identity, degraded: bool
) -> None:
    first = make_chunk(CHUNK_ID)
    later = make_chunk("later-edition")
    repository = FakeRepository((first, later))

    async def switch_edition() -> None:
        repository.chunks.pop(first.id, None)
        if degraded:
            raise TimeoutError
        store.dense_hits = [SearchHit(chunk_id=later.id, score=1, rank=1)]
        store.sparse_hits = []

    store = scripted_store(fail_from=2, action=switch_edition)
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    queue_partial_rounds(model, rounds=1 if degraded else 3)
    telemetry = _RecordingTelemetry(repository)
    handler = build_orchestrator(
        model=model, repository=repository, vector_store=store, telemetry=telemetry
    )
    response = await handler.query(
        identity, QueryRequest(message="Explain the feature", conversation_id="guide")
    )
    (withdrawn,) = telemetry.named("rag.evidence.withdrawn")
    assert withdrawn["reason"] == "unavailable"
    assert withdrawn["selected_status"] == "VERIFIED_PARTIAL"

    trace = repository.traces[response.request_id]
    assert trace.final_status is ResultStatus.VERIFIED_PARTIAL
    assert trace.degraded is degraded
    assert trace.final_source_ids == (first.id,)
    assert repository.hydrate_calls[-1] == (first.id,)
    assert response.status is ResultStatus.INSUFFICIENT_EVIDENCE
    assert response.sources == ()
    assert response.citations == ()
    assert (
        repository.messages[identity.tenant_id, identity.user_id, "guide"][-1].content
        == response.answer
    )


async def test_blank_repair_cannot_replace_an_earlier_verified_partial(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(GenerationDraft, GenerationDraft(answer="Advantages [E1]."))
    model.queue(AttributedFaithfulness, _attributed(("E1",)))
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=False,
            failure_type=CompletenessStatus.MISSING_ANSWER_CONTENT,
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
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
            missing_information_available=True,
            confidence=1,
        ),
    )
    model.queue(GenerationDraft, GenerationDraft(answer=" [E1] "))
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        policy=PolicyConfig(verify_answers=True, max_strategies=1),
    )

    response = await handler.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.answer.startswith("Advantages [1].")
    assert response.unanswered_requirements == ("limitations",)
    trace = repository.traces[response.request_id]
    original, repair = trace.attempts
    assert original.outcome is AttemptOutcome.EVALUATED
    assert repair.outcome is AttemptOutcome.EMPTY_DRAFT
    assert repair.generation_repair_attempted and trace.generation_repair_used
    assert model.calls.count(AttributedFaithfulness) == 1


async def test_malformed_complete_verdict_salvages_an_earlier_verified_partial(
    identity: Identity,
) -> None:
    repository = FakeRepository((make_chunk(CHUNK_ID),))
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput())
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="Advantages [E1]."),
        GenerationDraft(answer="Advantages and limitations [E1]."),
    )
    model.queue(AttributedFaithfulness, _attributed(("E1",)), _attributed(("E1",)))
    model.queue(
        CompletenessEvaluation,
        _partial_evaluation(),
        _complete_evaluation().model_copy(update={"requirements": ()}),
    )
    handler = build_orchestrator(
        model=model,
        repository=repository,
        vector_store=hit_store(),
        policy=PolicyConfig(verify_answers=True),
    )

    response = await handler.query(identity, QueryRequest(message="Explain the feature"))

    assert response.status is ResultStatus.VERIFIED_PARTIAL
    assert response.metadata.degraded is True
    assert response.answer.startswith("Advantages [1].")
    assert response.unanswered_requirements == ("limitations",)
    trace = repository.traces[response.request_id]
    assert trace.error_code == "ProviderResponseError"
    assert trace.attempted_strategies == (HAPPY, RECOVERY)
    assert trace.final_source_ids == (CHUNK_ID,)
