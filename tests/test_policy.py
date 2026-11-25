from __future__ import annotations

import pytest

from handlers.policy import DecisionPolicy
from handlers.search_strategies.context_isolation import ContextIsolationStrategy
from handlers.search_strategies.contracts import SearchStrategy, StrategyPlan
from handlers.search_strategies.happy_path import HappyPathStrategy
from handlers.search_strategies.query_recovery import QueryRecoveryStrategy
from handlers.search_strategies.recall_expansion import RecallExpansionStrategy
from handlers.search_strategies.registry import SearchStrategyRegistry, build_strategy_registry
from models.domain import (
    AttemptOutcome,
    AttributedFaithfulness,
    Chunk,
    CompletenessEvaluation,
    CompletenessStatus,
    EvidenceRef,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    Identity,
    ModelUsage,
    QueryUnderstanding,
    RagFailure,
    RagState,
    RequirementEvaluation,
    RequirementStatus,
    RetrievalPlan,
    RetrievalStrategy,
    RetryDecision,
    RetryHypothesis,
)
from services.evaluation import EvaluationResult
from tests.fakes import make_chunk
from utils.settings import PolicyConfig, RetrievalConfig

CANONICAL = "Explain product B migration limitations"
# The rewrite drops "A" and "pricing", two of the literal's three content words.
DRIFTED_LITERAL = "What about product A pricing?"
FACT = "Product A was selected."
HAPPY = RetrievalStrategy.HAPPY_PATH
RECALL = RetrievalStrategy.RECALL_EXPANSION
RECOVERY = RetrievalStrategy.QUERY_RECOVERY
ISOLATION = RetrievalStrategy.CONTEXT_ISOLATION
UNSUPPORTED = FaithfulnessStatus.UNSUPPORTED_CLAIMS
CONTRADICTED = FaithfulnessStatus.CONTRADICTED_CLAIMS
NO_MATERIAL = FaithfulnessStatus.NO_SUPPORTED_MATERIAL_CLAIMS
MISSING_ANSWER = CompletenessStatus.MISSING_ANSWER_CONTENT
MISSING_EVIDENCE = CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE
NOT_ANSWERABLE = CompletenessStatus.QUESTION_NOT_ANSWERABLE
COMPLETE = CompletenessStatus.COMPLETE


def requirement(
    text: str,
    *,
    answered: bool = False,
    available: bool = True,
) -> RequirementEvaluation:
    status = RequirementStatus.ANSWERED if answered else RequirementStatus.MISSING
    return RequirementEvaluation(requirement=text, status=status, available_in_context=available)


MISSING_LIMITATIONS = (requirement("limitations"),)


def evaluation(
    *,
    faithfulness: FaithfulnessStatus = FaithfulnessStatus.SUPPORTED,
    completeness: CompletenessStatus = MISSING_ANSWER,
    requirements: tuple[RequirementEvaluation, ...] = MISSING_LIMITATIONS,
    verified_sources: tuple[str, ...] = (),
) -> EvaluationResult:
    return EvaluationResult(
        faithfulness=FaithfulnessEvaluation(
            passed=faithfulness is FaithfulnessStatus.SUPPORTED,
            failure_type=faithfulness,
            verified_sources=verified_sources,
            confidence=1,
        ),
        completeness=CompletenessEvaluation(
            passed=completeness is CompletenessStatus.COMPLETE,
            failure_type=completeness,
            requirements=requirements,
            confidence=1,
        ),
        usage=ModelUsage(),
    )


def partial_result(*requirements: RequirementEvaluation) -> EvaluationResult:
    return evaluation(requirements=requirements, verified_sources=("chunk-a",))


def make_state(
    *,
    literal: str = CANONICAL,
    facts: tuple[str, ...] = (),
    attempted: tuple[RetrievalStrategy, ...] = (HAPPY,),
) -> RagState:
    return RagState(
        request_id="request",
        identity=Identity(tenant_id="tenant", user_id="user", roles=frozenset({"reader"})),
        query=QueryUnderstanding(
            literal_question=literal,
            canonical_question=CANONICAL,
            rewritten=literal != CANONICAL,
            relevant_conversation_context=facts,
        ),
        attempted_strategies=list(attempted),
    )


def decide(
    state: RagState,
    result: EvaluationResult,
    *,
    max_strategies: int = 4,
    facts_used: bool = False,
    evidence: bool = True,
) -> RetryDecision:
    policy = DecisionPolicy(
        policy=PolicyConfig(max_strategies=max_strategies),
        retrieval=RetrievalConfig(),
    )
    return policy.decide_next(
        state,
        result,
        conversation_facts_used=facts_used,
        evidence_available=evidence,
    )


def preserve(state: RagState, result: EvaluationResult, answer: str) -> None:
    DecisionPolicy.preserve_best(
        state,
        answer=answer,
        evaluation=result,
        strategy=HAPPY,
        repaired=False,
        evidence_refs={},
    )


@pytest.mark.parametrize("faithfulness", [UNSUPPORTED, CONTRADICTED])
def test_unsupported_claims_after_conversation_facts_isolate_context(
    faithfulness: FaithfulnessStatus,
) -> None:
    decision = decide(
        make_state(facts=(FACT,)),
        evaluation(faithfulness=faithfulness),
        facts_used=True,
    )

    assert decision.hypothesis is RetryHypothesis.CONTEXT_CONTAMINATION
    assert decision.next_strategy is ISOLATION


def test_drifted_literal_question_selects_query_recovery() -> None:
    decision = decide(make_state(literal=DRIFTED_LITERAL), evaluation())

    assert decision.hypothesis is RetryHypothesis.QUERY_DRIFT
    assert decision.next_strategy is RECOVERY
    assert decision.drift_score == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        pytest.param(
            "What about product pricing?", RetryHypothesis.QUERY_DRIFT, id="one-of-two-dropped"
        ),
        pytest.param(
            "What about product migration pricing?",
            RetryHypothesis.GENERATION_ONLY,
            id="one-of-three-dropped",
        ),
    ],
)
def test_default_drift_threshold_counts_half_the_words_dropped_as_drift(
    literal: str,
    expected: RetryHypothesis,
) -> None:
    decision = decide(make_state(literal=literal), evaluation())

    assert decision.hypothesis is expected


def test_no_supported_material_with_evidence_is_query_drift() -> None:
    decision = decide(make_state(), evaluation(faithfulness=NO_MATERIAL))

    assert decision.hypothesis is RetryHypothesis.QUERY_DRIFT
    assert decision.next_strategy is RECOVERY


@pytest.mark.parametrize(
    ("faithfulness", "expected"),
    [
        (FaithfulnessStatus.SUPPORTED, RetryHypothesis.GENERATION_ONLY),
        (NO_MATERIAL, RetryHypothesis.QUERY_DRIFT),
    ],
)
def test_zero_threshold_requires_positive_loss_or_independent_evidence_for_drift(
    faithfulness: FaithfulnessStatus, expected: RetryHypothesis
) -> None:
    policy = DecisionPolicy(
        policy=PolicyConfig(), retrieval=RetrievalConfig(query_drift_threshold=0)
    )
    decision = policy.decide_next(
        make_state(),
        evaluation(faithfulness=faithfulness),
        conversation_facts_used=False,
        evidence_available=True,
    )

    assert decision.drift_score == 0
    assert decision.hypothesis is expected


def test_unavailable_requirements_in_multi_part_question_are_query_drift() -> None:
    requirements = (
        requirement("limitations", answered=True),
        requirement("pricing", available=False),
    )
    decision = decide(
        make_state(),
        evaluation(completeness=MISSING_EVIDENCE, requirements=requirements),
    )

    assert decision.hypothesis is RetryHypothesis.QUERY_DRIFT
    assert decision.next_strategy is RECOVERY


def test_question_not_answerable_expands_recall_without_drift() -> None:
    decision = decide(make_state(), evaluation(completeness=NOT_ANSWERABLE))

    assert decision.hypothesis is RetryHypothesis.NOT_ANSWERABLE
    assert decision.next_strategy is RECALL


def test_drifted_not_answerable_question_recovers_the_query_before_giving_up() -> None:
    decision = decide(
        make_state(literal=DRIFTED_LITERAL),
        evaluation(completeness=NOT_ANSWERABLE),
    )

    assert decision.hypothesis is RetryHypothesis.QUERY_DRIFT
    assert decision.next_strategy is RECOVERY


def test_missing_evidence_without_retrieved_chunks_is_low_recall() -> None:
    decision = decide(make_state(), evaluation(faithfulness=NO_MATERIAL), evidence=False)

    assert decision.hypothesis is RetryHypothesis.LOW_RECALL
    assert decision.next_strategy is RECALL


def test_missing_retrieval_evidence_is_low_recall() -> None:
    decision = decide(make_state(), evaluation(completeness=MISSING_EVIDENCE))

    assert decision.hypothesis is RetryHypothesis.LOW_RECALL
    assert decision.next_strategy is RECALL


def test_unavailable_single_requirement_is_low_recall() -> None:
    decision = decide(
        make_state(),
        evaluation(requirements=(requirement("limitations", available=False),)),
    )

    assert decision.hypothesis is RetryHypothesis.LOW_RECALL
    assert decision.next_strategy is RECALL


def test_omitted_answer_content_with_facts_isolates_context() -> None:
    decision = decide(make_state(facts=(FACT,)), evaluation())

    assert decision.hypothesis is RetryHypothesis.GENERATION_ONLY
    assert decision.next_strategy is ISOLATION


def test_unsupported_claims_without_conversation_facts_have_no_retry() -> None:
    decision = decide(make_state(), evaluation(faithfulness=UNSUPPORTED))

    assert decision.hypothesis is RetryHypothesis.GENERATION_ONLY
    assert decision.next_strategy is None


def test_passing_shape_without_missing_content_is_unclassified() -> None:
    decision = decide(
        make_state(),
        evaluation(
            completeness=COMPLETE,
            requirements=(requirement("limitations", answered=True),),
        ),
    )

    assert decision.hypothesis is RetryHypothesis.UNCLASSIFIED
    assert decision.next_strategy is RECALL


def test_contamination_takes_precedence_over_query_drift() -> None:
    decision = decide(
        make_state(literal=DRIFTED_LITERAL, facts=(FACT,)),
        evaluation(faithfulness=UNSUPPORTED),
        facts_used=True,
    )

    assert decision.hypothesis is RetryHypothesis.CONTEXT_CONTAMINATION
    assert decision.next_strategy is ISOLATION


def test_query_drift_falls_back_to_recall_expansion_after_recovery() -> None:
    decision = decide(
        make_state(literal=DRIFTED_LITERAL, attempted=(HAPPY, RECOVERY)),
        evaluation(),
    )

    assert decision.hypothesis is RetryHypothesis.QUERY_DRIFT
    assert decision.next_strategy is RECALL


def test_low_recall_falls_back_to_query_recovery_after_recall_expansion() -> None:
    decision = decide(
        make_state(attempted=(HAPPY, RECALL)),
        evaluation(completeness=MISSING_EVIDENCE),
    )

    assert decision.hypothesis is RetryHypothesis.LOW_RECALL
    assert decision.next_strategy is RECOVERY


def test_not_answerable_never_retries_query_recovery() -> None:
    decision = decide(
        make_state(attempted=(HAPPY, RECALL)), evaluation(completeness=NOT_ANSWERABLE)
    )

    assert decision.hypothesis is RetryHypothesis.NOT_ANSWERABLE
    assert decision.next_strategy is None


def test_generation_only_does_not_repeat_context_isolation() -> None:
    decision = decide(make_state(facts=(FACT,), attempted=(HAPPY, ISOLATION)), evaluation())

    assert decision.hypothesis is RetryHypothesis.GENERATION_ONLY
    assert decision.next_strategy is None


@pytest.mark.parametrize(
    ("attempted", "expected"),
    [
        ((HAPPY,), RECALL),
        ((HAPPY, RECALL), RECOVERY),
        ((HAPPY, RECALL, RECOVERY), ISOLATION),
    ],
)
def test_unclassified_walks_eligible_strategies_in_order(
    attempted: tuple[RetrievalStrategy, ...],
    expected: RetrievalStrategy,
) -> None:
    decision = decide(
        make_state(facts=(FACT,), attempted=attempted),
        evaluation(
            completeness=COMPLETE,
            requirements=(requirement("limitations", answered=True),),
        ),
    )

    assert decision.hypothesis is RetryHypothesis.UNCLASSIFIED
    assert decision.next_strategy is expected


def test_happy_path_is_never_selected_for_retry() -> None:
    results = (
        evaluation(completeness=MISSING_EVIDENCE),
        evaluation(completeness=NOT_ANSWERABLE),
        evaluation(faithfulness=UNSUPPORTED),
        evaluation(faithfulness=NO_MATERIAL),
    )
    for result in results:
        decision = decide(make_state(facts=(FACT,), attempted=()), result, facts_used=True)

        assert decision.next_strategy is not HAPPY


def test_strategy_budget_stops_retries_at_the_cap() -> None:
    decision = decide(
        make_state(attempted=(HAPPY, RECALL)),
        evaluation(completeness=MISSING_EVIDENCE),
        max_strategies=2,
    )

    assert decision.hypothesis is RetryHypothesis.LOW_RECALL
    assert decision.next_strategy is None
    assert decision.reason == "Classified as low recall; no eligible retry remains."


def test_strategy_budget_allows_retry_below_the_cap() -> None:
    decision = decide(
        make_state(attempted=(HAPPY, RECALL)),
        evaluation(completeness=MISSING_EVIDENCE),
        max_strategies=3,
    )

    assert decision.next_strategy is RECOVERY


def test_decision_reason_names_the_selected_strategy() -> None:
    decision = decide(make_state(), evaluation(completeness=MISSING_EVIDENCE))

    assert decision.reason == "Classified as low recall; retrying with RECALL_EXPANSION."


def test_default_registry_declares_each_strategy_plan() -> None:
    config = RetrievalConfig()
    registry = build_strategy_registry(config)

    assert registry.get(HAPPY) == StrategyPlan(
        kind=HAPPY,
        retrieval=RetrievalPlan(
            dense_k=config.happy_dense_k,
            sparse_k=config.happy_sparse_k,
            candidate_k=config.happy_candidate_k,
        ),
    )
    assert registry.get(RECALL) == StrategyPlan(
        kind=RECALL,
        retrieval=RetrievalPlan(
            dense_k=config.recall_dense_k,
            sparse_k=config.recall_sparse_k,
            candidate_k=config.recall_candidate_k,
        ),
    )
    assert registry.get(RECOVERY) == StrategyPlan(
        kind=RECOVERY,
        retrieval=RetrievalPlan(
            dense_k=config.happy_dense_k,
            sparse_k=config.happy_sparse_k,
            candidate_k=config.happy_candidate_k,
            recover_queries=True,
        ),
    )
    assert registry.get(ISOLATION) == StrategyPlan(
        kind=ISOLATION,
        retrieval=RetrievalPlan(dense_k=0, sparse_k=0, candidate_k=0),
        reuse_previous_context=True,
        isolate_conversation=True,
    )


def test_registry_rejects_duplicate_strategy_kind() -> None:
    config = RetrievalConfig()
    strategies: list[SearchStrategy] = [
        HappyPathStrategy(config),
        HappyPathStrategy(config),
        RecallExpansionStrategy(config),
        QueryRecoveryStrategy(config),
        ContextIsolationStrategy(),
    ]

    with pytest.raises(ValueError, match="registered once"):
        SearchStrategyRegistry(strategies)


def test_registry_rejects_missing_strategy_kind() -> None:
    config = RetrievalConfig()
    strategies: list[SearchStrategy] = [
        HappyPathStrategy(config),
        RecallExpansionStrategy(config),
        QueryRecoveryStrategy(config),
    ]

    with pytest.raises(ValueError, match="Every deterministic"):
        SearchStrategyRegistry(strategies)


def test_failures_are_empty_when_the_evaluation_passes() -> None:
    passing = evaluation(completeness=COMPLETE)

    assert DecisionPolicy.failures(passing, decision=None, passed=True) == ()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (evaluation(completeness=MISSING_ANSWER), (RagFailure.GENERATION_OMISSION,)),
        (evaluation(completeness=MISSING_EVIDENCE), (RagFailure.LOW_RECALL,)),
        (evaluation(completeness=NOT_ANSWERABLE), (RagFailure.INSUFFICIENT_CORPUS,)),
        (
            evaluation(completeness=COMPLETE, faithfulness=UNSUPPORTED),
            (RagFailure.GENERATION_HALLUCINATION,),
        ),
        (
            evaluation(completeness=COMPLETE, faithfulness=CONTRADICTED),
            (RagFailure.GENERATION_HALLUCINATION,),
        ),
    ],
)
def test_evaluation_statuses_map_to_rag_failures(
    result: EvaluationResult,
    expected: tuple[RagFailure, ...],
) -> None:
    assert DecisionPolicy.failures(result, decision=None, passed=False) == expected


def test_retry_hypothesis_contributes_its_failure_and_order_is_kept() -> None:
    decision = RetryDecision(
        hypothesis=RetryHypothesis.QUERY_DRIFT,
        next_strategy=RECOVERY,
        drift_score=0.9,
        reason="test",
    )

    failures = DecisionPolicy.failures(
        evaluation(completeness=MISSING_ANSWER, faithfulness=UNSUPPORTED),
        decision=decision,
        passed=False,
    )

    assert failures == (
        RagFailure.GENERATION_OMISSION,
        RagFailure.GENERATION_HALLUCINATION,
        RagFailure.QUERY_DRIFT,
    )


def test_repeated_failure_types_are_deduplicated() -> None:
    decision = RetryDecision(
        hypothesis=RetryHypothesis.LOW_RECALL,
        next_strategy=RECALL,
        drift_score=0,
        reason="test",
    )

    failures = DecisionPolicy.failures(
        evaluation(completeness=MISSING_EVIDENCE, faithfulness=UNSUPPORTED),
        decision=decision,
        passed=False,
    )

    assert failures == (RagFailure.LOW_RECALL, RagFailure.GENERATION_HALLUCINATION)


def test_unmapped_hypothesis_adds_no_failure() -> None:
    decision = RetryDecision(
        hypothesis=RetryHypothesis.NOT_ANSWERABLE,
        next_strategy=RECALL,
        drift_score=0,
        reason="test",
    )

    failures = DecisionPolicy.failures(
        evaluation(completeness=NOT_ANSWERABLE),
        decision=decision,
        passed=False,
    )

    assert failures == (RagFailure.INSUFFICIENT_CORPUS,)


def test_partial_answer_keeps_verified_sources_and_evidence() -> None:
    state = make_state()
    chunk: Chunk = make_chunk("chunk-a")

    DecisionPolicy.preserve_best(
        state,
        answer="Supported advantages.",
        evaluation=partial_result(
            requirement("definition", answered=True),
            requirement("limitations"),
        ),
        strategy=RECALL,
        repaired=True,
        evidence_refs={"E1": EvidenceRef.from_chunk(chunk)},
    )

    best = state.best_verified_answer
    assert best is not None
    assert best.answer == "Supported advantages."
    assert best.verified_chunk_ids == ("chunk-a",)
    assert best.unanswered_requirements == ("limitations",)
    assert best.strategy is RECALL
    assert best.generation_repaired is True
    assert best.evidence_refs == {"E1": EvidenceRef.from_chunk(chunk)}


@pytest.mark.parametrize(
    "result",
    [
        evaluation(
            faithfulness=UNSUPPORTED,
            requirements=(requirement("definition", answered=True),),
            verified_sources=("chunk-a",),
        ),
        evaluation(requirements=(requirement("definition", answered=True),)),
        partial_result(requirement("limitations")),
    ],
)
def test_preserve_rejects_unfaithful_unsourced_or_unanswered_candidates(
    result: EvaluationResult,
) -> None:
    state = make_state()

    preserve(state, result, "Claim")

    assert state.best_verified_answer is None


def test_later_answer_replaces_best_only_with_fewer_unanswered_requirements() -> None:
    state = make_state()
    answered = requirement("definition", answered=True)

    preserve(
        state,
        partial_result(answered, requirement("limitations"), requirement("pricing")),
        "two gaps",
    )
    preserve(state, partial_result(answered, requirement("limitations")), "one gap")
    preserve(
        state, partial_result(answered, requirement("pricing"), requirement("limitations")), "again"
    )

    best = state.best_verified_answer
    assert best is not None
    assert best.answer == "one gap"
    assert best.unanswered_requirements == ("limitations",)


def test_equal_unanswered_count_keeps_the_earlier_answer() -> None:
    state = make_state()
    answered = requirement("definition", answered=True)

    preserve(state, partial_result(answered, requirement("limitations")), "first")
    preserve(state, partial_result(answered, requirement("pricing")), "second")

    best = state.best_verified_answer
    assert best is not None
    assert best.answer == "first"


def test_more_answered_requirements_outrank_fewer_unanswered_gaps() -> None:
    state = make_state()
    definition = requirement("definition", answered=True)
    scope = requirement("scope", answered=True)

    preserve(state, partial_result(definition), "narrow")
    preserve(
        state,
        partial_result(definition, scope, requirement("limitations"), requirement("pricing")),
        "broad",
    )

    best = state.best_verified_answer
    assert best is not None
    assert best.answer == "broad"
    assert best.answered_requirements == 2


def test_fewer_answered_requirements_never_displace_a_better_answer() -> None:
    state = make_state()
    definition = requirement("definition", answered=True)
    scope = requirement("scope", answered=True)

    preserve(
        state,
        partial_result(definition, scope, requirement("limitations"), requirement("pricing")),
        "broad",
    )
    preserve(state, partial_result(definition, requirement("limitations")), "narrow")

    best = state.best_verified_answer
    assert best is not None
    assert best.answer == "broad"


def test_missing_gap_list_discloses_the_literal_question() -> None:
    state = make_state(literal="What about its migration limitations?")

    preserve(state, partial_result(requirement("definition", answered=True)), "answer")

    best = state.best_verified_answer
    assert best is not None
    assert best.unanswered_requirements == ("What about its migration limitations?",)


@pytest.mark.parametrize(
    ("faithfulness", "completeness", "repairable"),
    [
        pytest.param(UNSUPPORTED, COMPLETE, True, id="unsupported-claim"),
        pytest.param(CONTRADICTED, COMPLETE, True, id="contradicted-claim"),
        pytest.param(NO_MATERIAL, COMPLETE, False, id="no-supported-material"),
        pytest.param(UNSUPPORTED, MISSING_EVIDENCE, False, id="missing-retrieval-evidence"),
        pytest.param(UNSUPPORTED, NOT_ANSWERABLE, False, id="hallucinated-unanswerable"),
        pytest.param(CONTRADICTED, NOT_ANSWERABLE, False, id="contradicted-unanswerable"),
    ],
)
def test_claims_to_remove_are_repairable_without_correcting_evidence(
    faithfulness: FaithfulnessStatus,
    completeness: CompletenessStatus,
    repairable: bool,
) -> None:
    policy = DecisionPolicy(policy=PolicyConfig(), retrieval=RetrievalConfig())
    result = evaluation(faithfulness=faithfulness, completeness=completeness, requirements=())

    assert result.faithfulness.correcting_evidence_available is False
    assert policy.can_repair(make_state(), result) is repairable


def test_generation_repair_is_global_and_bounded() -> None:
    policy = DecisionPolicy(policy=PolicyConfig(), retrieval=RetrievalConfig())
    state = make_state()
    result = evaluation()

    assert policy.can_repair(state, result)
    state.generation_repairs_used = 1
    assert not policy.can_repair(state, result)


def _attributed(result: EvaluationResult) -> EvaluationResult:
    faithfulness = AttributedFaithfulness(
        **result.faithfulness.model_dump(),
        occurrence_assessments=(),
        has_uncited_material_claims=False,
    )
    return EvaluationResult(
        faithfulness=faithfulness, completeness=result.completeness, usage=ModelUsage()
    )


@pytest.mark.parametrize(
    "requirements",
    [
        (requirement("definition", answered=True), requirement("  ")),
        (requirement("definition", answered=True, available=False), requirement("limitations")),
        (requirement("definition", answered=True),),
    ],
    ids=["blank-requirement", "answered-without-support", "nothing-missing"],
)
def test_verified_partial_needs_supported_answers_and_a_real_missing_requirement(
    requirements: tuple[RequirementEvaluation, ...],
) -> None:
    state = make_state()
    result = partial_result(*requirements)
    # The legacy verdict is still eligible; the attributed one applies the stricter rule.
    preserve(state, result, "Supported definition [E1].")
    assert state.best_verified_answer is not None

    state = make_state()
    preserve(state, _attributed(result), "Supported definition [E1].")

    assert state.best_verified_answer is None


def test_attributed_partial_with_a_supported_answer_and_a_missing_requirement_is_kept() -> None:
    state = make_state()
    result = _attributed(
        partial_result(requirement("definition", answered=True), requirement("limitations"))
    )

    preserve(state, result, "Supported definition [E1].")

    assert state.best_verified_answer is not None
    assert state.best_verified_answer.unanswered_requirements == ("limitations",)


@pytest.mark.parametrize("outcome", [AttemptOutcome.EMPTY_DRAFT, AttemptOutcome.CITATION_LIMIT])
def test_rejected_draft_follows_the_unclassified_order_and_budget(
    outcome: AttemptOutcome,
) -> None:
    policy = DecisionPolicy(policy=PolicyConfig(), retrieval=RetrievalConfig())

    def next_after(*attempted: RetrievalStrategy, facts: tuple[str, ...] = ()) -> RetryDecision:
        return policy.decide_next(
            make_state(attempted=attempted, facts=facts),
            outcome,
            conversation_facts_used=False,
            evidence_available=True,
        )

    assert next_after(HAPPY).hypothesis is RetryHypothesis.UNCLASSIFIED
    assert next_after(HAPPY).next_strategy is RECALL
    assert next_after(HAPPY, RECALL).next_strategy is RECOVERY
    assert next_after(HAPPY, RECALL, RECOVERY).next_strategy is None
    assert next_after(HAPPY, RECALL, RECOVERY, facts=(FACT,)).next_strategy is ISOLATION
    assert DecisionPolicy.failures(outcome, decision=next_after(HAPPY), passed=False) == ()


@pytest.mark.parametrize("literal", [CANONICAL, DRIFTED_LITERAL], ids=["unchanged", "drifted"])
def test_empty_evidence_outcome_matches_the_synthesized_no_evidence_verdict(literal: str) -> None:
    policy = DecisionPolicy(policy=PolicyConfig(), retrieval=RetrievalConfig())
    legacy = EvaluationResult(
        faithfulness=FaithfulnessEvaluation(passed=False, failure_type=NO_MATERIAL, confidence=1),
        completeness=CompletenessEvaluation(
            passed=False,
            failure_type=MISSING_EVIDENCE,
            requirements=(requirement(literal, available=False),),
            confidence=1,
        ),
        usage=ModelUsage(),
    )
    decisions = [
        policy.decide_next(
            make_state(literal=literal),
            item,
            conversation_facts_used=False,
            evidence_available=False,
        )
        for item in (legacy, AttemptOutcome.EMPTY_EVIDENCE)
    ]

    assert decisions[0] == decisions[1]
    assert DecisionPolicy.failures(
        legacy, decision=decisions[0], passed=False
    ) == DecisionPolicy.failures(AttemptOutcome.EMPTY_EVIDENCE, decision=decisions[1], passed=False)
