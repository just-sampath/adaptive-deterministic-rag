"""Deterministic branching policy for bounded RAG attempts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from models.domain import (
    CompletenessStatus,
    EvidenceRef,
    FaithfulnessStatus,
    RagFailure,
    RagState,
    RequirementEvaluation,
    RequirementStatus,
    ResultStatus,
    RetrievalStrategy,
    RetryDecision,
    RetryHypothesis,
    VerifiedAnswer,
)
from services.evaluation import EvaluationResult
from services.understanding import drift_reaches_threshold, query_drift_score
from utils.settings import PolicyConfig, RetrievalConfig

_UNSUPPORTED = frozenset(
    {FaithfulnessStatus.UNSUPPORTED_CLAIMS, FaithfulnessStatus.CONTRADICTED_CLAIMS}
)
_PREFERENCES: Mapping[RetryHypothesis, tuple[RetrievalStrategy, ...]] = {
    RetryHypothesis.CONTEXT_CONTAMINATION: (RetrievalStrategy.CONTEXT_ISOLATION,),
    RetryHypothesis.QUERY_DRIFT: (
        RetrievalStrategy.QUERY_RECOVERY,
        RetrievalStrategy.RECALL_EXPANSION,
    ),
    RetryHypothesis.NOT_ANSWERABLE: (RetrievalStrategy.RECALL_EXPANSION,),
    RetryHypothesis.LOW_RECALL: (
        RetrievalStrategy.RECALL_EXPANSION,
        RetrievalStrategy.QUERY_RECOVERY,
    ),
    RetryHypothesis.GENERATION_ONLY: (RetrievalStrategy.CONTEXT_ISOLATION,),
    RetryHypothesis.UNCLASSIFIED: (
        RetrievalStrategy.RECALL_EXPANSION,
        RetrievalStrategy.QUERY_RECOVERY,
        RetrievalStrategy.CONTEXT_ISOLATION,
    ),
}
_COMPLETENESS_FAILURES: Mapping[CompletenessStatus, RagFailure] = {
    CompletenessStatus.MISSING_ANSWER_CONTENT: RagFailure.GENERATION_OMISSION,
    CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE: RagFailure.LOW_RECALL,
    CompletenessStatus.QUESTION_NOT_ANSWERABLE: RagFailure.INSUFFICIENT_CORPUS,
}
_HYPOTHESIS_FAILURES: Mapping[RetryHypothesis, RagFailure] = {
    RetryHypothesis.CONTEXT_CONTAMINATION: RagFailure.CONTEXT_CONTAMINATION,
    RetryHypothesis.QUERY_DRIFT: RagFailure.QUERY_DRIFT,
    RetryHypothesis.LOW_RECALL: RagFailure.LOW_RECALL,
}


@dataclass(frozen=True, slots=True)
class TerminalResult:
    status: ResultStatus
    answer: str
    verified_chunk_ids: tuple[str, ...]
    unanswered_requirements: tuple[str, ...]
    strategy: RetrievalStrategy | None
    generation_repaired: bool
    evidence_refs: dict[str, EvidenceRef] = field(default_factory=dict)
    degraded: bool = False


class DecisionPolicy:
    """Own every workflow branch; model outputs remain diagnostic data."""

    def __init__(self, *, policy: PolicyConfig, retrieval: RetrievalConfig) -> None:
        self._policy = policy
        self._retrieval = retrieval

    @staticmethod
    def passed(evaluation: EvaluationResult) -> bool:
        return evaluation.faithfulness.passed and evaluation.completeness.passed

    def can_repair(self, state: RagState, evaluation: EvaluationResult) -> bool:
        """Repair answers the evidence can fix: omitted content, or claims to remove or correct.

        Removing unsupported claims only helps an answer that is otherwise complete; when the
        question isn't answerable from the evidence, a later strategy needs the single repair more.
        """

        faithfulness = evaluation.faithfulness
        completeness = evaluation.completeness
        return (
            state.generation_repairs_used < self._policy.max_generation_repairs
            and (
                completeness.failure_type is CompletenessStatus.MISSING_ANSWER_CONTENT
                or faithfulness.correcting_evidence_available
                or (
                    faithfulness.failure_type in _UNSUPPORTED
                    and completeness.failure_type is CompletenessStatus.COMPLETE
                )
            )
            and completeness.failure_type is not CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE
            and faithfulness.failure_type is not FaithfulnessStatus.NO_SUPPORTED_MATERIAL_CLAIMS
        )

    def decide_next(
        self,
        state: RagState,
        evaluation: EvaluationResult,
        *,
        conversation_facts_used: bool,
        evidence_available: bool,
    ) -> RetryDecision:
        drift = query_drift_score(state.query.literal_question, state.query.canonical_question)
        hypothesis = _classify(
            evaluation,
            drift_high=drift_reaches_threshold(drift, self._retrieval.query_drift_threshold),
            conversation_facts_used=conversation_facts_used,
            evidence_available=evidence_available,
        )
        strategy = self._next_strategy(state, hypothesis)
        return RetryDecision(
            hypothesis=hypothesis,
            next_strategy=strategy,
            drift_score=drift,
            reason=_reason(hypothesis, strategy),
        )

    def _next_strategy(
        self,
        state: RagState,
        hypothesis: RetryHypothesis,
    ) -> RetrievalStrategy | None:
        if len(state.attempted_strategies) >= self._policy.max_strategies:
            return None
        attempted = set(state.attempted_strategies)
        facts_present = bool(state.query.relevant_conversation_context)
        for strategy in _PREFERENCES[hypothesis]:
            if strategy in attempted:
                continue
            if strategy is RetrievalStrategy.CONTEXT_ISOLATION and not facts_present:
                continue
            return strategy
        return None

    @staticmethod
    def failures(
        evaluation: EvaluationResult,
        *,
        decision: RetryDecision | None,
        passed: bool,
    ) -> tuple[RagFailure, ...]:
        if passed:
            return ()
        unsupported = evaluation.faithfulness.failure_type in _UNSUPPORTED
        candidates = (
            _COMPLETENESS_FAILURES.get(evaluation.completeness.failure_type),
            RagFailure.GENERATION_HALLUCINATION if unsupported else None,
            _decision_failure(decision),
        )
        return tuple(dict.fromkeys(item for item in candidates if item is not None))

    @staticmethod
    def preserve_best(
        state: RagState,
        *,
        answer: str,
        evaluation: EvaluationResult,
        strategy: RetrievalStrategy,
        repaired: bool,
        evidence_refs: dict[str, EvidenceRef],
    ) -> None:
        faithfulness = evaluation.faithfulness
        requirements = evaluation.completeness.requirements
        if not faithfulness.passed or not faithfulness.verified_sources:
            return
        answered = sum(item.status is RequirementStatus.ANSWERED for item in requirements)
        if answered == 0:
            return
        candidate = VerifiedAnswer(
            answer=answer,
            verified_chunk_ids=faithfulness.verified_sources,
            unanswered_requirements=_unanswered(state, requirements),
            strategy=strategy,
            generation_repaired=repaired,
            answered_requirements=answered,
            evidence_refs=evidence_refs,
        )
        best = state.best_verified_answer
        if best is None or _outranks(candidate, best):
            state.best_verified_answer = candidate


def _outranks(candidate: VerifiedAnswer, best: VerifiedAnswer) -> bool:
    """More answered requirements wins; on a tie, fewer unanswered requirements wins."""

    if candidate.answered_requirements != best.answered_requirements:
        return candidate.answered_requirements > best.answered_requirements
    return len(candidate.unanswered_requirements) < len(best.unanswered_requirements)


def _classify(
    evaluation: EvaluationResult,
    *,
    drift_high: bool,
    conversation_facts_used: bool,
    evidence_available: bool,
) -> RetryHypothesis:
    faithfulness = evaluation.faithfulness.failure_type
    completeness = evaluation.completeness
    unsupported = faithfulness in _UNSUPPORTED
    unavailable_missing = _has_unavailable_missing(completeness.requirements)
    no_material = faithfulness is FaithfulnessStatus.NO_SUPPORTED_MATERIAL_CLAIMS
    if conversation_facts_used and unsupported:
        return RetryHypothesis.CONTEXT_CONTAMINATION
    if (
        drift_high
        or (evidence_available and no_material)
        or (len(completeness.requirements) >= 2 and unavailable_missing)
    ):
        return RetryHypothesis.QUERY_DRIFT
    if completeness.failure_type is CompletenessStatus.QUESTION_NOT_ANSWERABLE:
        return RetryHypothesis.NOT_ANSWERABLE
    if (
        not evidence_available
        or completeness.failure_type is CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE
        or unavailable_missing
    ):
        return RetryHypothesis.LOW_RECALL
    if unsupported or completeness.failure_type is CompletenessStatus.MISSING_ANSWER_CONTENT:
        return RetryHypothesis.GENERATION_ONLY
    return RetryHypothesis.UNCLASSIFIED


def _has_unavailable_missing(requirements: tuple[RequirementEvaluation, ...]) -> bool:
    return any(
        item.status is RequirementStatus.MISSING and not item.available_in_context
        for item in requirements
    )


def _reason(hypothesis: RetryHypothesis, strategy: RetrievalStrategy | None) -> str:
    label = hypothesis.value.lower().replace("_", " ")
    if strategy is None:
        return f"Classified as {label}; no eligible retry remains."
    return f"Classified as {label}; retrying with {strategy.value}."


def _decision_failure(decision: RetryDecision | None) -> RagFailure | None:
    if decision is None:
        return None
    return _HYPOTHESIS_FAILURES.get(decision.hypothesis)


def _unanswered(
    state: RagState,
    requirements: tuple[RequirementEvaluation, ...],
) -> tuple[str, ...]:
    missing = tuple(
        item.requirement for item in requirements if item.status is RequirementStatus.MISSING
    )
    return missing or (state.query.literal_question,)
