"""Grounded generation, parallel online evaluators, repair, and offline metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from models.domain import (
    AttributedFaithfulness,
    AttributionStatus,
    CitationAssessment,
    CitationOccurrence,
    CompletenessEvaluation,
    CompletenessStatus,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    GenerationDraft,
    ModelCallTrace,
    ModelResult,
    ModelUsage,
    PackedContext,
    QueryUnderstanding,
    RequirementEvaluation,
    RequirementStatus,
)
from services.citations import citation_occurrences
from services.contracts import TextModel
from utils.concurrency import run_pair
from utils.errors import ProviderResponseError
from utils.prompts import (
    ATTRIBUTION_REPAIR_SYSTEM,
    ATTRIBUTION_SYSTEM,
    COMPLETENESS_SYSTEM,
    FAITHFULNESS_SYSTEM,
    GENERATOR_SYSTEM,
    REPAIR_SYSTEM,
    REQUIREMENTS_SYSTEM,
    evaluation_user_prompt,
    generation_user_prompt,
    repair_user_prompt,
)
from utils.request_context import observed_call, stage_timeout
from utils.retry import NO_RETRY, retry_transient
from utils.settings import RetryConfig


@dataclass(frozen=True, slots=True)
class GenerationResult:
    draft: GenerationDraft
    usage: ModelUsage
    model: str
    model_call: ModelCallTrace


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    faithfulness: FaithfulnessEvaluation
    completeness: CompletenessEvaluation
    usage: ModelUsage
    model_calls: tuple[ModelCallTrace, ...] = ()


class GenerationService:
    """Generate and repair answers without owning any workflow decisions."""

    def __init__(
        self,
        *,
        model: TextModel,
        model_name: str,
        max_output_tokens: int,
        timeout_seconds: float,
        retry: RetryConfig = NO_RETRY,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._retry = retry

    async def generate(
        self,
        *,
        query: QueryUnderstanding,
        context: PackedContext,
        conversation_facts: Sequence[str],
    ) -> GenerationResult:
        user_prompt = generation_user_prompt(query, context, conversation_facts)
        result = await self._complete("generation", GENERATOR_SYSTEM, user_prompt)
        return _generation(result, "generation")

    async def repair(
        self,
        *,
        query: QueryUnderstanding,
        previous_answer: str,
        context: PackedContext,
        faithfulness: FaithfulnessEvaluation,
        completeness: CompletenessEvaluation,
    ) -> GenerationResult:
        attributed = isinstance(faithfulness, AttributedFaithfulness)
        user_prompt = repair_user_prompt(
            query=query,
            previous_answer=previous_answer,
            context=context,
            faithfulness=faithfulness,
            completeness=completeness,
            # The judged draft fit the table limit, so its occurrence IDs resolve here.
            previous_occurrences=citation_occurrences(previous_answer) if attributed else None,
        )
        system_prompt = ATTRIBUTION_REPAIR_SYSTEM if attributed else REPAIR_SYSTEM
        result = await self._complete("repair", system_prompt, user_prompt)
        return _generation(result, "repair")

    async def _complete(
        self, stage: str, system_prompt: str, user_prompt: str
    ) -> ModelResult[GenerationDraft]:
        async with stage_timeout(self._timeout_seconds):
            return await retry_transient(
                stage,
                lambda retry: observed_call(
                    stage,
                    self._model_name,
                    self._model.complete(
                        model=self._model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        output_type=GenerationDraft,
                        max_output_tokens=self._max_output_tokens,
                    ),
                    retry_ordinal=retry,
                ),
                policy=self._retry,
            )


class OnlineEvaluator:
    """Run faithfulness and completeness concurrently and map evidence labels to chunk IDs."""

    def __init__(
        self,
        *,
        model: TextModel,
        model_name: str,
        max_output_tokens: int,
        timeout_seconds: float,
        retry: RetryConfig = NO_RETRY,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._retry = retry

    async def evaluate(
        self,
        *,
        query: QueryUnderstanding,
        answer: str,
        context: PackedContext,
        occurrences: tuple[CitationOccurrence, ...] | None = None,
    ) -> EvaluationResult:
        prompt = evaluation_user_prompt(query, answer, context)
        faithful_result, complete_result = await run_pair(
            self._judge(
                "faithfulness_evaluation",
                ATTRIBUTION_SYSTEM,
                evaluation_user_prompt(query, answer, context, occurrences),
                AttributedFaithfulness,
            )
            if occurrences is not None
            else self._judge(
                "faithfulness_evaluation", FAITHFULNESS_SYSTEM, prompt, FaithfulnessEvaluation
            ),
            self._judge(
                "completeness_evaluation",
                REQUIREMENTS_SYSTEM if occurrences is not None else COMPLETENESS_SYSTEM,
                prompt,
                CompletenessEvaluation,
            ),
        )
        chunk_ids = {label: chunk.id for label, chunk in context.evidence_by_label().items()}
        faithful = _sanitize_faithfulness(faithful_result.value, chunk_ids)
        if occurrences is not None:
            if not isinstance(faithful, AttributedFaithfulness):
                raise ProviderResponseError("Faithfulness response lacks citation assessments")
            faithful = _verify_attributions(faithful, occurrences, chunk_ids)
        return EvaluationResult(
            faithfulness=faithful,
            completeness=_sanitize_completeness(
                complete_result.value, chunk_ids, strict=occurrences is not None
            ),
            usage=faithful_result.usage + complete_result.usage,
            model_calls=(
                _model_call("faithfulness_evaluation", faithful_result),
                _model_call("completeness_evaluation", complete_result),
            ),
        )

    async def _judge[SchemaT: BaseModel](
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
    ) -> ModelResult[SchemaT]:
        # Each judge retries inside its own run_pair child, so a retry never cancels the other.
        async with stage_timeout(self._timeout_seconds):
            return await retry_transient(
                stage,
                lambda retry: observed_call(
                    stage,
                    self._model_name,
                    self._model.complete(
                        model=self._model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        output_type=output_type,
                        max_output_tokens=self._max_output_tokens,
                    ),
                    retry_ordinal=retry,
                ),
                policy=self._retry,
            )


def _sanitize_faithfulness(
    evaluation: FaithfulnessEvaluation,
    chunk_ids: Mapping[str, str],
) -> FaithfulnessEvaluation:
    verified = _resolve(evaluation.verified_sources, chunk_ids)
    correcting = _resolve(evaluation.correcting_chunks, chunk_ids)
    contradicted = tuple(
        item.model_copy(update={"evidence_chunk_ids": _resolve(item.evidence_chunk_ids, chunk_ids)})
        for item in evaluation.contradicted_claims
    )
    # A judge that passes the answer while listing claims it couldn't support has failed it.
    failure = evaluation.failure_type
    if evaluation.passed and contradicted:
        failure = FaithfulnessStatus.CONTRADICTED_CLAIMS
    elif evaluation.passed and evaluation.unsupported_claims:
        failure = FaithfulnessStatus.UNSUPPORTED_CLAIMS
    passed = failure is FaithfulnessStatus.SUPPORTED
    if passed and not verified:
        return evaluation.model_copy(
            update={
                "passed": False,
                "failure_type": FaithfulnessStatus.NO_SUPPORTED_MATERIAL_CLAIMS,
                "verified_sources": (),
                "correcting_evidence_available": False,
                "correcting_chunks": (),
                "contradicted_claims": contradicted,
            }
        )
    return evaluation.model_copy(
        update={
            "passed": passed,
            "failure_type": failure,
            "verified_sources": verified,
            "correcting_chunks": correcting,
            "correcting_evidence_available": bool(correcting)
            and evaluation.correcting_evidence_available,
            "contradicted_claims": contradicted,
        }
    )


def _verify_attributions(
    verdict: AttributedFaithfulness,
    occurrences: tuple[CitationOccurrence, ...],
    chunk_ids: Mapping[str, str],
) -> AttributedFaithfulness:
    expected = {row.occurrence_id: row.claimed_labels for row in occurrences}
    seen: set[int] = set()
    valid = bool(expected) and not verdict.has_uncited_material_claims
    safe_rows: list[CitationAssessment] = []
    for assessment in verdict.occurrence_assessments:
        ordinal = assessment.occurrence_id
        if ordinal not in expected or ordinal in seen:
            valid = False
            continue
        seen.add(ordinal)
        # A label repeated inside one group, as in [E1, E1], claims that passage once.
        claimed = tuple(dict.fromkeys(expected[ordinal]))
        supported = assessment.supported_labels
        # Only packed labels actually present at this location can enter a trace or repair prompt.
        safe = tuple(
            dict.fromkeys(label for label in supported if label in chunk_ids and label in claimed)
        )
        safe_rows.append(assessment.model_copy(update={"supported_labels": safe}))
        if (
            assessment.attribution_status is not AttributionStatus.SUPPORTED
            or len(safe) != len(supported)
            or len(safe) != len(claimed)
            or set(safe) != set(claimed)
        ):
            valid = False
    cited = _resolve((label for row in occurrences for label in row.claimed_labels), chunk_ids)
    valid = valid and seen == expected.keys() and set(cited) <= set(verdict.verified_sources)
    failure = verdict.failure_type
    if verdict.passed and not valid:
        failure = FaithfulnessStatus.UNSUPPORTED_CLAIMS
    passed = failure is FaithfulnessStatus.SUPPORTED
    return verdict.model_copy(
        update={
            "passed": passed,
            "failure_type": failure,
            "verified_sources": cited if passed else (),
            "occurrence_assessments": tuple(safe_rows),
        }
    )


def _sanitize_completeness(
    evaluation: CompletenessEvaluation,
    chunk_ids: Mapping[str, str],
    *,
    strict: bool = False,
) -> CompletenessEvaluation:
    requirements = tuple(_resolve_requirement(item, chunk_ids) for item in evaluation.requirements)
    if (
        strict
        and evaluation.passed
        and (
            not requirements
            or any(
                not item.requirement.strip()
                or (item.status is RequirementStatus.ANSWERED and not item.available_in_context)
                for item in requirements
            )
        )
    ):
        raise ProviderResponseError(
            "Complete verdict lacks a material requirement or valid supporting evidence"
        )
    if evaluation.passed and not chunk_ids:
        return evaluation.model_copy(
            update={
                "passed": False,
                "failure_type": CompletenessStatus.QUESTION_NOT_ANSWERABLE,
                "requirements": requirements,
                "missing_information_available": False,
            }
        )
    failure = evaluation.failure_type
    # The missing requirements decide the failure, so a judge that passes the answer while
    # marking a requirement missing has failed it.
    if failure is not CompletenessStatus.QUESTION_NOT_ANSWERABLE:
        failure = _missing_failure(requirements, failure)
    return evaluation.model_copy(
        update={
            "passed": failure is CompletenessStatus.COMPLETE,
            "failure_type": failure,
            "requirements": requirements,
        }
    )


def _resolve_requirement(
    requirement: RequirementEvaluation,
    chunk_ids: Mapping[str, str],
) -> RequirementEvaluation:
    # Only labels that resolve to the packed evidence count as support.
    supporting = _resolve(requirement.supporting_chunks, chunk_ids)
    return requirement.model_copy(
        update={
            "supporting_chunks": supporting,
            "available_in_context": requirement.available_in_context and bool(supporting),
        }
    )


def _missing_failure(
    requirements: Sequence[RequirementEvaluation],
    fallback: CompletenessStatus,
) -> CompletenessStatus:
    """Classify a failed completeness check from its missing requirements alone."""

    missing = [item for item in requirements if item.status is RequirementStatus.MISSING]
    if any(not item.available_in_context for item in missing):
        return CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE
    if missing:
        return CompletenessStatus.MISSING_ANSWER_CONTENT
    return fallback


def _resolve(labels: Iterable[str], chunk_ids: Mapping[str, str]) -> tuple[str, ...]:
    """Map evidence labels to unique chunk IDs and drop labels outside the packed context."""

    return tuple(dict.fromkeys(chunk_ids[label] for label in labels if label in chunk_ids))


def _generation(result: ModelResult[GenerationDraft], operation: str) -> GenerationResult:
    return GenerationResult(
        draft=result.value,
        usage=result.usage,
        model=result.model,
        model_call=_model_call(operation, result),
    )


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return the fraction of relevant IDs present in the first k results."""

    if not relevant:
        return 1
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return the relevant fraction of the first k result positions."""

    selected = retrieved[:k]
    if not selected:
        return 0
    return len(set(selected) & relevant) / len(selected)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Return the reciprocal position of the first relevant result."""

    return next((1 / rank for rank, item in enumerate(retrieved, start=1) if item in relevant), 0)


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return binary normalized discounted cumulative gain at k, for a nonnegative k.

    A relevant ID earns its gain at its first position only. A later copy still occupies its rank
    and earns nothing, so a repeated result cannot push the score above 1.
    """

    credited: set[str] = set()
    actual = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant and item not in credited:
            credited.add(item)
            actual += 1 / math.log2(rank + 1)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return actual / ideal if ideal else 1


def _model_call[SchemaT: BaseModel](operation: str, result: ModelResult[SchemaT]) -> ModelCallTrace:
    return ModelCallTrace(
        operation=operation,
        provider=result.provider,
        model=result.model,
        request_id=result.request_id,
        usage=result.usage,
    )
