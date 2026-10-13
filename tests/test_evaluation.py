from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from models.domain import (
    CompletenessEvaluation,
    CompletenessStatus,
    ContradictedClaim,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    GenerationDraft,
    PackedContext,
    RankedChunk,
    RequirementEvaluation,
    RequirementStatus,
)
from services.contracts import TextModel
from services.evaluation import (
    GenerationService,
    OnlineEvaluator,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tests.fakes import FakeTextModel, StalledTextModel, make_chunk
from utils.prompts import (
    GENERATOR_SYSTEM,
    PROMPT_VERSIONS,
    generation_user_prompt,
    repair_user_prompt,
)

CHUNK_ID = "00000000-0000-0000-0000-000000000001"


def _context() -> PackedContext:
    chunk = make_chunk(CHUNK_ID)
    return PackedContext(
        chunks=(RankedChunk(chunk=chunk, fusion_score=1, fusion_rank=1),),
        token_count=chunk.token_count,
    )


def _supported(*labels: str) -> FaithfulnessEvaluation:
    return FaithfulnessEvaluation(
        passed=True,
        failure_type=FaithfulnessStatus.SUPPORTED,
        verified_sources=labels,
        confidence=0.9,
    )


def _evaluator(model: TextModel) -> OnlineEvaluator:
    return OnlineEvaluator(
        model=model,
        model_name="fake",
        max_output_tokens=256,
        timeout_seconds=1,
    )


async def test_online_evaluator_maps_labels_to_chunk_ids_and_drops_unknown_values() -> None:
    model = FakeTextModel()
    model.queue(
        FaithfulnessEvaluation,
        _supported("E1", "E9", CHUNK_ID),
    )
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            requirements=(
                RequirementEvaluation(
                    requirement="Answer",
                    status=RequirementStatus.ANSWERED,
                    available_in_context=True,
                    supporting_chunks=("E1", "invented"),
                ),
            ),
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(
        question="Question", answer="Answer [E1].", context=_context()
    )

    assert result.faithfulness.verified_sources == (CHUNK_ID,)
    assert result.completeness.requirements[0].supporting_chunks == (CHUNK_ID,)
    assert result.completeness.requirements[0].available_in_context
    assert result.completeness.passed
    assert {call.request_id for call in result.model_calls} == {"fake-request"}


async def test_chunk_identifiers_returned_by_the_model_are_not_trusted() -> None:
    model = FakeTextModel()
    model.queue(FaithfulnessEvaluation, _supported(CHUNK_ID))
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            requirements=(
                RequirementEvaluation(
                    requirement="Answer",
                    status=RequirementStatus.ANSWERED,
                    supporting_chunks=(CHUNK_ID,),
                ),
            ),
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(
        question="Question", answer="Answer", context=_context()
    )

    assert result.faithfulness.passed is False
    assert result.faithfulness.failure_type is FaithfulnessStatus.NO_SUPPORTED_MATERIAL_CLAIMS
    assert result.faithfulness.verified_sources == ()
    assert result.completeness.requirements[0].supporting_chunks == ()


async def test_unsupported_requirement_without_surviving_support_is_not_available() -> None:
    model = FakeTextModel()
    model.queue(FaithfulnessEvaluation, _supported("E1"))
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
                    supporting_chunks=("E9",),
                ),
            ),
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(
        question="Question", answer="Answer", context=_context()
    )

    requirement = result.completeness.requirements[0]
    assert requirement.available_in_context is False
    assert requirement.supporting_chunks == ()
    assert result.completeness.failure_type is CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE


async def test_missing_requirement_supported_by_context_is_missing_answer_content() -> None:
    model = FakeTextModel()
    model.queue(FaithfulnessEvaluation, _supported("E1"))
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=False,
            failure_type=CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE,
            requirements=(
                RequirementEvaluation(
                    requirement="limitations",
                    status=RequirementStatus.MISSING,
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(
        question="Question", answer="Answer", context=_context()
    )

    assert result.completeness.failure_type is CompletenessStatus.MISSING_ANSWER_CONTENT
    assert result.completeness.requirements[0].available_in_context
    assert result.completeness.requirements[0].supporting_chunks == (CHUNK_ID,)


@pytest.mark.parametrize(
    "declared",
    [CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE, CompletenessStatus.MISSING_ANSWER_CONTENT],
)
async def test_failed_completeness_keeps_model_failure_type_when_nothing_is_missing(
    declared: CompletenessStatus,
) -> None:
    model = FakeTextModel()
    model.queue(FaithfulnessEvaluation, _supported("E1"))
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=False,
            failure_type=declared,
            requirements=(
                RequirementEvaluation(
                    requirement="advantages",
                    status=RequirementStatus.ANSWERED,
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(
        question="Question", answer="Answer", context=_context()
    )

    assert result.completeness.failure_type is declared


async def test_model_declared_question_not_answerable_is_kept() -> None:
    model = FakeTextModel()
    model.queue(FaithfulnessEvaluation, _supported("E1"))
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=False,
            failure_type=CompletenessStatus.QUESTION_NOT_ANSWERABLE,
            requirements=(
                RequirementEvaluation(
                    requirement="limitations",
                    status=RequirementStatus.MISSING,
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(
        question="Question", answer="Answer", context=_context()
    )

    assert result.completeness.failure_type is CompletenessStatus.QUESTION_NOT_ANSWERABLE


async def test_contradicted_claim_evidence_and_correcting_labels_map_to_chunk_ids() -> None:
    model = FakeTextModel()
    model.queue(
        FaithfulnessEvaluation,
        FaithfulnessEvaluation(
            passed=False,
            failure_type=FaithfulnessStatus.CONTRADICTED_CLAIMS,
            verified_sources=("E1",),
            contradicted_claims=(ContradictedClaim(claim="A", evidence_chunk_ids=("E1", "E9")),),
            correcting_evidence_available=True,
            correcting_chunks=("E1", "E9"),
            confidence=0.9,
        ),
    )
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(
        question="Question", answer="Answer", context=_context()
    )

    assert result.faithfulness.contradicted_claims[0].evidence_chunk_ids == (CHUNK_ID,)
    assert result.faithfulness.correcting_chunks == (CHUNK_ID,)
    assert result.faithfulness.correcting_evidence_available is True


async def test_generation_and_repair_calls_are_traced_under_their_own_operations() -> None:
    model = FakeTextModel()
    model.queue(
        GenerationDraft,
        GenerationDraft(answer="First"),
        GenerationDraft(answer="Second"),
    )
    service = GenerationService(
        model=model,
        model_name="generation-model",
        max_output_tokens=256,
        timeout_seconds=1,
    )
    context = _context()
    failing = FaithfulnessEvaluation(
        passed=False,
        failure_type=FaithfulnessStatus.UNSUPPORTED_CLAIMS,
        confidence=0.5,
    )
    incomplete = CompletenessEvaluation(
        passed=False,
        failure_type=CompletenessStatus.MISSING_ANSWER_CONTENT,
        confidence=0.5,
    )

    generated = await service.generate(question="Q", context=context, conversation_facts=())
    repaired = await service.repair(
        question="Q",
        previous_answer="First",
        context=context,
        faithfulness=failing,
        completeness=incomplete,
    )

    assert generated.model_call.operation == "generation"
    assert repaired.model_call.operation == "repair"
    assert repaired.draft.answer == "Second"


def test_evidence_payload_carries_labels_and_never_chunk_identifiers() -> None:
    prompt = generation_user_prompt("Question", _context(), ())

    assert json.loads(prompt)["evidence"][0]["evidence_label"] == "E1"
    assert CHUNK_ID not in prompt


def test_repair_feedback_refers_to_evidence_labels_only() -> None:
    prompt = repair_user_prompt(
        question="Question",
        previous_answer="Answer [E1].",
        context=_context(),
        faithfulness=FaithfulnessEvaluation(
            passed=False,
            failure_type=FaithfulnessStatus.UNSUPPORTED_CLAIMS,
            verified_sources=(CHUNK_ID,),
            contradicted_claims=(ContradictedClaim(claim="A", evidence_chunk_ids=(CHUNK_ID,)),),
            confidence=0.5,
        ),
        completeness=CompletenessEvaluation(
            passed=False,
            failure_type=CompletenessStatus.MISSING_ANSWER_CONTENT,
            requirements=(
                RequirementEvaluation(
                    requirement="limitations",
                    status=RequirementStatus.MISSING,
                    available_in_context=True,
                    supporting_chunks=(CHUNK_ID,),
                ),
            ),
            confidence=0.5,
        ),
    )

    feedback = json.loads(prompt)
    assert feedback["faithfulness_feedback"]["verified_sources"] == ["E1"]
    assert feedback["faithfulness_feedback"]["contradicted_claims"][0]["evidence_chunk_ids"] == [
        "E1"
    ]
    assert feedback["completeness_feedback"]["requirements"][0]["supporting_chunks"] == ["E1"]
    assert CHUNK_ID not in prompt


def test_offline_ranking_metrics_have_expected_values() -> None:
    retrieved = ["noise", "relevant", "other"]
    relevant = {"relevant", "missing"}

    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.3868528072)


async def test_generation_is_bounded_by_its_stage_budget() -> None:
    service = GenerationService(
        model=StalledTextModel(),
        model_name="generation-model",
        max_output_tokens=256,
        timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await service.generate(question="Q", context=_context(), conversation_facts=())
    with pytest.raises(TimeoutError):
        await service.repair(
            question="Q",
            previous_answer="A",
            context=_context(),
            faithfulness=FaithfulnessEvaluation(
                passed=False,
                failure_type=FaithfulnessStatus.UNSUPPORTED_CLAIMS,
                confidence=0.5,
            ),
            completeness=CompletenessEvaluation(
                passed=False,
                failure_type=CompletenessStatus.MISSING_ANSWER_CONTENT,
                confidence=0.5,
            ),
        )


async def test_evaluation_judges_are_bounded_by_their_stage_budget() -> None:
    with pytest.raises(TimeoutError):
        await _evaluator(StalledTextModel()).evaluate(
            question="Q",
            answer="A",
            context=_context(),
        )


def test_generation_draft_carries_only_the_answer_the_handler_consumes() -> None:
    assert set(GenerationDraft.model_fields) == {"answer"}
    with pytest.raises(ValidationError):
        GenerationDraft.model_validate({"answer": "A", "proposed_source_ids": ["E1"]})
    assert "proposed_source_ids" not in GENERATOR_SYSTEM


def test_prompt_versions_record_the_revision_of_each_model_stage() -> None:
    stages = ("followup_detector", "generator", "faithfulness", "completeness", "repair")

    assert {stage: PROMPT_VERSIONS[stage] for stage in stages} == {
        "followup_detector": "followup_detector:v1",
        "generator": "generator:v2",
        "faithfulness": "faithfulness:v2",
        "completeness": "completeness:v2",
        "repair": "repair:v2",
    }
