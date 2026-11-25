from __future__ import annotations

import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from models.domain import (
    AttributedFaithfulness,
    AttributionStatus,
    CitationAssessment,
    CompletenessEvaluation,
    CompletenessStatus,
    ContradictedClaim,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    GenerationDraft,
    PackedContext,
    QueryUnderstanding,
    RankedChunk,
    RequirementEvaluation,
    RequirementStatus,
    UnsupportedClaim,
)
from services.citations import citation_occurrences
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
from utils.errors import ProviderResponseError
from utils.prompts import (
    ATTRIBUTION_REPAIR_SYSTEM,
    GENERATOR_SYSTEM,
    PROMPT_VERSIONS,
    generation_user_prompt,
    repair_user_prompt,
)

CHUNK_ID = "00000000-0000-0000-0000-000000000001"
QUERY = QueryUnderstanding(
    literal_question="When does it stop?",
    canonical_question="When does retrieval stop?",
    rewritten=True,
)


def _assert_answer_questions(prompt: str) -> None:
    payload = json.loads(prompt)
    assert payload.get("literal_question") == "When does it stop?"
    assert payload.get("accepted_question") == "When does retrieval stop?"
    assert "question" not in payload


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


@pytest.mark.parametrize(
    ("answer", "assessments", "uncited", "passes"),
    [
        ("Both [E1, E2]. Tea [E1].", [(1, ("E1", "E2")), (2, ("E1",))], False, True),
        ("Tea [E2]. Coffee [E1].", [(1, ("E1",)), (2, ("E2",))], False, False),
        ("Both [E1, E2].", [(1, ("E1",))], False, False),
        ("Both [E1, E9].", [(1, ("E1", "E9"))], False, False),
        ("Tea [E1].", [(1, ("E1", "E1"))], False, False),
        ("Tea [E1]. Coffee [E2].", [(1, ("E1",))], False, False),
        ("Tea [E1].", [(1, ("E1",)), (1, ("E1",))], False, False),
        ("Tea [E1].", [(2, ("E1",))], False, False),
        ("Tea [E1]. Coffee.", [(1, ("E1",))], True, False),
        ("Tea.", [], False, False),
        ("Tea [E1].", [(1, ("untrusted model text",))], False, False),
    ],
)
async def test_location_and_label_coverage(
    answer: str, assessments: list[tuple[int, tuple[str, ...]]], uncited: bool, passes: bool
) -> None:
    model = FakeTextModel()
    context = _context()
    second = make_chunk("second")
    context = context.model_copy(
        update={
            "chunks": (*context.chunks, RankedChunk(chunk=second, fusion_score=1, fusion_rank=2))
        }
    )
    model.queue(
        AttributedFaithfulness,
        AttributedFaithfulness(
            passed=True,
            failure_type=FaithfulnessStatus.SUPPORTED,
            verified_sources=("E1", "E2"),
            confidence=1,
            occurrence_assessments=tuple(
                CitationAssessment(
                    occurrence_id=i,
                    supported_labels=labels,
                    attribution_status=AttributionStatus.SUPPORTED,
                )
                for i, labels in assessments
            ),
            has_uncited_material_claims=uncited,
        ),
    )
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            confidence=1,
            requirements=(
                RequirementEvaluation(
                    requirement="Drinks",
                    status=RequirementStatus.ANSWERED,
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
        ),
    )
    rows = citation_occurrences(answer)
    assert rows is not None
    result = await _evaluator(model).evaluate(
        query=QUERY, answer=answer, context=context, occurrences=rows
    )
    assert result.faithfulness.passed is passes
    assert result.faithfulness.verified_sources == ((CHUNK_ID, second.id) if passes else ())
    assert "untrusted model text" not in result.faithfulness.model_dump_json()


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
        query=QUERY, answer="Answer [E1].", context=_context()
    )

    prompts = dict(model.output_prompts)
    _assert_answer_questions(prompts[FaithfulnessEvaluation])
    _assert_answer_questions(prompts[CompletenessEvaluation])
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

    result = await _evaluator(model).evaluate(query=QUERY, answer="Answer", context=_context())

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

    result = await _evaluator(model).evaluate(query=QUERY, answer="Answer", context=_context())

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

    result = await _evaluator(model).evaluate(query=QUERY, answer="Answer", context=_context())

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

    result = await _evaluator(model).evaluate(query=QUERY, answer="Answer", context=_context())

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

    result = await _evaluator(model).evaluate(query=QUERY, answer="Answer", context=_context())

    assert result.completeness.failure_type is CompletenessStatus.QUESTION_NOT_ANSWERABLE


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        pytest.param(True, CompletenessStatus.MISSING_ANSWER_CONTENT, id="left-out-of-answer"),
        pytest.param(False, CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE, id="not-in-evidence"),
    ],
)
async def test_completeness_pass_with_a_missing_requirement_is_a_failure(
    available: bool,
    expected: CompletenessStatus,
) -> None:
    model = FakeTextModel()
    model.queue(FaithfulnessEvaluation, _supported("E1"))
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            requirements=(
                RequirementEvaluation(
                    requirement="limitations",
                    status=RequirementStatus.MISSING,
                    available_in_context=available,
                    supporting_chunks=("E1",) if available else (),
                ),
            ),
            confidence=0.9,
        ),
    )

    result = await _evaluator(model).evaluate(query=QUERY, answer="Answer", context=_context())

    assert result.completeness.passed is False
    assert result.completeness.failure_type is expected


@pytest.mark.parametrize(
    ("unsupported", "contradicted", "expected"),
    [
        pytest.param(("B",), (), FaithfulnessStatus.UNSUPPORTED_CLAIMS, id="unsupported"),
        pytest.param((), ("A",), FaithfulnessStatus.CONTRADICTED_CLAIMS, id="contradicted"),
        pytest.param(("B",), ("A",), FaithfulnessStatus.CONTRADICTED_CLAIMS, id="both"),
    ],
)
async def test_faithfulness_pass_that_lists_failed_claims_is_a_failure(
    unsupported: tuple[str, ...],
    contradicted: tuple[str, ...],
    expected: FaithfulnessStatus,
) -> None:
    model = FakeTextModel()
    model.queue(
        FaithfulnessEvaluation,
        FaithfulnessEvaluation(
            passed=True,
            failure_type=FaithfulnessStatus.SUPPORTED,
            verified_sources=("E1",),
            unsupported_claims=tuple(UnsupportedClaim(claim=claim) for claim in unsupported),
            contradicted_claims=tuple(
                ContradictedClaim(claim=claim, evidence_chunk_ids=("E1",)) for claim in contradicted
            ),
            confidence=0.9,
        ),
    )
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True, failure_type=CompletenessStatus.COMPLETE, confidence=0.9
        ),
    )

    result = await _evaluator(model).evaluate(
        query=QUERY, answer="Answer [E1].", context=_context()
    )

    assert result.faithfulness.passed is False
    assert result.faithfulness.failure_type is expected
    assert result.faithfulness.verified_sources == (CHUNK_ID,)


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

    result = await _evaluator(model).evaluate(query=QUERY, answer="Answer", context=_context())

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

    generated = await service.generate(query=QUERY, context=context, conversation_facts=())
    repaired = await service.repair(
        query=QUERY,
        previous_answer="First",
        context=context,
        faithfulness=failing,
        completeness=incomplete,
    )

    assert generated.model_call.operation == "generation"
    assert repaired.model_call.operation == "repair"
    assert repaired.draft.answer == "Second"
    (_, generation_prompt), (_, repair_prompt) = model.output_prompts
    _assert_answer_questions(generation_prompt)
    _assert_answer_questions(repair_prompt)


def test_evidence_payload_carries_labels_and_never_chunk_identifiers() -> None:
    prompt = generation_user_prompt(QUERY, _context(), ())

    assert json.loads(prompt)["evidence"][0]["evidence_label"] == "E1"
    assert CHUNK_ID not in prompt


def test_repair_feedback_refers_to_evidence_labels_only() -> None:
    prompt = repair_user_prompt(
        query=QUERY,
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


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    [
        pytest.param(["r", "s"], {"r", "s"}, 2, 1.0, id="unique-perfect"),
        pytest.param(
            ["x", "y", "r"],
            {"r"},
            3,
            1 / math.log2(4),
            id="relevant-last",
        ),
        pytest.param(
            ["x", "s", "r"],
            {"r", "s"},
            3,
            (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
            id="reversed",
        ),
        pytest.param(["r", "r"], {"r"}, 2, 1.0, id="repeated-relevant"),
        pytest.param(["x", "r"], {"r"}, 2, 1 / math.log2(3), id="relevant-second"),
        pytest.param(["x", "x", "r"], {"r"}, 3, 1 / math.log2(4), id="repeated-irrelevant"),
        # The copy of r keeps rank 2, so s falls below the cutoff instead of being promoted.
        pytest.param(
            ["r", "r", "s"],
            {"r", "s"},
            2,
            1 / (1 + 1 / math.log2(3)),
            id="second-relevant-below-cutoff",
        ),
        pytest.param(["r"], {"r"}, 0, 1, id="zero-cutoff"),
        pytest.param(["x"], set(), 3, 1, id="nothing-relevant"),
    ],
)
def test_ndcg_counts_each_relevant_id_once_at_its_first_rank(
    retrieved: list[str], relevant: set[str], k: int, expected: float
) -> None:
    score = ndcg_at_k(retrieved, relevant, k)

    assert score == pytest.approx(expected)
    assert 0 <= score <= 1


def test_ndcg_stays_normalized_when_every_result_repeats_a_relevant_id() -> None:
    # The old sum gave 1.63093 for the first case: every copy of r earned a gain.
    assert ndcg_at_k(["r", "r"], {"r"}, 2) == 1.0
    assert ndcg_at_k(["r", "r", "r"], {"r"}, 3) == 1.0
    assert ndcg_at_k(["r", "r", "s"], {"r", "s"}, 2) < 1.0


async def test_generation_is_bounded_by_its_stage_budget() -> None:
    service = GenerationService(
        model=StalledTextModel(),
        model_name="generation-model",
        max_output_tokens=256,
        timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await service.generate(query=QUERY, context=_context(), conversation_facts=())
    with pytest.raises(TimeoutError):
        await service.repair(
            query=QUERY,
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
            query=QUERY,
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
        "generator": "generator:v3",
        "faithfulness": "faithfulness:v3",
        "completeness": "completeness:v3",
        "repair": "repair:v3",
    }


def _attributed_pass() -> AttributedFaithfulness:
    return AttributedFaithfulness(
        passed=True,
        failure_type=FaithfulnessStatus.SUPPORTED,
        verified_sources=("E1",),
        confidence=1,
        occurrence_assessments=(
            CitationAssessment(
                occurrence_id=1,
                supported_labels=("E1",),
                attribution_status=AttributionStatus.SUPPORTED,
            ),
        ),
        has_uncited_material_claims=False,
    )


def _answered(requirement: str, *labels: str) -> RequirementEvaluation:
    return RequirementEvaluation(
        requirement=requirement,
        status=RequirementStatus.ANSWERED,
        available_in_context=True,
        supporting_chunks=labels,
    )


@pytest.mark.parametrize(
    ("status", "requirements", "outcome"),
    [
        pytest.param(
            CompletenessStatus.COMPLETE,
            (_answered("stop", "E1", "E9"),),
            "passes",
            id="valid-label",
        ),
        pytest.param(CompletenessStatus.COMPLETE, (), "malformed", id="no-requirements"),
        pytest.param(
            CompletenessStatus.COMPLETE, (_answered("  ", "E1"),), "malformed", id="blank"
        ),
        pytest.param(
            CompletenessStatus.COMPLETE, (_answered("stop", "E9"),), "malformed", id="no-support"
        ),
        pytest.param(
            CompletenessStatus.COMPLETE,
            (
                _answered("stop", "E1"),
                RequirementEvaluation(requirement="cost", status=RequirementStatus.MISSING),
            ),
            "fails",
            id="complete-with-missing-item",
        ),
        pytest.param(CompletenessStatus.QUESTION_NOT_ANSWERABLE, (), "fails", id="not-answerable"),
    ],
)
async def test_verified_completeness_needs_a_supported_material_requirement(
    status: CompletenessStatus,
    requirements: tuple[RequirementEvaluation, ...],
    outcome: str,
) -> None:
    model = FakeTextModel()
    model.queue(AttributedFaithfulness, _attributed_pass())
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=status is CompletenessStatus.COMPLETE,
            failure_type=status,
            requirements=requirements,
            confidence=1,
        ),
    )
    answer = "It stops at the limit [E1]."
    evaluate = _evaluator(model).evaluate(
        query=QUERY, answer=answer, context=_context(), occurrences=citation_occurrences(answer)
    )

    if outcome == "malformed":
        with pytest.raises(ProviderResponseError):
            await evaluate
        return
    result = await evaluate
    assert result.completeness.passed is (outcome == "passes")
    if outcome == "passes":
        assert result.completeness.requirements[0].supporting_chunks == (CHUNK_ID,)


class _PromptRecorder(FakeTextModel):
    def __init__(self) -> None:
        super().__init__()
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    async def complete(self, **kwargs: Any) -> Any:
        self.system_prompts.append(kwargs["system_prompt"])
        self.user_prompts.append(kwargs["user_prompt"])
        return await super().complete(**kwargs)


async def test_attribution_repair_receives_the_judged_occurrence_table() -> None:
    model = _PromptRecorder()
    model.queue(GenerationDraft, GenerationDraft(answer="Tea [E1]."))
    service = GenerationService(
        model=model, model_name="fake", max_output_tokens=256, timeout_seconds=1
    )
    faithfulness = _attributed_pass().model_copy(
        update={
            "passed": False,
            "failure_type": FaithfulnessStatus.UNSUPPORTED_CLAIMS,
            "verified_sources": (),
        }
    )

    await service.repair(
        query=QUERY,
        previous_answer="Coffee [E2]. Tea [E1].",
        context=_context(),
        faithfulness=faithfulness,
        completeness=CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            requirements=(_answered("tea", CHUNK_ID),),
            confidence=1,
        ),
    )

    assert model.system_prompts == [ATTRIBUTION_REPAIR_SYSTEM]
    payload = json.loads(model.user_prompts[0])
    assert payload["previous_citation_occurrences"] == [
        {"occurrence_id": 1, "start": 7, "end": 11, "claimed_labels": ["E2"]},
        {"occurrence_id": 2, "start": 17, "end": 21, "claimed_labels": ["E1"]},
    ]
    assert payload["faithfulness_feedback"]["occurrence_assessments"][0]["occurrence_id"] == 1


def _two_passage_context() -> tuple[PackedContext, str]:
    context = _context()
    second = make_chunk("second")
    return (
        context.model_copy(
            update={
                "chunks": (
                    *context.chunks,
                    RankedChunk(chunk=second, fusion_score=1, fusion_rank=2),
                )
            }
        ),
        second.id,
    )


@pytest.mark.parametrize(
    ("answer", "status", "labels", "judge_verified", "expected"),
    [
        # The answer-wide verified list can't add a passage no marker cites.
        ("Tea [E1].", "supported", ("E1",), ("E1", "E2"), ("first",)),
        # A label repeated in one group claims one passage.
        ("Tea [E1, E1].", "supported", ("E1",), ("E1",), ("first",)),
        # Supported labels with a contradicted status still fail.
        ("Tea [E1].", "contradicted", ("E1",), ("E1",), ()),
        # An occurrence-level pass the answer-wide verdict doesn't confirm fails.
        ("Tea [E1]. Coffee [E2].", "supported", ("E1",), ("E1",), ()),
    ],
    ids=["uncited-verified", "repeated-label", "contradicted", "unconfirmed"],
)
async def test_verified_sources_come_only_from_accepted_occurrences(
    answer: str,
    status: str,
    labels: tuple[str, ...],
    judge_verified: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    context, second = _two_passage_context()
    occurrences = citation_occurrences(answer)
    assert occurrences is not None
    model = FakeTextModel()
    model.queue(
        AttributedFaithfulness,
        AttributedFaithfulness(
            passed=True,
            failure_type=FaithfulnessStatus.SUPPORTED,
            verified_sources=judge_verified,
            confidence=1,
            occurrence_assessments=tuple(
                CitationAssessment(
                    occurrence_id=row.occurrence_id,
                    supported_labels=row.claimed_labels if len(occurrences) > 1 else labels,
                    attribution_status=AttributionStatus(status),
                )
                for row in occurrences
            ),
            has_uncited_material_claims=False,
        ),
    )
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            confidence=1,
            requirements=(_answered("Drinks", "E1"),),
        ),
    )

    result = await _evaluator(model).evaluate(
        query=QUERY, answer=answer, context=context, occurrences=occurrences
    )

    ids = {"first": CHUNK_ID, "second": second}
    assert result.faithfulness.passed is bool(expected)
    assert result.faithfulness.verified_sources == tuple(ids[item] for item in expected)
