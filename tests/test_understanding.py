from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import TypeVar

import pytest
from pydantic import BaseModel

from models.domain import (
    ContextualizationOutput,
    ConversationMessage,
    FollowupClassification,
    FollowupDecision,
    MessageRelevance,
    ModelResult,
)
from services.contracts import FollowupDetector, TextModel
from services.followup import ModelFollowupDetector
from services.understanding import QueryUnderstandingService, query_drift_score
from tests.fakes import FakeFollowupDetector, FakeTextModel
from utils.errors import InfrastructureError, ProviderError
from utils.settings import ContextualizationConfig

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class RecordingTextModel(FakeTextModel):
    """Fake text model that also keeps every user prompt it completed."""

    def __init__(self) -> None:
        super().__init__()
        self.user_prompts: list[str] = []

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]:
        self.user_prompts.append(user_prompt)
        return await super().complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=output_type,
            max_output_tokens=max_output_tokens,
        )


class HangingTextModel:
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
        raise AssertionError("contextualization should time out")


class HangingDetector:
    name = "hanging"

    async def detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> FollowupDecision:
        del question, history
        await asyncio.sleep(60)
        raise AssertionError("detection should time out")


class FailingDetector:
    name = "failing"

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> FollowupDecision:
        del question, history
        raise self._error


def make_history(*contents: str) -> list[ConversationMessage]:
    return [ConversationMessage(role="user", content=content) for content in contents]


def make_service(
    *,
    detector: FollowupDetector,
    model: TextModel,
    config: ContextualizationConfig | None = None,
    detection_timeout_seconds: float = 1.0,
    contextualization_timeout_seconds: float = 1.0,
) -> QueryUnderstandingService:
    return QueryUnderstandingService(
        detector=detector,
        model=model,
        model_name="contextualizer-test",
        max_output_tokens=256,
        config=config if config is not None else ContextualizationConfig(),
        detection_timeout_seconds=detection_timeout_seconds,
        contextualization_timeout_seconds=contextualization_timeout_seconds,
    )


def prompted_history(model: RecordingTextModel) -> list[str]:
    payload = json.loads(model.user_prompts[-1])
    return list(payload["conversation_history"])


@pytest.mark.parametrize(
    ("literal", "canonical", "score"),
    [
        pytest.param(
            "What about the second one?",
            "What is the retention period for administrator audit logs?",
            0,
            id="resolved-selection",
        ),
        pytest.param(
            "And for contractors?",
            "What is the laptop policy for contractors?",
            0,
            id="kept-content-word",
        ),
        pytest.param(
            "How long are traces kept?",
            "How long are RAG request traces kept in MongoDB?",
            0,
            id="expanded-question",
        ),
        pytest.param(
            "What is the refund policy for EU customers?",
            "What is the refund policy?",
            0.5,
            id="dropped-content",
        ),
        pytest.param(
            "What about product A?",
            "Explain product B migration limitations",
            0.5,
            id="replaced-identifier",
        ),
        pytest.param(
            "What changed in v2?",
            "What changed in the latest release?",
            0.5,
            id="replaced-versioned-identifier",
        ),
        pytest.param(
            "What should I do about it?",
            "What should the user do about the expired token?",
            0,
            id="pronoun-i-is-not-an-identifier",
        ),
        pytest.param(
            "What's that one?",
            "What is the retention period for administrator audit logs?",
            0,
            id="contraction-s",
        ),
        pytest.param(
            "Isn\u2019t that one free?",
            "Is the premium plan free?",
            0,
            id="curly-apostrophe-negation",
        ),
        pytest.param(
            "Doesn't it apply to contractors?",
            "Does the laptop policy apply to contractors?",
            0,
            id="negated-contraction",
        ),
        pytest.param(
            "WHAT ABOUT THE SECOND ONE?",
            "What is the retention period for administrator audit logs?",
            0,
            id="all-caps-message",
        ),
        pytest.param(
            "A similar one for contractors?",
            "What is the laptop policy for contractors?",
            0,
            id="opening-article-a",
        ),
        pytest.param(
            "What about the 2nd one?",
            "What is the retention period for audit logs?",
            0,
            id="ordinal",
        ),
        pytest.param(
            "And the next one?",
            "What is the retention period for telemetry events?",
            0,
            id="next-one",
        ),
        pytest.param(
            "What about the follow-up?",
            "What is the follow up process for audits?",
            0,
            id="hyphenated-word-split-in-rewrite",
        ),
        pytest.param(
            "What about the IT department?",
            "What is the laptop policy for the finance department?",
            0.5,
            id="all-caps-name-kept",
        ),
        pytest.param("product A limits", "product A limits", 0, id="identical"),
        pytest.param("", "", 0, id="empty"),
        pytest.param("What is this?", "What is the audit policy?", 0, id="no-content-terms"),
    ],
)
def test_drift_score_is_the_share_of_content_words_the_rewrite_dropped(
    literal: str, canonical: str, score: float
) -> None:
    assert query_drift_score(literal, canonical) == pytest.approx(score)


async def test_disabled_contextualization_never_calls_detector_or_model() -> None:
    detector = FakeFollowupDetector(probability=1.0)
    model = FakeTextModel()
    service = make_service(
        detector=detector,
        model=model,
        config=ContextualizationConfig(enabled=False),
    )

    result = await service.understand("What about it?", make_history("context"))

    assert result.query.canonical_question == "What about it?"
    assert result.query.rewritten is False
    assert result.trace.enabled is False
    assert detector.detect_calls == 0
    assert model.calls == []


async def test_empty_history_returns_literal_question_without_detection() -> None:
    detector = FakeFollowupDetector(probability=1.0)
    model = FakeTextModel()
    service = make_service(detector=detector, model=model)

    result = await service.understand("What about it?", [])

    assert result.query.canonical_question == "What about it?"
    assert result.trace.enabled is True
    assert result.trace.history_messages == 0
    assert result.trace.followup_probability is None
    assert detector.detect_calls == 0
    assert model.calls == []


async def test_probability_below_threshold_keeps_literal_question() -> None:
    detector = FakeFollowupDetector(probability=0.1)
    model = FakeTextModel()
    service = make_service(detector=detector, model=model)

    result = await service.understand("Standalone question?", make_history("context"))

    assert result.query.canonical_question == "Standalone question?"
    assert result.query.rewritten is False
    assert result.trace.followup_probability == 0.1
    assert result.trace.selected_messages == 0
    assert detector.detect_calls == 1
    assert model.calls == []


async def test_probability_at_threshold_triggers_rewrite() -> None:
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="Limits of product A?"),
    )
    service = make_service(detector=FakeFollowupDetector(probability=0.2), model=model)

    result = await service.understand("And its limits?", make_history("product A"))

    assert result.query.canonical_question == "Limits of product A?"
    assert result.trace.rewritten is True
    assert model.calls == [ContextualizationOutput]


async def test_messages_at_threshold_union_recent_tail_in_chronological_order() -> None:
    model = RecordingTextModel()
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question="Q"))
    detector = FakeFollowupDetector(probability=1.0, scores=(0.0, 0.3, 0.29, 0.0))
    config = ContextualizationConfig(min_recent_messages=1, message_threshold=0.3)
    service = make_service(detector=detector, model=model, config=config)

    result = await service.understand("Why?", make_history("m0", "m1", "m2", "m3"))

    assert prompted_history(model) == ["user: m1", "user: m3"]
    assert result.trace.selected_messages == 2
    assert result.trace.history_messages == 4


async def test_recent_floor_is_kept_when_no_message_is_relevant() -> None:
    model = RecordingTextModel()
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question="Q"))
    detector = FakeFollowupDetector(probability=1.0, scores=(0.0, 0.0, 0.0, 0.0, 0.0))
    config = ContextualizationConfig(min_recent_messages=2)
    service = make_service(detector=detector, model=model, config=config)

    await service.understand("Why?", make_history("m0", "m1", "m2", "m3", "m4"))

    assert prompted_history(model) == ["user: m3", "user: m4"]


async def test_nothing_selected_skips_rewrite() -> None:
    model = FakeTextModel()
    detector = FakeFollowupDetector(probability=1.0, scores=(0.0, 0.0))
    config = ContextualizationConfig(min_recent_messages=0)
    service = make_service(detector=detector, model=model, config=config)

    result = await service.understand("Why?", make_history("m0", "m1"))

    assert result.query.canonical_question == "Why?"
    assert result.query.rewritten is False
    assert result.trace.selected_messages == 0
    assert result.trace.followup_probability == 1.0
    assert model.calls == []


async def test_character_cap_drops_oldest_non_recent_messages_first() -> None:
    model = RecordingTextModel()
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question="Q"))
    contents = [f"m{index}" + "x" * 200 for index in range(5)]
    config = ContextualizationConfig(min_recent_messages=1, max_context_characters=500)
    service = make_service(
        detector=FakeFollowupDetector(probability=1.0),
        model=model,
        config=config,
    )

    result = await service.understand("Why?", make_history(*contents))

    assert prompted_history(model) == [f"user: {content}" for content in contents[3:]]
    assert result.trace.selected_messages == 2


async def test_character_cap_falls_back_to_recent_messages_when_still_over() -> None:
    model = RecordingTextModel()
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question="Q"))
    recent = "recent " + "x" * 600
    config = ContextualizationConfig(min_recent_messages=1, max_context_characters=500)
    service = make_service(
        detector=FakeFollowupDetector(probability=1.0),
        model=model,
        config=config,
    )

    result = await service.understand("Why?", make_history("old", "older", recent))

    assert prompted_history(model) == [f"user: {recent}"]
    assert result.trace.selected_messages == 1


async def test_rewrite_returns_canonical_question_and_conversation_context() -> None:
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(
            standalone_question="What are the limits of product A?",
            relevant_conversation_context=("Product A was discussed",),
            entities=("product A",),
            constraints=("limits",),
        ),
    )
    service = make_service(detector=FakeFollowupDetector(probability=1.0), model=model)

    result = await service.understand("And its limits?", make_history("product A", "intro"))

    assert result.query.literal_question == "And its limits?"
    assert result.query.canonical_question == "What are the limits of product A?"
    assert result.query.rewritten is True
    assert result.query.relevant_conversation_context == ("Product A was discussed",)
    assert result.query.entities == ("product A",)
    assert result.query.constraints == ("limits",)
    assert result.trace.rewritten is True
    assert result.usage.total_tokens == 15
    assert [call.operation for call in result.model_calls] == ["contextualization"]


async def test_blank_rewrite_falls_back_to_literal_question() -> None:
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(
            standalone_question="   ",
            relevant_conversation_context=("stale fact",),
        ),
    )
    service = make_service(detector=FakeFollowupDetector(probability=1.0), model=model)

    result = await service.understand("And its limits?", make_history("product A", "intro"))

    assert result.query.canonical_question == "And its limits?"
    assert result.query.rewritten is False
    assert result.query.relevant_conversation_context == ()
    assert result.trace.rewritten is False
    assert result.trace.selected_messages == 2
    assert result.usage.total_tokens == 15


async def test_detection_timeout_fails_open_to_full_history_rewrite() -> None:
    model = RecordingTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="Rewritten question"),
    )
    service = make_service(
        detector=HangingDetector(),
        model=model,
        detection_timeout_seconds=0.01,
    )

    result = await service.understand("And its limits?", make_history("m0", "m1"))

    assert result.trace.fail_open is True
    assert result.trace.followup_probability == 1.0
    assert result.trace.detector == "hanging"
    assert prompted_history(model) == ["user: m0", "user: m1"]
    assert result.query.canonical_question == "Rewritten question"


@pytest.mark.parametrize("error", [ProviderError("upstream down"), InfrastructureError("down")])
async def test_infrastructure_error_fails_open(error: InfrastructureError) -> None:
    model = RecordingTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="Rewritten question"),
    )
    service = make_service(detector=FailingDetector(error), model=model)

    result = await service.understand("And its limits?", make_history("m0", "m1"))

    assert result.trace.fail_open is True
    assert result.trace.followup_probability == 1.0
    assert prompted_history(model) == ["user: m0", "user: m1"]


async def test_unexpected_detection_errors_propagate() -> None:
    service = make_service(
        detector=FailingDetector(RuntimeError("bug")),
        model=FakeTextModel(),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await service.understand("And its limits?", make_history("m0"))


async def test_rewrite_timeout_propagates_as_stage_timeout() -> None:
    service = make_service(
        detector=FakeFollowupDetector(probability=1.0),
        model=HangingTextModel(),
        contextualization_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await service.understand("And its limits?", make_history("m0"))


async def test_detection_and_rewrite_usage_and_calls_are_combined() -> None:
    model = FakeTextModel()
    model.queue(
        FollowupClassification,
        FollowupClassification(
            is_followup_probability=0.9,
            message_relevance=(MessageRelevance(index=0, probability=0.8),),
        ),
    )
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="Standalone question"),
    )
    detector = ModelFollowupDetector(model=model, model_name="detector", max_output_tokens=64)
    service = make_service(detector=detector, model=model)

    result = await service.understand("And its limits?", make_history("m0", "m1"))

    assert [call.operation for call in result.model_calls] == [
        "followup_detection",
        "contextualization",
    ]
    assert result.usage.total_tokens == 30
    assert result.trace.detector == "model"
    assert result.trace.selected_messages == 2
    assert result.trace.fail_open is False
