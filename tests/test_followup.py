from __future__ import annotations

import json
from typing import TypeVar

import pytest
from pydantic import BaseModel

from models.domain import (
    ConversationMessage,
    FollowupClassification,
    MessageRelevance,
    ModelResult,
)
from services.followup import (
    AlwaysFollowupDetector,
    HeuristicFollowupDetector,
    ModelFollowupDetector,
)
from tests.fakes import FakeTextModel

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


def make_history(*contents: str) -> list[ConversationMessage]:
    return [
        ConversationMessage(role="user" if index % 2 == 0 else "assistant", content=content)
        for index, content in enumerate(contents)
    ]


@pytest.mark.parametrize(
    "question",
    [
        "What about its limitations?",
        "Which one is cheaper?",
        "Compare the first option",
        "Is the latter supported?",
        "Do the same for staging",
        "Can you go over that again",
        "Give me more detail",
        "Also include the rollout dates",
    ],
)
async def test_heuristic_flags_references_and_selections(question: str) -> None:
    decision = await HeuristicFollowupDetector().detect(question, make_history("context"))

    assert decision.is_followup_probability == 1.0


@pytest.mark.parametrize(
    "question",
    [
        "Explain the rollout plan",
        "summarize",
        "EXPLAIN briefly",
        "Rephrase concisely",
        "Restate for a beginner",
        "Recap briefly",
        "Cite your sources",
    ],
)
async def test_heuristic_flags_short_verb_fragments(question: str) -> None:
    decision = await HeuristicFollowupDetector().detect(question, make_history("context"))

    assert decision.is_followup_probability == 1.0


@pytest.mark.parametrize(
    "question",
    [
        "Please explain the second quarter revenue breakdown for the finance team",
        "Summarize the third quarter revenue breakdown for the finance team",
        "Walk me through the last quarter revenue breakdown for the finance team",
        "Compare the finance team plan with the other vendor contract terms",
        "Could the finance team use a cheaper hosting plan instead of renting servers",
        "The finance team reviewed the former vendor contract terms in detail",
    ],
)
async def test_heuristic_flags_selection_words_that_are_the_only_cue(question: str) -> None:
    decision = await HeuristicFollowupDetector().detect(question, make_history("context"))

    assert decision.is_followup_probability == 1.0


@pytest.mark.parametrize(
    "question",
    [
        "How does reciprocal rank fusion combine rankings?",
        "Summarize the quarterly rollout plan for the finance team",
    ],
)
async def test_heuristic_ignores_standalone_questions(question: str) -> None:
    decision = await HeuristicFollowupDetector().detect(question, make_history("context"))

    assert decision.is_followup_probability == 0.0


async def test_heuristic_never_flags_without_history() -> None:
    decision = await HeuristicFollowupDetector().detect("What about it?", [])

    assert decision.is_followup_probability == 0.0
    assert decision.message_scores == ()


async def test_heuristic_scores_every_message_as_relevant() -> None:
    history = make_history("a", "b", "c")

    decision = await HeuristicFollowupDetector().detect("What about it?", history)

    assert decision.message_scores == (1.0, 1.0, 1.0)
    assert decision.detector == "heuristic"
    assert decision.model_call is None


async def test_always_detector_treats_every_message_as_relevant() -> None:
    history = make_history("a", "b")

    decision = await AlwaysFollowupDetector().detect("Standalone question?", history)

    assert decision.detector == "always"
    assert decision.is_followup_probability == 1.0
    assert decision.message_scores == (1.0, 1.0)


async def test_model_detector_aligns_relevance_with_history() -> None:
    model = FakeTextModel()
    model.queue(
        FollowupClassification,
        FollowupClassification(
            is_followup_probability=0.7,
            message_relevance=(
                MessageRelevance(index=0, probability=0.2),
                MessageRelevance(index=2, probability=0.4),
                MessageRelevance(index=2, probability=0.9),
                MessageRelevance(index=7, probability=1.0),
            ),
        ),
    )
    detector = ModelFollowupDetector(model=model, model_name="detector", max_output_tokens=64)

    decision = await detector.detect("Why is that?", make_history("a", "b", "c"))

    assert decision.detector == "model"
    assert decision.is_followup_probability == 0.7
    assert decision.message_scores == (0.2, 0.0, 0.9)
    assert decision.model_call is not None
    assert decision.model_call.operation == "followup_detection"
    assert decision.usage.total_tokens == 15
    assert model.calls == [FollowupClassification]


async def test_model_detector_sends_indexed_history_lines() -> None:
    model = RecordingTextModel()
    model.queue(
        FollowupClassification,
        FollowupClassification(is_followup_probability=0.0),
    )
    detector = ModelFollowupDetector(model=model, model_name="detector", max_output_tokens=64)

    decision = await detector.detect("Why?", make_history("first", "second"))

    payload = json.loads(model.user_prompts[0])
    assert payload == {
        "current_message": "Why?",
        "conversation_history": [
            {"index": 0, "message": "user: first"},
            {"index": 1, "message": "assistant: second"},
        ],
    }
    assert decision.message_scores == (0.0, 0.0)
