from __future__ import annotations

import asyncio
import json
import threading
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
from services.query_anchors import preserves_explicit_anchors
from services.understanding import QueryUnderstandingService, query_drift_score
from tests.fakes import FakeFollowupDetector, FakeTextModel
from utils.errors import InfrastructureError, ProviderError
from utils.settings import ContextualizationConfig, RetrievalConfig

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
    query_drift_threshold: float = RetrievalConfig().query_drift_threshold,
    detection_timeout_seconds: float = 1.0,
    contextualization_timeout_seconds: float = 1.0,
) -> QueryUnderstandingService:
    return QueryUnderstandingService(
        detector=detector,
        model=model,
        model_name="contextualizer-test",
        max_output_tokens=256,
        config=config if config is not None else ContextualizationConfig(),
        query_drift_threshold=query_drift_threshold,
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
    assert result.trace.proposed_drift_score is None
    assert result.trace.rewrite_decision is None
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


async def test_oversized_newest_message_fits_both_prompts_and_keeps_its_disclosure() -> None:
    model = RecordingTextModel()
    model.queue(FollowupClassification, FollowupClassification(is_followup_probability=1.0))
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question="Q"))
    disclosure = "Unanswered: renewal date."
    content = "Earlier context " + "🦉" * 600 + disclosure
    latest = ConversationMessage(role="assistant", content=content)
    service = make_service(
        detector=ModelFollowupDetector(model=model, model_name="detector", max_output_tokens=64),
        model=model,
        config=ContextualizationConfig(min_recent_messages=2, max_context_characters=500),
    )

    result = await service.understand("Why?", [*make_history("old", "older"), latest])

    entries = json.loads(model.user_prompts[0])["conversation_history"]
    lines = prompted_history(model)
    assert sum(len(entry["message"]) for entry in entries) == 500
    assert sum(map(len, lines)) == 500
    assert entries == [{"index": 0, "message": lines[0]}]
    assert lines == ["assistant: " + content[-489:]]
    assert latest.content == content
    assert result.trace.history_messages == 1
    assert result.trace.selected_messages == 1


async def test_exact_fit_relevance_indexes_refer_to_the_trimmed_window() -> None:
    model = RecordingTextModel()
    model.queue(
        FollowupClassification,
        FollowupClassification(
            is_followup_probability=1.0,
            message_relevance=(MessageRelevance(index=0, probability=1.0),),
        ),
    )
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question="Q"))
    service = make_service(
        detector=ModelFollowupDetector(model=model, model_name="detector", max_output_tokens=64),
        model=model,
        config=ContextualizationConfig(min_recent_messages=0, max_context_characters=500),
    )
    first = "Δ" * 200 + '"\n'
    second = "🦉" * 281
    history = [
        *make_history("omit me", first),
        ConversationMessage(role="assistant", content=second),
    ]

    await service.understand("Why?", history)

    entries = json.loads(model.user_prompts[0])["conversation_history"]
    assert sum(len(entry["message"]) for entry in entries) == 500
    assert entries == [
        {"index": 0, "message": "user: " + first},
        {"index": 1, "message": "assistant: " + second},
    ]
    assert prompted_history(model) == ["user: " + first]


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


@pytest.mark.parametrize(
    ("literal", "proposed", "reason"),
    [
        ("When does retrieval stop?", "Where are traces stored?", "rejected_drift"),
        ("Explain pricing limits", "Explain pricing", "accepted"),
        ("Pricing limits?", "Pricing?", "rejected_drift"),
        (
            "Show the audit retention rules for policy P17 in the archive",
            "Show the audit retention rules for policy P18 in the archive",
            "rejected_anchor",
        ),
        (
            "Where does P17 define archive retention?",
            "Where does the policy define archive retention?",
            "rejected_anchor",
        ),
        ("P17?", "Storage?", "rejected_anchor"),
        (
            "Does the archive charge €50 per month per user?",
            "Does the archive charge €50 per month?",
            "rejected_anchor",
        ),
        (
            "Is the fee £1,200.50/month/user in that archive?",
            "Is the fee £1,200.50 per month per user in the migration archive?",
            "accepted",
        ),
        (
            "Is the archive balance -$50 per month?",
            "Is the archive balance $50/month?",
            "rejected_anchor",
        ),
        (
            "Does the archive charge €50 per month?",
            "Does the archive charge £50 per month?",
            "rejected_anchor",
        ),
        (
            "Explain the Tag7, OwnerID, and Region JSON keys in the archive schema.",
            "Explain the Region, Tag7, and OwnerID JSON keys in the archive schema.",
            "accepted",
        ),
        (
            "What do the AWS JSON keys `Tag7` and `OwnerID` mean in the archive schema?",
            "What do the AWS JSON keys `Tag7` and `ownerid` mean in the migration archive schema?",
            "rejected_anchor",
        ),
        (
            "Explain the archive schema JSON keys:Tag7,OwnerID.",
            "Explain the archive schema JSON keys:Tag7,ownerid.",
            "rejected_anchor",
        ),
        (
            "Explain the Tag7, OwnerID, and Region JSON keys in the archive schema.",
            "Explain the Tag7, ownerid, and Region JSON keys in the archive schema.",
            "rejected_anchor",
        ),
        (
            "Explain JSON keys A and B in the archive schema.",
            "Explain JSON keys A and b in the archive schema.",
            "rejected_anchor",
        ),
        (
            "Compare the date of the 2025 filing with that archive deadline on July 21.",
            "Compare the date of the 2025 filing with the archive deadline on July 21, 2025.",
            "accepted",
        ),
        (
            "Select records with 3 <= retry_count < 10 in that archive.",
            "Select records with retry_count < 10 and retry_count >= 3 in the archive.",
            "accepted",
        ),
        (
            "Show archive records with latency >= 2 days and retention >= 2 hours.",
            "Show archive records with latency >= 2 hours and retention >= 2 days.",
            "rejected_anchor",
        ),
        (
            "Show archive records with 3 >= retry_count and 10 >= batch_size.",
            "Show archive records with retry_count <= 3 and batch_size <= 10.",
            "accepted",
        ),
        (
            "Show archive records with 3 >= retry_count and 10 >= batch_size.",
            "Show archive records with 10 >= retry_count and 3 >= batch_size.",
            "rejected_anchor",
        ),
        (
            "Show archive records with retry_count != 3 and batch_size != 10.",
            "Show archive records with retry_count != 10 and batch_size != 3.",
            "rejected_anchor",
        ),
        (
            "Compare filing deadlines on 21 July 2025 and 22 July 2026.",
            "Compare filing deadlines on July 22nd, 2026 and July 21st, 2025.",
            "accepted",
        ),
        (
            "Compare filing deadlines on 21 July 2025 and 22 July 2026.",
            "Compare filing deadlines on 21 July 2026 and 22 July 2025.",
            "rejected_anchor",
        ),
        (
            "Compare the archive filing deadlines July 21st, 2025 and July 22nd, 2026.",
            "Compare the archive filing deadlines July 21st and July 22nd, 2026 "
            "in the 2025 report.",
            "rejected_anchor",
        ),
        (
            "What about the JSON keys `Tag7`, `OwnerID`, and `Region` in the archive schema?",
            "What about the JSON keys `Tag7`, `ownerId`, and `Region` in the archive schema?",
            "rejected_anchor",
        ),
        (
            "What about the JSON keys Tag7, OwnerID and Region in the archive schema?",
            "What about the JSON keys Tag7, OwnerID and region in the archive schema?",
            "rejected_anchor",
        ),
        (
            "Compare the JSON key `Tag7` with `OwnerID` in the archive schema.",
            "Compare the JSON key `Tag7` with `ownerId` in the archive schema.",
            "rejected_anchor",
        ),
        (
            "Explain the `Tag7` and `OwnerID` JSON keys in the archive schema.",
            "Explain the `tag7` and `OwnerID` JSON keys in the archive schema.",
            "rejected_anchor",
        ),
        (
            "Compare the JSON keys `Owner, ID` and `Tag7` in the archive schema.",
            "Compare the JSON keys `Tag7` and `Owner, ID` in the migration archive schema.",
            "accepted",
        ),
        (
            "Explain the JSON key `Tag7` and describe the migration archive schema.",
            "Describe the migration archive schema and explain the JSON key `Tag7`.",
            "accepted",
        ),
        (
            "What does that schema say about key `Tag7` (case-sensitive)?",
            "What does the archive schema say about key `tag7` (case-sensitive)?",
            "rejected_anchor",
        ),
        (
            "What does that archive say about the JSON key named \"'Tag7'\"?",
            'What does the migration archive say about the JSON key named "Tag7"?',
            "rejected_anchor",
        ),
        (
            "Where is the `Tag7` JSON key documented in that schema?",
            "Where can I find the `Tag7` JSON key in the archive schema?",
            "accepted",
        ),
        (
            "Which JSON key is used for the archive retention date?",
            "Which JSON key holds the archive retention date?",
            "accepted",
        ),
        (
            "What does the JSON key `Tag7` mean for the archive?",
            "What does the JSON key `tag7` mean for the archive named Tag7?",
            "rejected_anchor",
        ),
        (
            "Is this archive limited to 500 B/s during migration?",
            "Is this archive limited to 500 b/s during migration?",
            "rejected_anchor",
        ),
        (
            "Are the upload limits 500 MB/s/user for that archive?",
            "Are the upload limits 500 MB/s for the migration archive?",
            "rejected_anchor",
        ),
        (
            "Show the transfer limit of 500 MB per second per user",
            "Show the archive transfer limit of 500 MB/s/user",
            "accepted",
        ),
        (
            "What does the `Tag7` JSON key mean in that schema?",
            "What does the `tag7` JSON key mean in the archive schema?",
            "rejected_anchor",
        ),
        (
            "What does that schema say about the case-sensitive key 'display name'?",
            "What does the archive schema say about the case-sensitive key 'Display name'?",
            "rejected_anchor",
        ),
        (
            "Is the retention period longer for plan A?",
            "Is the retention period longer for the A plan in Europe?",
            "accepted",
        ),
        (
            "Could you give plan A a longer retention period?",
            "Could you give a plan a longer retention period?",
            "rejected_anchor",
        ),
        (
            "How does policy P17 handle archive retention?",
            "How does P17 handle archive retention in Europe?",
            "accepted",
        ),
        (
            "For that archive, is the filing deadline July 21st?",
            "For the migration archive, is the filing deadline July 22nd?",
            "rejected_anchor",
        ),
        (
            "Is the archive deadline the 21st of July?",
            "Is the archive deadline the 21st of June?",
            "rejected_anchor",
        ),
        (
            "Is the archive deadline July 21?",
            "Is the migration archive deadline the 21st of July?",
            "accepted",
        ),
        (
            "Which retention rules apply to P17 = 30 days and P18 = 90 days?",
            "Which retention rules apply to P17 = 90 days and P18 = 30 days?",
            "rejected_anchor",
        ),
        (
            "Which records have retry_count=3 and batch_size=10?",
            "Which archive records have retry_count=10 and batch_size=3?",
            "rejected_anchor",
        ),
        (
            "Which retention rules apply to P17 = 30 days and P18 = 90 days?",
            "Which archive retention rules apply to P18=90 days and P17=30 days?",
            "accepted",
        ),
        (
            "Show records with A=10 days and B=10 hours",
            "Show archive records with A=10 hours and B=10 days",
            "rejected_anchor",
        ),
        (
            "What are the archive retention rules for policy `ABC`?",
            "What are the archive retention rules for policy `XYZ`?",
            "rejected_anchor",
        ),
        (
            "What about the retention period for policy ABC?",
            "What is the retention period for the ABC policy in the migration archive?",
            "accepted",
        ),
        (
            "What about the retention period for the 'ABC' policy?",
            'Show the archive retention period for policy "abc".',
            "accepted",
        ),
        (
            "Could you give plan A a longer retention period?",
            "Could you give the plan a longer retention period?",
            "rejected_anchor",
        ),
        (
            "Is the transfer limit 500 MB/s for that archive?",
            "Is the transfer limit 500 MB for the migration archive?",
            "rejected_anchor",
        ),
        (
            "Is the transfer limit 500 MB/s for that archive?",
            "Is the transfer limit 500 MB/min for the migration archive?",
            "rejected_anchor",
        ),
        (
            "Is the transfer limit 500 MB/s for that archive?",
            "Is the transfer limit 500 MB per second for the migration archive?",
            "accepted",
        ),
        (
            "Is the transfer limit 500 MB/s for that archive?",
            "Is the transfer limit 500 Mb/s for the migration archive?",
            "rejected_anchor",
        ),
        (
            "What does the case-sensitive JSON key `Tag7` mean in that schema?",
            "What does the case-sensitive JSON key `tag7` mean in the archive schema?",
            "rejected_anchor",
        ),
        (
            "What does the case-sensitive JSON key `Tag7` mean in that schema?",
            "What does the JSON key Tag7 mean in the archive schema?",
            "accepted",
        ),
        (
            "Where does that schema set the case-sensitive identifier RunID?",
            "Where does the archive schema set the identifier runid?",
            "rejected_anchor",
        ),
        (
            "What are the audit limits for policy `P17`?",
            "What are the audit limits for policy `p17` in the archive?",
            "accepted",
        ),
        (
            "Show the audit retention rules for policy ABC in the archive",
            "Show the audit retention rules for policy XYZ in the archive",
            "rejected_anchor",
        ),
        (
            "What about the archive in region EU?",
            "What about the archive in region US?",
            "rejected_anchor",
        ),
        (
            "What about the archive in region EU?",
            "What about the migration archive in region eu?",
            "accepted",
        ),
        ("What about the second one for P17?", "What are audit limits for p17?", "accepted"),
        ("What about the 2nd one?", "What are audit limits?", "accepted"),
        (
            "Is IT subject to audit retention?",
            "Is the service subject to audit retention?",
            "accepted",
        ),
        (
            "Is the archive retention period 1 day?",
            "Is the archive retention period 10 days?",
            "rejected_anchor",
        ),
        (
            "Is the archive retention period 1 day?",
            "Is the archive retention period 1 year?",
            "rejected_anchor",
        ),
        (
            "Is the archive retention period 1,000 days?",
            "Is the archive retention period 1,000 years?",
            "rejected_anchor",
        ),
        (
            "Is the archive retention period 1,000 days?",
            "Is the archive retention period 1000 days in Europe?",
            "accepted",
        ),
        (
            "Does the migration archive hold 500 MB per policy?",
            "Does the migration archive hold 500 GB per policy?",
            "rejected_anchor",
        ),
        (
            "Does the archive cover records with age <= 30 days?",
            "Does the archive cover records with age >= 30 days?",
            "rejected_anchor",
        ),
        (
            "Does the archive support version 1.2?",
            "Does the archive support version 2.1?",
            "rejected_anchor",
        ),
        (
            "Does the archive cover policy AC-17?",
            "Does the archive cover AC and policy 17?",
            "rejected_anchor",
        ),
        (
            "Are there archive retention limits for product A?",
            "Are there archive retention limits for product B on a plan?",
            "rejected_anchor",
        ),
        (
            "Are there archive retention limits for product A?",
            "Are there archive retention limits for product a in Europe?",
            "accepted",
        ),
    ],
)
async def test_rewrite_admission(literal: str, proposed: str, reason: str) -> None:
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(
            standalone_question=proposed,
            entities=("proposed entity",),
            constraints=("proposed constraint",),
            relevant_conversation_context=("proposed fact",),
        ),
    )
    service = make_service(detector=FakeFollowupDetector(), model=model)

    result = await service.understand(literal, make_history("audit limits for P17"))

    accepted = reason == "accepted"
    assert result.query.canonical_question == (proposed if accepted else literal)
    assert result.query.rewritten is accepted
    assert result.trace.rewritten is accepted
    assert result.trace.rewrite_decision == reason
    assert result.trace.proposed_drift_score == query_drift_score(literal, proposed)
    assert result.query.entities == (("proposed entity",) if accepted else ())
    assert result.query.constraints == (("proposed constraint",) if accepted else ())
    assert result.query.relevant_conversation_context == (("proposed fact",) if accepted else ())


async def test_another_request_can_run_during_rewrite_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeTextModel()
    model.queue(
        ContextualizationOutput, ContextualizationOutput(standalone_question="Archive P17?")
    )
    service = make_service(detector=FakeFollowupDetector(probability=1), model=model)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    def wait_for_other_request(literal: str, proposed: str) -> bool:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=2), "rewrite validation blocked the other request"
        return preserves_explicit_anchors(literal, proposed)

    monkeypatch.setattr("services.understanding.preserves_explicit_anchors", wait_for_other_request)
    async with asyncio.TaskGroup() as tasks:
        pending = tasks.create_task(service.understand("P17?", make_history("Archive policies")))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            other = await service.understand("Other question?", [])
            assert other.query.canonical_question == "Other question?"
        finally:
            release.set()
    assert pending.result().query.canonical_question == "Archive P17?"


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
    assert result.trace.rewrite_decision == "empty"
    assert result.trace.proposed_drift_score == 1
    assert result.trace.selected_messages == 2
    assert result.usage.total_tokens == 15


@pytest.mark.parametrize(
    ("proposed", "expected", "reason"),
    [
        (" What about the second one? ", "What about the second one?", "unchanged"),
        ("How long do audit logs last?", "How long do audit logs last?", "accepted"),
    ],
)
async def test_zero_threshold_preserves_lossless_followups(
    proposed: str, expected: str, reason: str
) -> None:
    model = FakeTextModel()
    model.queue(ContextualizationOutput, ContextualizationOutput(standalone_question=proposed))
    service = make_service(detector=FakeFollowupDetector(), model=model, query_drift_threshold=0)

    result = await service.understand(
        "What about the second one?", make_history("traces, audit logs")
    )

    assert result.query.canonical_question == expected
    assert result.trace.proposed_drift_score == 0
    assert result.trace.rewrite_decision == reason


async def test_detection_timeout_fails_open_to_bounded_history_rewrite() -> None:
    model = RecordingTextModel()
    model.queue(
        ContextualizationOutput,
        ContextualizationOutput(standalone_question="What are the audit limits?"),
    )
    service = make_service(
        detector=HangingDetector(),
        model=model,
        config=ContextualizationConfig(min_recent_messages=2, max_context_characters=500),
        detection_timeout_seconds=0.01,
    )

    result = await service.understand("And its limits?", make_history("old", "x" * 600, "new"))

    assert result.trace.fail_open is True
    assert result.trace.followup_probability == 1.0
    assert result.trace.detector == "hanging"
    assert prompted_history(model) == ["user: new"]
    assert result.query.canonical_question == "What are the audit limits?"


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
    assert result.query.canonical_question == "And its limits?"
    assert result.trace.rewrite_decision == "rejected_drift"
    assert result.trace.detector == "model"
    assert result.trace.selected_messages == 2
    assert result.trace.fail_open is False
