"""Follow-up detectors that estimate whether a message depends on earlier conversation turns."""

from __future__ import annotations

import re
from collections.abc import Sequence

from models.domain import (
    ConversationMessage,
    FollowupClassification,
    FollowupDecision,
    MessageRelevance,
    ModelCallTrace,
)
from services.contracts import TextModel
from utils.prompts import FOLLOWUP_DETECTOR_SYSTEM, followup_detection_user_prompt

_FOLLOWUP_REFERENCE = re.compile(
    r"\b(?:it|its|that|this|they|them|those|these|former|latter|above|previous)\b",
    re.IGNORECASE,
)
_FOLLOWUP_SELECTION = re.compile(
    r"\b(?:first|second|third|last|other|option|instead|one|same|again|more|also)\b",
    re.IGNORECASE,
)
_FOLLOWUP_PHRASE = re.compile(
    r"^(?:what|how|why|when|where)\s+(?:about|else)|^(?:and|but|also)\b",
    re.IGNORECASE,
)
_LEADING_WORD = re.compile(r"[A-Za-z]+")
_SHORT_MESSAGE_WORDS = 6
_IMPERATIVE_VERBS = frozenset(
    {
        "add",
        "bullet",
        "cite",
        "clarify",
        "compare",
        "compress",
        "condense",
        "continue",
        "contrast",
        "correct",
        "define",
        "describe",
        "detail",
        "elaborate",
        "explain",
        "expand",
        "give",
        "go",
        "illustrate",
        "include",
        "list",
        "make",
        "outline",
        "provide",
        "quote",
        "recap",
        "reformat",
        "remove",
        "repeat",
        "rephrase",
        "restate",
        "reword",
        "revise",
        "rewrite",
        "show",
        "shorten",
        "simplify",
        "summarise",
        "summarize",
        "tell",
        "translate",
        "walk",
    }
)


class HeuristicFollowupDetector:
    """Deterministic follow-up cues: references, selections, short questions, and verb fragments."""

    name = "heuristic"

    async def detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> FollowupDecision:
        is_followup = bool(history) and _looks_like_followup(question)
        return FollowupDecision(
            detector=self.name,
            is_followup_probability=1.0 if is_followup else 0.0,
            message_scores=_unit_scores(history),
        )


class AlwaysFollowupDetector:
    """Treat every message as a follow-up that may depend on the whole history."""

    name = "always"

    async def detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> FollowupDecision:
        del question
        return FollowupDecision(
            detector=self.name,
            is_followup_probability=1.0,
            message_scores=_unit_scores(history),
        )


class ModelFollowupDetector:
    """Ask a text model for the follow-up probability and per-message relevance."""

    name = "model"

    def __init__(self, *, model: TextModel, model_name: str, max_output_tokens: int) -> None:
        self._model = model
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens

    async def detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> FollowupDecision:
        history_lines = [f"{message.role}: {message.content}" for message in history]
        result = await self._model.complete(
            model=self._model_name,
            system_prompt=FOLLOWUP_DETECTOR_SYSTEM,
            user_prompt=followup_detection_user_prompt(question, history_lines),
            output_type=FollowupClassification,
            max_output_tokens=self._max_output_tokens,
        )
        return FollowupDecision(
            detector=self.name,
            is_followup_probability=result.value.is_followup_probability,
            message_scores=_relevance_scores(result.value.message_relevance, len(history)),
            model_call=ModelCallTrace(
                operation="followup_detection",
                provider=result.provider,
                model=result.model,
                request_id=result.request_id,
                usage=result.usage,
            ),
            usage=result.usage,
        )


def _looks_like_followup(question: str) -> bool:
    stripped = question.strip()
    words = stripped.split()
    return bool(
        _FOLLOWUP_REFERENCE.search(stripped)
        or _FOLLOWUP_SELECTION.search(stripped)
        or _FOLLOWUP_PHRASE.search(stripped)
        or (len(words) <= _SHORT_MESSAGE_WORDS and stripped.endswith("?"))
        or _starts_with_verb_fragment(stripped)
    )


def _starts_with_verb_fragment(text: str) -> bool:
    words = text.split()
    if not 0 < len(words) <= _SHORT_MESSAGE_WORDS:
        return False
    match = _LEADING_WORD.match(text)
    return match is not None and match.group(0).lower() in _IMPERATIVE_VERBS


def _relevance_scores(relevance: Sequence[MessageRelevance], total: int) -> tuple[float, ...]:
    """Align model relevance with history: missing messages score 0, duplicates keep the max."""

    scores = [0.0] * total
    for item in relevance:
        if item.index < total:
            scores[item.index] = max(scores[item.index], item.probability)
    return tuple(scores)


def _unit_scores(history: Sequence[ConversationMessage]) -> tuple[float, ...]:
    return (1.0,) * len(history)
