"""Query understanding: follow-up detection, contextualization, and drift scoring."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from models.domain import (
    ContextualizationOutput,
    ContextualizationTrace,
    ConversationMessage,
    FollowupDecision,
    ModelCallTrace,
    ModelResult,
    ModelUsage,
    QueryUnderstanding,
)
from services.contracts import FollowupDetector, TextModel
from utils.errors import InfrastructureError
from utils.prompts import CONTEXTUALIZER_SYSTEM, contextualization_user_prompt
from utils.settings import ContextualizationConfig
from utils.text import ENGLISH_STOPWORDS

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+(?:[-:]\w+)*")
_PART_SEPARATOR = re.compile(r"[-:_]")
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)")
# Contractions fold back to their head word so "what's" and "isn't" don't leave fragments behind.
_CONTRACTIONS = (
    (re.compile(r"\bcan['\u2019]t\b", re.IGNORECASE), "can"),
    (re.compile(r"\bwon['\u2019]t\b", re.IGNORECASE), "will"),
    (re.compile(r"n['\u2019]t\b", re.IGNORECASE), ""),
    (re.compile(r"['\u2019](?:s|re|ve|ll|d|m)\b", re.IGNORECASE), ""),
)
# Words a follow-up uses to point at earlier turns; a rewrite replaces them, so losing them isn't
# drift.
_REFERENCE_WORDS = frozenset(
    {
        "about",
        "above",
        "another",
        "first",
        "former",
        "it",
        "its",
        "last",
        "latter",
        "next",
        "one",
        "ones",
        "other",
        "previous",
        "same",
        "second",
        "similar",
        "that",
        "them",
        "these",
        "they",
        "third",
        "this",
        "those",
    }
)


@dataclass(frozen=True, slots=True)
class UnderstandingResult:
    query: QueryUnderstanding
    usage: ModelUsage
    trace: ContextualizationTrace
    model_calls: tuple[ModelCallTrace, ...] = ()


def query_drift_score(literal: str, canonical: str) -> float:
    """Return the share of the literal message's content terms the canonical question dropped.

    Resolving a reference ("the second one") loses no content and scores 0; a rewrite that drops
    what the user actually named scores higher. A message without content terms scores 0.
    """

    literal_terms = _content_terms(literal)
    if not literal_terms:
        return 0
    canonical_terms = _all_terms(canonical)
    dropped = [term for term in literal_terms if not _covered(term, canonical_terms)]
    return len(dropped) / len(literal_terms)


def _tokens(text: str) -> list[str]:
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return _WORD.findall(text)


def _content_terms(text: str) -> set[str]:
    # In an all-caps message capitals mark nothing, so no word is kept as an identifier for them.
    shouting = not any(character.islower() for character in text)
    terms: set[str] = set()
    for position, token in enumerate(_tokens(text)):
        term = token.casefold()
        if _ORDINAL.fullmatch(term):
            continue
        if _is_identifier(token, first=position == 0, shouting=shouting) or (
            term not in ENGLISH_STOPWORDS
            and term not in _REFERENCE_WORDS
            and (len(term) > 1 or term.isdigit())
        ):
            terms.add(term)
    return terms


def _all_terms(text: str) -> set[str]:
    """Return every word of the text, plus the parts of hyphenated or joined words."""

    terms: set[str] = set()
    for token in _tokens(text):
        term = token.casefold()
        terms.add(term)
        terms.update(part for part in _PART_SEPARATOR.split(term) if part)
    return terms


def _covered(term: str, canonical_terms: set[str]) -> bool:
    if term in canonical_terms:
        return True
    parts = [part for part in _PART_SEPARATOR.split(term) if part]
    return bool(parts) and all(part in canonical_terms for part in parts)


def _is_identifier(word: str, *, first: bool, shouting: bool) -> bool:
    """Keep names the stopword list would drop: "product A", "EU", "v2".

    A capital letter opening the message ("A similar one?") is an article, not a name.
    """

    if any(character.isdigit() for character in word):
        return True
    if shouting:
        return False
    if len(word) == 1:
        return word.isupper() and word != "I" and not first
    return word.isalpha() and word.isupper()


class QueryUnderstandingService:
    """Contextualize only ambiguous follow-ups and preserve literal and canonical forms."""

    def __init__(
        self,
        *,
        detector: FollowupDetector,
        model: TextModel,
        model_name: str,
        max_output_tokens: int,
        config: ContextualizationConfig,
        detection_timeout_seconds: float,
        contextualization_timeout_seconds: float,
    ) -> None:
        self._detector = detector
        self._model = model
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._config = config
        self._detection_timeout_seconds = detection_timeout_seconds
        self._contextualization_timeout_seconds = contextualization_timeout_seconds

    async def understand(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> UnderstandingResult:
        if not self._config.enabled:
            return _literal(question, ContextualizationTrace(enabled=False), ModelUsage(), ())
        trace = ContextualizationTrace(
            enabled=True,
            detector=self._detector.name,
            history_messages=len(history),
        )
        if not history:
            return _literal(question, trace, ModelUsage(), ())
        detection, fail_open = await self._detect(question, history)
        trace = trace.model_copy(
            update={
                "followup_probability": detection.is_followup_probability,
                "fail_open": fail_open,
            }
        )
        detection_calls = _calls(detection)
        if detection.is_followup_probability < self._config.followup_threshold:
            return _literal(question, trace, detection.usage, detection_calls)
        selected = _select_indices(history, detection.message_scores, self._config)
        trace = trace.model_copy(update={"selected_messages": len(selected)})
        if not selected:
            return _literal(question, trace, detection.usage, detection_calls)
        rewrite = await self._rewrite(question, [history[index] for index in selected])
        return _rewritten(
            question,
            rewrite,
            trace,
            usage=detection.usage + rewrite.usage,
            model_calls=(*detection_calls, _model_call("contextualization", rewrite)),
        )

    async def _detect(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> tuple[FollowupDecision, bool]:
        """Return the detector decision, failing open to a full-history follow-up on errors."""

        try:
            async with asyncio.timeout(self._detection_timeout_seconds):
                return await self._detector.detect(question, history), False
        except (TimeoutError, InfrastructureError) as error:
            logger.warning("Follow-up detection failed open: %s", type(error).__name__)
            fallback = FollowupDecision(
                detector=self._detector.name,
                is_followup_probability=1.0,
                message_scores=(1.0,) * len(history),
            )
            return fallback, True

    async def _rewrite(
        self,
        question: str,
        messages: Sequence[ConversationMessage],
    ) -> ModelResult[ContextualizationOutput]:
        history_lines = [_history_line(message) for message in messages]
        async with asyncio.timeout(self._contextualization_timeout_seconds):
            return await self._model.complete(
                model=self._model_name,
                system_prompt=CONTEXTUALIZER_SYSTEM,
                user_prompt=contextualization_user_prompt(question, history_lines),
                output_type=ContextualizationOutput,
                max_output_tokens=self._max_output_tokens,
            )


def _select_indices(
    history: Sequence[ConversationMessage],
    scores: Sequence[float],
    config: ContextualizationConfig,
) -> tuple[int, ...]:
    """Return the chronological history indices needed to interpret the current message."""

    total = len(history)
    recent = frozenset(range(max(total - config.min_recent_messages, 0), total))
    relevant = {
        index for index, score in enumerate(scores[:total]) if score >= config.message_threshold
    }
    return _within_character_budget(
        history,
        sorted(relevant | recent),
        recent,
        config.max_context_characters,
    )


def _within_character_budget(
    history: Sequence[ConversationMessage],
    indices: Sequence[int],
    recent: frozenset[int],
    limit: int,
) -> tuple[int, ...]:
    """Drop the oldest non-recent messages until the rendered history fits the limit.

    When the limit is still exceeded after every non-recent message is gone, only the recent
    messages remain.
    """

    kept = list(indices)
    for index in indices:
        if _rendered_length(history, kept) <= limit:
            break
        if index not in recent:
            kept.remove(index)
    return tuple(kept)


def _rendered_length(history: Sequence[ConversationMessage], indices: Sequence[int]) -> int:
    return sum(len(_history_line(history[index])) for index in indices)


def _history_line(message: ConversationMessage) -> str:
    return f"{message.role}: {message.content}"


def _calls(decision: FollowupDecision) -> tuple[ModelCallTrace, ...]:
    return () if decision.model_call is None else (decision.model_call,)


def _literal(
    question: str,
    trace: ContextualizationTrace,
    usage: ModelUsage,
    model_calls: tuple[ModelCallTrace, ...],
) -> UnderstandingResult:
    return UnderstandingResult(
        query=QueryUnderstanding(
            literal_question=question,
            canonical_question=question,
            rewritten=False,
        ),
        usage=usage,
        trace=trace,
        model_calls=model_calls,
    )


def _rewritten(
    question: str,
    result: ModelResult[ContextualizationOutput],
    trace: ContextualizationTrace,
    *,
    usage: ModelUsage,
    model_calls: tuple[ModelCallTrace, ...],
) -> UnderstandingResult:
    standalone = result.value.standalone_question
    if not standalone.strip():
        return _literal(question, trace, usage, model_calls)
    rewritten = standalone.strip() != question.strip()
    return UnderstandingResult(
        query=QueryUnderstanding(
            literal_question=question,
            canonical_question=standalone,
            rewritten=rewritten,
            relevant_conversation_context=result.value.relevant_conversation_context,
            entities=result.value.entities,
            constraints=result.value.constraints,
        ),
        usage=usage,
        trace=trace.model_copy(update={"rewritten": rewritten}),
        model_calls=model_calls,
    )


def _model_call[SchemaT: BaseModel](operation: str, result: ModelResult[SchemaT]) -> ModelCallTrace:
    return ModelCallTrace(
        operation=operation,
        provider=result.provider,
        model=result.model,
        request_id=result.request_id,
        usage=result.usage,
    )
