"""OpenAI structured-output adapter: unusable replies become provider response errors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from models.domain import GenerationDraft
from providers.openai import OpenAITextModel
from utils.errors import ProviderResponseError


class _Responses:
    """Answers parse() with a fixed outcome: a reply object, or an exception to raise."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    async def parse(self, **_: Any) -> object:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _model(outcome: object) -> OpenAITextModel:
    client = SimpleNamespace(responses=_Responses(outcome))
    return OpenAITextModel(cast(AsyncOpenAI, client))


def _truncated_parse_error() -> ValidationError:
    """Reproduce the SDK failure for a reply cut off at max_output_tokens."""

    try:
        GenerationDraft.model_validate_json('{"answer": "Acme bills 4')
    except ValidationError as exc:
        return exc
    raise AssertionError("A truncated JSON reply must fail validation")


async def _complete(model: OpenAITextModel) -> object:
    result = await model.complete(
        model="gpt-test",
        system_prompt="system",
        user_prompt="user",
        output_type=GenerationDraft,
        max_output_tokens=64,
    )
    return result.value


async def test_parsed_reply_is_returned_as_the_requested_schema() -> None:
    reply = SimpleNamespace(
        status="completed",
        output_parsed={"answer": "Advantages."},
        model="gpt-test",
        usage=None,
    )

    assert await _complete(_model(reply)) == GenerationDraft(answer="Advantages.")


async def test_truncated_reply_maps_to_provider_response_error_without_model_text() -> None:
    model = _model(_truncated_parse_error())

    with pytest.raises(ProviderResponseError, match="structured output was invalid") as caught:
        await _complete(model)

    assert "Acme" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


async def test_incomplete_reply_reports_its_reason() -> None:
    reply = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_parsed=None,
        model="gpt-test",
        usage=None,
    )

    with pytest.raises(ProviderResponseError, match="max_output_tokens"):
        await _complete(_model(reply))
