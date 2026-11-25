"""OpenAI structured-output adapter: strict request schemas, and unusable replies as errors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, Omit
from pydantic import BaseModel

from models.domain import (
    AttributedFaithfulness,
    CompletenessEvaluation,
    ContextualizationOutput,
    FaithfulnessEvaluation,
    FollowupClassification,
    GenerationDraft,
    QueryRecoveryOutput,
)
from providers.openai import OpenAITextModel, structured_output_format
from utils.errors import ProviderResponseError

# Every schema the application asks OpenAI to fill.
_OUTPUT_TYPES: tuple[type[BaseModel], ...] = (
    GenerationDraft,
    FaithfulnessEvaluation,
    AttributedFaithfulness,
    CompletenessEvaluation,
    ContextualizationOutput,
    QueryRecoveryOutput,
    FollowupClassification,
)


class _Responses:
    """Answers create() with a fixed outcome and records the request it received."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome
        self.request: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> object:
        self.request = kwargs
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _model(
    outcome: object, *, reasoning_effort: dict[str, str] | None = None
) -> tuple[OpenAITextModel, _Responses]:
    responses = _Responses(outcome)
    client = SimpleNamespace(responses=responses)
    return OpenAITextModel(cast(AsyncOpenAI, client), reasoning_effort=reasoning_effort), responses


def _reply(output_text: str, *, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        incomplete_details=None,
        output_text=output_text,
        model="gpt-test",
        usage=None,
    )


async def _complete(model: OpenAITextModel, *, name: str = "gpt-test") -> object:
    result = await model.complete(
        model=name,
        system_prompt="system",
        user_prompt="user",
        output_type=GenerationDraft,
        max_output_tokens=64,
    )
    return result.value


def _keys(node: Any) -> set[str]:
    """Collect every dictionary key in a JSON schema, at any depth."""

    if isinstance(node, list):
        return set().union(*(_keys(item) for item in node))
    if isinstance(node, dict):
        return set(node).union(*(_keys(value) for value in node.values()))
    return set()


def _objects(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        return [found for item in node for found in _objects(item)]
    if not isinstance(node, dict):
        return []
    nested = [found for value in node.values() for found in _objects(value)]
    return [node, *nested] if node.get("type") == "object" else nested


async def test_reply_text_is_validated_as_the_requested_schema() -> None:
    model, _ = _model(_reply('{"answer": "Advantages."}'))

    assert await _complete(model) == GenerationDraft(answer="Advantages.")


async def test_fresh_attribution_response_requires_assessments_even_for_legacy_shaped_json() -> (
    None
):
    historical = (
        '{"passed":true,"failure_type":"SUPPORTED","confidence":1,"verified_sources":["E1"]}'
    )
    assert FaithfulnessEvaluation.model_validate_json(historical).passed
    model, _ = _model(_reply(historical))
    with pytest.raises(ProviderResponseError):
        await model.complete(
            model="test",
            system_prompt="judge",
            user_prompt="draft",
            output_type=AttributedFaithfulness,
            max_output_tokens=128,
        )


@pytest.mark.parametrize(
    ("output_type", "reply", "passed"),
    [
        pytest.param(
            FaithfulnessEvaluation,
            '{"passed": true, "failure_type": "UNSUPPORTED_CLAIMS", "confidence": 0.9}',
            False,
            id="faithfulness-flag-passes-a-failure",
        ),
        pytest.param(
            FaithfulnessEvaluation,
            '{"passed": false, "failure_type": "SUPPORTED", "confidence": 0.9}',
            True,
            id="faithfulness-flag-fails-a-pass",
        ),
        pytest.param(
            CompletenessEvaluation,
            '{"passed": true, "failure_type": "MISSING_ANSWER_CONTENT", "confidence": 0.9}',
            False,
            id="completeness-flag-passes-a-failure",
        ),
        pytest.param(
            CompletenessEvaluation,
            '{"passed": false, "failure_type": "COMPLETE", "confidence": 0.9}',
            True,
            id="completeness-flag-fails-a-pass",
        ),
    ],
)
async def test_judge_flag_that_disagrees_with_its_status_follows_the_status(
    output_type: type[FaithfulnessEvaluation | CompletenessEvaluation],
    reply: str,
    passed: bool,
) -> None:
    model, _ = _model(_reply(reply))

    result = await model.complete(
        model="gpt-test",
        system_prompt="system",
        user_prompt="user",
        output_type=output_type,
        max_output_tokens=64,
    )

    assert result.value.passed is passed


async def test_request_sends_a_strict_schema_and_no_storage() -> None:
    model, responses = _model(_reply('{"answer": "Advantages."}'))

    await _complete(model)

    assert responses.request["text"] == {"format": structured_output_format(GenerationDraft)}
    assert responses.request["store"] is False
    assert isinstance(responses.request["reasoning"], Omit)


@pytest.mark.parametrize("output_type", _OUTPUT_TYPES, ids=lambda item: item.__name__)
def test_output_schemas_meet_openai_strict_mode(output_type: type[BaseModel]) -> None:
    response_format = structured_output_format(output_type)
    schema = response_format["schema"]

    assert response_format["strict"] is True
    assert "default" not in _keys(schema)
    for node in _objects(schema):
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])


def test_a_property_named_default_survives_default_removal() -> None:
    class Settings(BaseModel):
        default: str
        retries: int = 3

    schema = cast(dict[str, Any], structured_output_format(Settings)["schema"])

    assert set(schema["properties"]) == {"default", "retries"}
    assert "default" not in schema["properties"]["retries"]


async def test_reasoning_effort_is_sent_only_for_configured_models() -> None:
    model, responses = _model(
        _reply('{"answer": "Advantages."}'), reasoning_effort={"gpt-luna": "low"}
    )

    await _complete(model, name="gpt-luna")
    assert responses.request["reasoning"] == {"effort": "low"}

    await _complete(model, name="gpt-other")
    assert isinstance(responses.request["reasoning"], Omit)


async def test_truncated_reply_maps_to_provider_response_error_without_model_text() -> None:
    model, _ = _model(_reply('{"answer": "Acme bills 4'))

    with pytest.raises(ProviderResponseError, match="structured output was invalid") as caught:
        await _complete(model)

    assert "Acme" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


async def test_empty_reply_such_as_a_refusal_is_a_provider_response_error() -> None:
    model, _ = _model(_reply(""))

    with pytest.raises(ProviderResponseError, match="no structured output"):
        await _complete(model)


async def test_incomplete_reply_reports_its_reason() -> None:
    reply = _reply("")
    reply.status = "incomplete"
    reply.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    model, _ = _model(reply)

    with pytest.raises(ProviderResponseError, match="max_output_tokens"):
        await _complete(model)
