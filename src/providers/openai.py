"""OpenAI adapters for structured text generation and embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    Omit,
    OpenAIError,
    omit,
)
from openai.lib._pydantic import to_strict_json_schema
from openai.types.responses import ResponseFormatTextJSONSchemaConfigParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, ValidationError

from models.domain import EmbeddingResult, ModelResult, ModelUsage
from utils.errors import (
    OperationTimeoutError,
    ProviderError,
    ProviderResponseError,
    TransientProviderError,
)
from utils.request_context import observe_response, operation
from utils.retry import retry_after_seconds

SchemaT = TypeVar("SchemaT", bound=BaseModel)
_EMBEDDING_BATCH_SIZE = 256
# Besides any 5xx, the statuses the SDK's own retry rule accepts (openai 3.3.1 ``_should_retry``).
_TRANSIENT_STATUSES = frozenset({408, 409, 429})
# Schema keys whose values map names to subschemas; the names themselves are not keywords.
_NAMED_SUBSCHEMAS = frozenset({"properties", "$defs", "definitions", "patternProperties"})


def build_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    max_retries: int,
    timeout_seconds: float,
) -> AsyncOpenAI:
    """Create one asynchronous OpenAI client for the process."""

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout_seconds,
    )


class OpenAITextModel:
    """Execute schema-constrained model calls through the Responses API."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        reasoning_effort: Mapping[str, str] | None = None,
    ) -> None:
        self._client = client
        self._reasoning_effort = dict(reasoning_effort or {})

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]:
        with operation("text", model, provider="openai", adapter=True):
            try:
                response = await self._client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={"format": structured_output_format(output_type)},
                    max_output_tokens=max_output_tokens,
                    store=False,
                    reasoning=self._reasoning(model),
                )
            except APITimeoutError as exc:
                raise OperationTimeoutError("OpenAI structured model call timed out") from exc
            except OpenAIError as exc:
                raise _provider_error("OpenAI structured model call failed", exc) from exc
            usage = _response_usage(response.usage)
            request_id = getattr(response, "_request_id", None)
            observe_response(
                provider="openai",
                model=response.model,
                request_id=request_id,
                usage=usage if response.usage is not None else None,
            )
            if response.status == "incomplete":
                reason = getattr(response.incomplete_details, "reason", "unknown")
                msg = f"OpenAI response was incomplete: {reason}"
                raise ProviderResponseError(msg)
            if not response.output_text:
                msg = "OpenAI returned no structured output"
                raise ProviderResponseError(msg)

            try:
                parsed = output_type.model_validate_json(response.output_text)
            except ValidationError:
                # Validation errors embed the model's output, so the error is replaced, not chained.
                raise ProviderResponseError("OpenAI structured output was invalid") from None
            return ModelResult(
                value=parsed,
                model=response.model,
                provider="openai",
                usage=usage,
                request_id=request_id,
            )

    def _reasoning(self, model: str) -> Reasoning | Omit:
        effort = self._reasoning_effort.get(model)
        return omit if effort is None else Reasoning(effort=cast(ReasoningEffort, effort))


def structured_output_format(
    output_type: type[BaseModel],
) -> ResponseFormatTextJSONSchemaConfigParam:
    """Return the strict JSON-schema response format for a Pydantic output model.

    Strict mode requires every property and rejects ``default``, so defaults are removed from the
    SDK's strict schema. The defaults still apply when the application builds models itself.
    """

    return ResponseFormatTextJSONSchemaConfigParam(
        type="json_schema",
        name=output_type.__name__,
        schema=_without_defaults(to_strict_json_schema(output_type)),
        strict=True,
    )


def _without_defaults(node: Any) -> Any:
    if isinstance(node, list):
        return [_without_defaults(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: (
            {name: _without_defaults(schema) for name, schema in value.items()}
            if key in _NAMED_SUBSCHEMAS and isinstance(value, dict)
            else _without_defaults(value)
        )
        for key, value in node.items()
        if key != "default"
    }


class OpenAIEmbeddingProvider:
    """Create query and document embeddings with one configured model version."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        dimension: int,
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fingerprint(self) -> str:
        return f"openai:{self._model}:{self._dimension}"

    async def embed_query(self, text: str) -> EmbeddingResult:
        (result,) = await self._request_embeddings([text])
        return result

    async def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
            results.extend(await self._request_embeddings(batch))
        return results

    async def _request_embeddings(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        with operation("embedding", self._model, provider="openai", adapter=True):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=list(texts),
                    dimensions=self._dimension,
                    encoding_format="float",
                )
            except APITimeoutError as exc:
                raise OperationTimeoutError("OpenAI embedding call timed out") from exc
            except OpenAIError as exc:
                raise _provider_error("OpenAI embedding call failed", exc) from exc
            request_id = getattr(response, "_request_id", None)
            batch_usage = (
                ModelUsage(
                    input_tokens=response.usage.prompt_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage is not None
                else ModelUsage()
            )
            observe_response(
                provider="openai",
                model=response.model,
                request_id=request_id,
                usage=batch_usage if response.usage is not None else None,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(texts):
                raise ProviderResponseError("OpenAI returned an unexpected embedding count")
            if any(len(item.embedding) != self._dimension for item in ordered):
                raise ProviderResponseError(
                    "Embedding dimension does not match configured vector collection"
                )
            # The batch usage is reported once, so summing results matches the billed total.
            return [
                EmbeddingResult(
                    vector=item.embedding,
                    model=response.model,
                    provider="openai",
                    usage=batch_usage if position == 0 else ModelUsage(),
                    request_id=request_id,
                )
                for position, item in enumerate(ordered)
            ]


def _provider_error(message: str, error: OpenAIError) -> ProviderError:
    """Mark what the SDK would retry on its own as transient; the application retries it instead.

    Timeouts are translated before this is reached. The SDK reports any other transport failure as
    a connection error, a TLS or protocol failure included, so those get one retry too.
    """

    if isinstance(error, APIConnectionError):
        return TransientProviderError(message)
    if isinstance(error, APIStatusError):
        headers = error.response.headers
        advice = headers.get("x-should-retry")
        status = error.status_code
        if advice == "true" or (
            advice != "false" and (status in _TRANSIENT_STATUSES or status >= 500)
        ):
            return TransientProviderError(message, retry_after_seconds=retry_after_seconds(headers))
    return ProviderError(message)


def _response_usage(usage: object | None) -> ModelUsage:
    if usage is None:
        return ModelUsage()
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return ModelUsage(
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
        total_tokens=getattr(usage, "total_tokens", 0),
        cached_input_tokens=getattr(input_details, "cached_tokens", 0) if input_details else 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) if output_details else 0,
    )
