"""OpenAI adapters for structured text generation and embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from models.domain import EmbeddingResult, ModelResult, ModelUsage
from utils.errors import ProviderError, ProviderResponseError

SchemaT = TypeVar("SchemaT", bound=BaseModel)
_EMBEDDING_BATCH_SIZE = 256


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

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]:
        try:
            response = await self._client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=output_type,
                max_output_tokens=max_output_tokens,
                store=False,
            )
        except OpenAIError as exc:
            raise ProviderError("OpenAI structured model call failed") from exc
        except ValidationError:
            # The SDK parses output text inside parse(), so a truncated reply fails here. Its error
            # embeds partial model output, so it is replaced rather than chained.
            raise ProviderResponseError("OpenAI structured output was invalid") from None
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            msg = f"OpenAI response was incomplete: {reason}"
            raise ProviderResponseError(msg)
        if response.output_parsed is None:
            msg = "OpenAI returned no parsed structured output"
            raise ProviderResponseError(msg)

        try:
            parsed = output_type.model_validate(response.output_parsed)
        except ValidationError as exc:
            raise ProviderResponseError("OpenAI output failed schema validation") from exc
        return ModelResult(
            value=parsed,
            model=response.model,
            provider="openai",
            usage=_response_usage(response.usage),
            request_id=getattr(response, "_request_id", None),
        )


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
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
                dimensions=self._dimension,
                encoding_format="float",
            )
        except OpenAIError as exc:
            raise ProviderError("OpenAI embedding call failed") from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise ProviderResponseError("OpenAI returned an unexpected embedding count")
        if any(len(item.embedding) != self._dimension for item in ordered):
            raise ProviderResponseError(
                "Embedding dimension does not match configured vector collection"
            )
        request_id = getattr(response, "_request_id", None)
        batch_usage = ModelUsage(
            input_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
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
