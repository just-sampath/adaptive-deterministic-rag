"""Cohere v2 reranker provider."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Any

import httpx

from models.domain import RankedChunk, RerankMetering, RerankResult
from utils.errors import (
    OperationTimeoutError,
    ProviderError,
    ProviderResponseError,
    TransientProviderError,
)
from utils.request_context import observe_response, operation
from utils.retry import retry_after_seconds

# Our choice of retryable statuses, read against https://docs.cohere.com/reference/errors: 429 is
# its rate limit, and 499 is a request cancelled on Cohere's side, which the page says to try
# again. The page doesn't call 5xx retryable; we retry it as a server-side failure.
_TRANSIENT_STATUSES = frozenset({429, 499})


class CohereReranker:
    """Rerank candidate text with Cohere while retaining retrieval provenance."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float,
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "X-Client-Name": "adaptive-rag"},
            timeout=timeout_seconds,
        )

    @property
    def version(self) -> str:
        return f"cohere:{self._model}"

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RankedChunk],
        top_n: int,
    ) -> RerankResult:
        if not chunks:
            return RerankResult([], request_id=None, metering=None)
        with operation("rerank", self._model, provider="cohere", adapter=True):
            try:
                response = await self._client.post(
                    "/v2/rerank",
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": [item.chunk.text for item in chunks],
                        "top_n": min(top_n, len(chunks)),
                    },
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise OperationTimeoutError("Cohere reranking call timed out") from exc
            except httpx.HTTPStatusError as exc:
                raise _status_error(exc.response) from exc
            except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                raise TransientProviderError("Cohere reranking call failed") from exc
            except httpx.HTTPError as exc:
                raise ProviderError("Cohere reranking call failed") from exc
            try:
                payload: Any = response.json()
            except ValueError as exc:
                raise ProviderResponseError("Cohere returned invalid JSON") from exc
            if not isinstance(payload, dict):
                msg = "Cohere rerank response must be an object"
                raise ProviderResponseError(msg)
            request_id = payload.get("id") if isinstance(payload.get("id"), str) else None
            metering = _metering(payload.get("meta"))
            usage = metering.token_usage() if metering else None
            observe_response(
                provider="cohere",
                model=self._model,
                request_id=request_id,
                usage=usage,
                metering=metering,
            )
            results = payload.get("results")
            if not isinstance(results, list):
                msg = "Cohere rerank response omitted results"
                raise ProviderResponseError(msg)

            reranked: list[RankedChunk] = []
            seen: set[int] = set()
            for rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    msg = "Cohere rerank result must be an object"
                    raise ProviderResponseError(msg)
                index = result.get("index")
                score = result.get("relevance_score")
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < len(chunks)
                    or index in seen
                ):
                    msg = "Cohere returned an invalid or duplicate document index"
                    raise ProviderResponseError(msg)
                if (
                    not isinstance(score, int | float)
                    or isinstance(score, bool)
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    msg = "Cohere returned an invalid relevance score"
                    raise ProviderResponseError(msg)
                seen.add(index)
                reranked.append(
                    chunks[index].model_copy(
                        update={"reranker_score": float(score), "reranker_rank": rank}
                    )
                )
            return RerankResult(reranked, request_id=request_id, metering=metering)

    async def close(self) -> None:
        await self._client.aclose()


def _status_error(response: httpx.Response) -> ProviderError:
    status = response.status_code
    if status in _TRANSIENT_STATUSES or status >= 500:
        return TransientProviderError(
            "Cohere reranking call failed",
            retry_after_seconds=retry_after_seconds(response.headers),
        )
    return ProviderError("Cohere reranking call failed")


def _metering(meta: object) -> RerankMetering | None:
    if not isinstance(meta, dict):
        return None
    billed = meta.get("billed_units")
    tokens = meta.get("tokens")
    units = {}
    if isinstance(billed, dict):
        for name in (
            "search_units",
            "input_tokens",
            "output_tokens",
            "image_tokens",
            "text_tokens",
            "images",
            "classifications",
            "pages",
        ):
            value = _counter(billed.get(name))
            if value is not None:
                units[name] = value
    input_tokens = _counter(tokens.get("input_tokens")) if isinstance(tokens, dict) else None
    output_tokens = _counter(tokens.get("output_tokens")) if isinstance(tokens, dict) else None
    cached = _counter(meta.get("cached_tokens"))
    if not units and input_tokens is None and output_tokens is None and cached is None:
        return None
    return RerankMetering(
        billed_units=units,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
    )


def _counter(value: object) -> float | None:
    if (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and 0 <= value <= sys.float_info.max
    ):
        return float(value)
    return None
