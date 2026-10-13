"""Cohere v2 reranker provider."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from models.domain import RankedChunk
from utils.errors import ProviderError, ProviderResponseError


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
    ) -> list[RankedChunk]:
        if not chunks:
            return []
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
        except httpx.HTTPError as exc:
            raise ProviderError("Cohere reranking call failed") from exc
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise ProviderResponseError("Cohere returned invalid JSON") from exc
        if not isinstance(payload, dict):
            msg = "Cohere rerank response must be an object"
            raise ProviderResponseError(msg)
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
        return reranked

    async def close(self) -> None:
        await self._client.aclose()
