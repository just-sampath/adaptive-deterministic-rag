"""Query encoding: single-flight cached dense embeddings paired with sparse terms."""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import replace

from models.domain import EmbeddingResult, EncodedQuery, ModelCallTrace, ModelUsage
from services.contracts import EmbeddingProvider, SparseEncoder


class EmbeddingCache:
    """Bounded LRU embedding cache where concurrent identical misses share one provider call."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._values: OrderedDict[str, EmbeddingResult] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[EmbeddingResult]] = {}

    async def get_or_create(self, text: str, provider: EmbeddingProvider) -> EmbeddingResult:
        if self._max_entries == 0:
            return await provider.embed_query(text)
        key = _cache_key(provider.fingerprint, text)
        while True:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                return _cache_hit(cached)
            pending = self._inflight.get(key)
            if pending is None:
                return await self._fill(key, text, provider)
            # Waiting does not cancel the shared future. A cancelled leader leaves no result, so
            # the waiter loops back and becomes the leader under its own budget.
            await asyncio.wait({pending})
            if not pending.cancelled():
                return _cache_hit(pending.result())

    async def _fill(self, key: str, text: str, provider: EmbeddingProvider) -> EmbeddingResult:
        future: asyncio.Future[EmbeddingResult] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            result = await provider.embed_query(text)
        except asyncio.CancelledError:
            # The leader's own cancellation (timeout or disconnect) is not a provider failure.
            future.cancel()
            raise
        except Exception as exc:
            _fail(future, exc)
            raise
        else:
            future.set_result(result)
            self._store(key, result)
            return result
        finally:
            del self._inflight[key]

    def _store(self, key: str, result: EmbeddingResult) -> None:
        self._values[key] = result
        self._values.move_to_end(key)
        while len(self._values) > self._max_entries:
            self._values.popitem(last=False)


class HybridQueryEncoder:
    """Encode one query into dense and sparse vectors under one embedding timeout."""

    def __init__(
        self,
        *,
        dense: EmbeddingProvider,
        sparse: SparseEncoder,
        cache: EmbeddingCache,
        embedding_timeout_seconds: float,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._cache = cache
        self._embedding_timeout_seconds = embedding_timeout_seconds

    @property
    def fingerprint(self) -> str:
        return f"{self._dense.fingerprint}|{self._sparse.fingerprint}"

    @property
    def dense_dimension(self) -> int:
        return self._dense.dimension

    @property
    def requires_idf_modifier(self) -> bool:
        return self._sparse.requires_idf_modifier

    async def encode(self, text: str) -> EncodedQuery:
        async with asyncio.timeout(self._embedding_timeout_seconds):
            embedding = await self._cache.get_or_create(text, self._dense)
        return EncodedQuery(
            text=text,
            dense=tuple(embedding.vector),
            sparse=self._sparse.encode_query(text),
            usage=embedding.usage,
            model_calls=() if embedding.cached else (_embedding_call(embedding),),
        )


def _embedding_call(embedding: EmbeddingResult) -> ModelCallTrace:
    return ModelCallTrace(
        operation="embedding",
        provider=embedding.provider,
        model=embedding.model,
        request_id=embedding.request_id,
        usage=embedding.usage,
    )


def _cache_key(fingerprint: str, text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(f"{fingerprint}\x00{normalized}".encode()).hexdigest()


def _cache_hit(result: EmbeddingResult) -> EmbeddingResult:
    """Return a reusable copy that reports no provider usage or request."""

    return replace(
        result,
        vector=list(result.vector),
        usage=ModelUsage(),
        request_id=None,
        cached=True,
    )


def _fail(future: asyncio.Future[EmbeddingResult], error: BaseException) -> None:
    future.set_exception(error)
    # Retrieve the exception here so a failure without waiters is not logged at garbage collection.
    future.exception()
