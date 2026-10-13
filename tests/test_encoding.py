from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, OpenAIError

from models.domain import EmbeddingResult, ModelUsage
from providers.openai import OpenAIEmbeddingProvider, build_openai_client
from providers.sparse import Bm25SparseEncoder
from services.encoding import EmbeddingCache, HybridQueryEncoder
from tests.fakes import FakeEmbeddings
from utils.errors import ProviderError, ProviderResponseError
from utils.settings import SparseConfig


class GatedEmbeddings:
    """Holds every provider call until released so concurrent misses overlap."""

    model = "gated-model"
    dimension = 3
    fingerprint = "gated:3"

    def __init__(self, failures: int = 0) -> None:
        self.calls = 0
        self.failures_remaining = failures
        self.release = asyncio.Event()

    async def embed_query(self, text: str) -> EmbeddingResult:
        self.calls += 1
        await self.release.wait()
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ProviderError("embedding unavailable")
        return EmbeddingResult(
            vector=[float(len(text)), 0.0, 1.0],
            model=self.model,
            provider="fake",
            usage=ModelUsage(total_tokens=2),
            request_id=f"gated-{self.calls}",
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        return [await self.embed_query(text) for text in texts]


class OtherFingerprint(FakeEmbeddings):
    fingerprint = "fake-embedding:4"


class HangingEmbeddings(FakeEmbeddings):
    async def embed_query(self, text: str) -> EmbeddingResult:
        del text
        await asyncio.Event().wait()
        raise AssertionError("the embedding stage should have timed out")


async def test_concurrent_identical_misses_share_one_provider_call() -> None:
    provider = GatedEmbeddings()
    cache = EmbeddingCache(max_entries=4)
    tasks = [asyncio.create_task(cache.get_or_create("same query", provider)) for _ in range(5)]
    await asyncio.sleep(0)

    provider.release.set()
    results = await asyncio.gather(*tasks)

    assert provider.calls == 1
    assert sum(not result.cached for result in results) == 1
    assert all(result.vector == results[0].vector for result in results)


async def test_failed_embedding_reaches_all_waiters_and_is_not_cached() -> None:
    provider = GatedEmbeddings(failures=1)
    cache = EmbeddingCache(max_entries=4)
    tasks = [asyncio.create_task(cache.get_or_create("query", provider)) for _ in range(3)]
    await asyncio.sleep(0)

    provider.release.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    retry = await cache.get_or_create("query", provider)

    assert all(isinstance(outcome, ProviderError) for outcome in outcomes)
    assert provider.calls == 2
    assert retry.cached is False


async def test_cancelled_leader_hands_the_miss_to_a_waiter_instead_of_failing_it() -> None:
    provider = GatedEmbeddings()
    cache = EmbeddingCache(max_entries=4)
    leader = asyncio.create_task(cache.get_or_create("query", provider))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(cache.get_or_create("query", provider))
    await asyncio.sleep(0)

    leader.cancel()
    provider.release.set()
    with pytest.raises(asyncio.CancelledError):
        await leader
    result = await waiter

    assert provider.calls == 2
    assert result.cached is False
    assert result.vector == [5.0, 0.0, 1.0]
    assert cache._inflight == {}


async def test_leader_timeout_does_not_fail_a_waiter_that_still_has_budget() -> None:
    provider = GatedEmbeddings()
    cache = EmbeddingCache(max_entries=4)
    leader_encoder = HybridQueryEncoder(
        dense=provider,
        sparse=_sparse(),
        cache=cache,
        embedding_timeout_seconds=0.05,
    )
    waiter_encoder = HybridQueryEncoder(
        dense=provider,
        sparse=_sparse(),
        cache=cache,
        embedding_timeout_seconds=5,
    )
    leader = asyncio.create_task(leader_encoder.encode("query"))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(waiter_encoder.encode("query"))

    with pytest.raises(TimeoutError):
        await leader
    provider.release.set()
    encoded = await waiter

    assert provider.calls == 2
    assert [call.operation for call in encoded.model_calls] == ["embedding"]


async def test_cache_keys_are_case_sensitive_and_whitespace_normalized() -> None:
    provider = FakeEmbeddings()
    cache = EmbeddingCache(max_entries=4)

    await cache.get_or_create("Query", provider)
    await cache.get_or_create("query", provider)
    await cache.get_or_create("  Query   ", provider)

    assert provider.query_calls == 2


async def test_cache_keys_include_the_provider_fingerprint() -> None:
    cache = EmbeddingCache(max_entries=4)
    first = FakeEmbeddings()
    second = OtherFingerprint()

    await cache.get_or_create("query", first)
    await cache.get_or_create("query", second)

    assert first.query_calls == 1
    assert second.query_calls == 1


async def test_cache_hit_reports_no_usage_or_request_and_keeps_the_vector() -> None:
    provider = FakeEmbeddings()
    cache = EmbeddingCache(max_entries=4)

    first = await cache.get_or_create("query", provider)
    second = await cache.get_or_create("query", provider)

    assert first.cached is False
    assert second.cached is True
    assert second.usage == ModelUsage()
    assert second.request_id is None
    assert second.vector == first.vector


async def test_least_recently_used_entry_is_evicted() -> None:
    provider = FakeEmbeddings()
    cache = EmbeddingCache(max_entries=1)

    await cache.get_or_create("alpha", provider)
    await cache.get_or_create("beta", provider)
    await cache.get_or_create("alpha", provider)

    assert provider.query_calls == 3


async def test_zero_entry_cache_always_calls_the_provider() -> None:
    provider = FakeEmbeddings()
    cache = EmbeddingCache(max_entries=0)

    await cache.get_or_create("query", provider)
    result = await cache.get_or_create("query", provider)

    assert provider.query_calls == 2
    assert result.cached is False


def _sparse() -> Bm25SparseEncoder:
    return Bm25SparseEncoder(SparseConfig())


async def test_hybrid_encoder_records_the_embedding_call_only_when_not_cached() -> None:
    encoder = HybridQueryEncoder(
        dense=FakeEmbeddings(),
        sparse=_sparse(),
        cache=EmbeddingCache(max_entries=4),
        embedding_timeout_seconds=1,
    )

    first = await encoder.encode("error code")
    second = await encoder.encode("error code")

    assert first.dense == (10.0, 0.0, 1.0)
    assert [call.operation for call in first.model_calls] == ["embedding"]
    assert first.model_calls[0].request_id == "embedding-1"
    assert second.model_calls == ()
    assert second.usage == ModelUsage()
    assert second.sparse == first.sparse
    assert first.sparse.indices


def test_hybrid_encoder_composes_fingerprint_and_exposes_sparse_requirements() -> None:
    sparse = _sparse()
    encoder = HybridQueryEncoder(
        dense=FakeEmbeddings(),
        sparse=sparse,
        cache=EmbeddingCache(max_entries=0),
        embedding_timeout_seconds=1,
    )

    assert encoder.fingerprint == f"fake-embedding:3|{sparse.fingerprint}"
    assert encoder.dense_dimension == 3
    assert encoder.requires_idf_modifier is True


async def test_embedding_stage_is_bounded_by_its_timeout() -> None:
    encoder = HybridQueryEncoder(
        dense=HangingEmbeddings(),
        sparse=_sparse(),
        cache=EmbeddingCache(max_entries=0),
        embedding_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await encoder.encode("query")


class _EmbeddingsApi:
    """Stands in for the OpenAI embeddings resource; returns items in reverse order."""

    def __init__(
        self,
        *,
        dimension: int,
        error: Exception | None = None,
        drop_last: bool = False,
    ) -> None:
        self.dimension = dimension
        self.error = error
        self.drop_last = drop_last
        self.batch_sizes: list[int] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        texts: list[str] = kwargs["input"]
        self.batch_sizes.append(len(texts))
        if self.error is not None:
            raise self.error
        items = [
            SimpleNamespace(
                index=index,
                embedding=[float(len(text))] + [0.0] * (self.dimension - 1),
            )
            for index, text in enumerate(texts)
        ]
        if self.drop_last:
            items = items[:-1]
        return SimpleNamespace(
            data=list(reversed(items)),
            model="text-embedding-3-small",
            usage=SimpleNamespace(prompt_tokens=len(texts), total_tokens=len(texts)),
        )


class _FakeClient:
    def __init__(self, embeddings: _EmbeddingsApi) -> None:
        self.embeddings = embeddings


def _provider(api: _EmbeddingsApi, *, dimension: int = 3) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        cast(AsyncOpenAI, _FakeClient(api)),
        model="text-embedding-3-small",
        dimension=dimension,
    )


async def test_embed_documents_batches_inputs_and_preserves_input_order() -> None:
    api = _EmbeddingsApi(dimension=3)
    texts = [f"text {index}" for index in range(300)]

    results = await _provider(api).embed_documents(texts)

    assert api.batch_sizes == [256, 44]
    assert [result.vector[0] for result in results] == [float(len(text)) for text in texts]
    assert sum(result.usage.total_tokens for result in results) == 300


async def test_embed_documents_reports_the_provider_fingerprint() -> None:
    provider = _provider(_EmbeddingsApi(dimension=3))

    assert provider.fingerprint == "openai:text-embedding-3-small:3"


async def test_embedding_count_mismatch_is_a_response_error() -> None:
    provider = _provider(_EmbeddingsApi(dimension=3, drop_last=True))

    with pytest.raises(ProviderResponseError, match="unexpected embedding count"):
        await provider.embed_documents(["first", "second"])


async def test_embedding_dimension_mismatch_is_a_response_error() -> None:
    provider = _provider(_EmbeddingsApi(dimension=2), dimension=3)

    with pytest.raises(ProviderResponseError, match="dimension"):
        await provider.embed_query("query")


async def test_transport_failure_is_translated_to_a_provider_error() -> None:
    provider = _provider(_EmbeddingsApi(dimension=3, error=OpenAIError("network down")))

    with pytest.raises(ProviderError, match="embedding call failed"):
        await provider.embed_query("query")


async def test_build_openai_client_applies_retry_budget() -> None:
    client = build_openai_client(
        api_key="test-key",
        base_url=None,
        max_retries=3,
        timeout_seconds=7,
    )

    assert client.max_retries == 3
    await client.close()
