"""Each adapter turns its own transport timeout into OperationTimeoutError, and nothing else."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    ExecutionTimeout,
    NetworkTimeout,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)
from qdrant_client import AsyncQdrantClient

from managers.errors import mongo_errors
from managers.vector import QdrantVectorManager
from models.domain import GenerationDraft, RankedChunk
from providers.cohere import CohereReranker
from providers.openai import OpenAIEmbeddingProvider, OpenAITextModel
from tests.fakes import make_chunk
from utils.errors import (
    InfrastructureError,
    OperationTimeoutError,
    ProviderError,
    ProviderResponseError,
)

_Handler2 = Callable[[httpx2.Request], httpx2.Response]
_Handler = Callable[[httpx.Request], httpx.Response]


def _openai(handler: _Handler2) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="http://openai.test/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _openai_timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("slow", request=request)


def _openai_refused(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("refused", request=request)


def _openai_unauthorized(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(401, json={"error": {"message": "bad key"}}, request=request)


async def _complete(client: AsyncOpenAI) -> object:
    return await OpenAITextModel(client).complete(
        model="test-model",
        system_prompt="system",
        user_prompt="user",
        output_type=GenerationDraft,
        max_output_tokens=16,
    )


async def _embed(client: AsyncOpenAI) -> object:
    return await OpenAIEmbeddingProvider(client, model="test-embedding", dimension=3).embed_query(
        "text"
    )


@pytest.mark.parametrize("call", [_complete, _embed], ids=["structured", "embedding"])
async def test_openai_transport_timeout_is_an_operation_timeout(
    call: Callable[[AsyncOpenAI], object],
) -> None:
    # The SDK raises APITimeoutError from the transport's own timeout exception.
    with pytest.raises(OperationTimeoutError) as caught:
        await call(_openai(_openai_timeout))  # type: ignore[misc]

    assert not isinstance(caught.value, InfrastructureError)


@pytest.mark.parametrize("call", [_complete, _embed], ids=["structured", "embedding"])
@pytest.mark.parametrize(
    "handler", [_openai_refused, _openai_unauthorized], ids=["connection", "auth"]
)
async def test_other_openai_failures_stay_service_errors(
    call: Callable[[AsyncOpenAI], object], handler: _Handler2
) -> None:
    with pytest.raises(ProviderError) as caught:
        await call(_openai(handler))  # type: ignore[misc]

    assert not isinstance(caught.value, TimeoutError)


def _chunk() -> RankedChunk:
    chunk = make_chunk("00000000-0000-0000-0000-000000000001")
    return RankedChunk(chunk=chunk, fusion_score=1.0, fusion_rank=1)


async def _rerank(handler: _Handler) -> object:
    reranker = CohereReranker(
        api_key="test-key", model="rerank-test", base_url="http://cohere.test", timeout_seconds=1
    )
    await reranker.close()
    reranker._client = httpx.AsyncClient(
        base_url="http://cohere.test", transport=httpx.MockTransport(handler)
    )
    try:
        return await reranker.rerank("query", [_chunk()], top_n=1)
    finally:
        await reranker.close()


def _cohere_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("slow", request=request)


def _cohere_pool_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.PoolTimeout("no connection", request=request)


def _cohere_refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _cohere_unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, request=request)


def _cohere_malformed(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"not json", request=request)


@pytest.mark.parametrize("handler", [_cohere_timeout, _cohere_pool_timeout])
async def test_cohere_transport_timeout_is_an_operation_timeout(handler: _Handler) -> None:
    with pytest.raises(OperationTimeoutError):
        await _rerank(handler)


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (_cohere_refused, ProviderError),
        (_cohere_unavailable, ProviderError),
        (_cohere_malformed, ProviderResponseError),
    ],
    ids=["connection", "status", "malformed"],
)
async def test_other_cohere_failures_stay_service_errors(
    handler: _Handler, expected: type[Exception]
) -> None:
    with pytest.raises(expected) as caught:
        await _rerank(handler)

    assert not isinstance(caught.value, TimeoutError)


async def _qdrant_health(handler: _Handler) -> None:
    client = AsyncQdrantClient(url="http://qdrant.test:6333", check_compatibility=False)
    # The REST client sends through this HTTPX client and wraps whatever it raises.
    client._client.http.client._async_client = httpx.AsyncClient(  # type: ignore[attr-defined]
        transport=httpx.MockTransport(handler)
    )
    manager = QdrantVectorManager(
        client,
        collection_alias="chunks",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
    )
    try:
        await manager.health_check()
    finally:
        await client.close()


async def test_qdrant_transport_timeout_inside_the_sdk_wrapper_is_an_operation_timeout() -> None:
    with pytest.raises(OperationTimeoutError):
        await _qdrant_health(_cohere_timeout)


@pytest.mark.parametrize(
    "handler", [_cohere_refused, _cohere_unavailable], ids=["connection", "status"]
)
async def test_other_qdrant_failures_stay_service_errors(handler: _Handler) -> None:
    with pytest.raises(ProviderError) as caught:
        await _qdrant_health(handler)

    assert not isinstance(caught.value, TimeoutError)


@mongo_errors
async def _mongo_call(error: PyMongoError) -> None:
    raise error


@pytest.mark.parametrize(
    "error",
    [
        ServerSelectionTimeoutError("no server"),
        NetworkTimeout("socket timeout"),
        ExecutionTimeout("operation exceeded time limit", code=50),
    ],
    ids=["server-selection", "network", "execution"],
)
async def test_mongo_driver_timeouts_are_operation_timeouts(error: PyMongoError) -> None:
    assert error.timeout

    with pytest.raises(OperationTimeoutError):
        await _mongo_call(error)


@pytest.mark.parametrize(
    "error",
    [
        AutoReconnect("connection reset"),
        ConnectionFailure("refused"),
        OperationFailure("unauthorized", code=13),
    ],
    ids=["reconnect", "connection", "operation"],
)
async def test_other_mongo_failures_stay_service_errors(error: PyMongoError) -> None:
    assert not error.timeout

    with pytest.raises(InfrastructureError) as caught:
        await _mongo_call(error)

    assert not isinstance(caught.value, TimeoutError)
