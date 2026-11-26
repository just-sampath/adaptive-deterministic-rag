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
    TransientProviderError,
)
from utils.request_context import collect_operations, observed_call

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


def _cohere_metered(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "rerank-1",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "meta": {"billed_units": {"search_units": 1}},
        },
        request=request,
    )


async def test_cohere_records_its_request_id_and_billed_units_but_no_tokens() -> None:
    with collect_operations() as collector:
        ranked = await observed_call("rerank", "rerank-test", _rerank(_cohere_metered))

    assert len(ranked) == 1  # type: ignore[arg-type]
    (row,) = collector.snapshot()
    assert row.provider == "cohere"
    assert row.request_id == "rerank-1"
    assert row.rerank_metering is not None
    assert row.rerank_metering.billed_units == {"search_units": 1.0}
    # Search units are a billing unit, never tokens.
    assert row.usage is None


def _openai_status(status: int, headers: dict[str, str] | None = None) -> _Handler2:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, json={"error": {"message": "failed"}}, headers=headers, request=request
        )

    return handler


@pytest.mark.parametrize("call", [_complete, _embed], ids=["structured", "embedding"])
@pytest.mark.parametrize(
    ("handler", "transient", "retry_after"),
    [
        (_openai_refused, True, None),
        (_openai_status(429, {"retry-after": "2"}), True, 2.0),
        (_openai_status(500), True, None),
        (_openai_status(408), True, None),
        (_openai_status(409), True, None),
        (_openai_status(500, {"x-should-retry": "false"}), False, None),
        (_openai_status(400, {"x-should-retry": "true"}), True, None),
        (_openai_status(400), False, None),
        (_openai_unauthorized, False, None),
    ],
    ids=[
        "connection",
        "rate-limit",
        "server",
        "request-timeout",
        "conflict",
        "server-says-no-retry",
        "server-says-retry",
        "bad-request",
        "auth",
    ],
)
async def test_openai_failures_the_sdk_would_retry_are_transient(
    call: Callable[[AsyncOpenAI], object],
    handler: _Handler2,
    transient: bool,
    retry_after: float | None,
) -> None:
    with pytest.raises(ProviderError) as caught:
        await call(_openai(handler))  # type: ignore[misc]

    assert isinstance(caught.value, TransientProviderError) is transient
    if isinstance(caught.value, TransientProviderError):
        assert caught.value.retry_after_seconds == retry_after


def _status(status: int, headers: dict[str, str] | None = None) -> _Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={}, headers=headers, request=request)

    return handler


@pytest.mark.parametrize(
    ("handler", "transient", "retry_after"),
    [
        (_cohere_refused, True, None),
        (_status(429, {"retry-after": "1"}), True, 1.0),
        (_status(499), True, None),
        (_status(500), True, None),
        (_cohere_unavailable, True, None),
        (_status(400), False, None),
        (_status(401), False, None),
        (_status(402), False, None),
        (_status(404), False, None),
    ],
    ids=[
        "connection",
        "rate-limit",
        "cancelled",
        "server",
        "unavailable",
        "bad-request",
        "auth",
        "billing",
        "not-found",
    ],
)
async def test_cohere_temporary_failures_are_transient(
    handler: _Handler, transient: bool, retry_after: float | None
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _rerank(handler)

    assert isinstance(caught.value, TransientProviderError) is transient
    if isinstance(caught.value, TransientProviderError):
        assert caught.value.retry_after_seconds == retry_after


@pytest.mark.parametrize(
    ("handler", "retry_after"),
    [
        (_status(429, {"Retry-After": "3"}), 3.0),
        (_status(429, {"Retry-After": "soon"}), None),
        (_status(429), None),
        (_cohere_unavailable, None),
        (_cohere_refused, None),
    ],
    ids=["rate-limit", "unreadable-retry-after", "rate-limit-no-header", "unavailable", "refused"],
)
async def test_qdrant_rate_limits_and_outages_are_transient_service_errors(
    handler: _Handler, retry_after: float | None
) -> None:
    # The client raises its rate-limit error outside its ApiException tree; it must still map.
    with pytest.raises(TransientProviderError) as caught:
        await _qdrant_health(handler)

    assert isinstance(caught.value, InfrastructureError)
    assert caught.value.retry_after_seconds == retry_after


async def test_qdrant_client_errors_are_not_transient() -> None:
    with pytest.raises(ProviderError) as caught:
        await _qdrant_health(_status(400))

    assert not isinstance(caught.value, TransientProviderError)
