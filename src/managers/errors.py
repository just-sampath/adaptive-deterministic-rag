"""Translate database SDK failures at manager boundaries."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any

import httpx
from pymongo.errors import PyMongoError
from qdrant_client.common.client_exceptions import QdrantException, ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from utils.errors import (
    InfrastructureError,
    OperationTimeoutError,
    ProviderError,
    TransientProviderError,
)
from utils.retry import retry_after_seconds


def mongo_errors[**P, T](
    function: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(function)
    async def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await function(*args, **kwargs)
        except PyMongoError as exc:
            # The driver marks its own expired deadlines, including server selection and
            # timeoutMS, with ``timeout``; every other driver failure is an outage.
            if exc.timeout:
                raise OperationTimeoutError("MongoDB operation timed out") from exc
            raise InfrastructureError("MongoDB operation failed") from exc

    return guarded


def qdrant_errors[**P, T](
    function: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(function)
    async def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await function(*args, **kwargs)
        except ResourceExhaustedResponse as exc:
            # A 429 that carried Retry-After; the client raises it outside its ApiException tree.
            raise TransientProviderError(
                "Qdrant rate limit reached", retry_after_seconds=max(0.0, exc.retry_after_s)
            ) from exc
        except QdrantException as exc:
            # The client raises its base exception for a 429 whose Retry-After isn't an integer.
            raise TransientProviderError("Qdrant rate limit reached") from exc
        except ResponseHandlingException as exc:
            # The client wraps the HTTPX transport failure it caught in ``source``.
            if isinstance(exc.source, httpx.TimeoutException):
                raise OperationTimeoutError("Qdrant operation timed out") from exc
            if isinstance(exc.source, httpx.NetworkError | httpx.RemoteProtocolError):
                raise TransientProviderError("Qdrant operation failed") from exc
            raise ProviderError("Qdrant operation failed") from exc
        except UnexpectedResponse as exc:
            status = exc.status_code
            if status is not None and (status == 429 or status >= 500):
                raise TransientProviderError(
                    "Qdrant operation failed", retry_after_seconds=retry_after_seconds(exc.headers)
                ) from exc
            raise ProviderError("Qdrant operation failed") from exc

    return guarded
