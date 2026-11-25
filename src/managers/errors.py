"""Translate database SDK failures at manager boundaries."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any

import httpx
from pymongo.errors import PyMongoError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from utils.errors import InfrastructureError, OperationTimeoutError, ProviderError


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
        except ResponseHandlingException as exc:
            # The client wraps the HTTPX transport failure it caught in ``source``.
            if isinstance(exc.source, httpx.TimeoutException):
                raise OperationTimeoutError("Qdrant operation timed out") from exc
            raise ProviderError("Qdrant operation failed") from exc
        except UnexpectedResponse as exc:
            raise ProviderError("Qdrant operation failed") from exc

    return guarded
