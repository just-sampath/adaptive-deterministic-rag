"""Translate database SDK failures at manager boundaries."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any

from pymongo.errors import PyMongoError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from utils.errors import InfrastructureError, ProviderError


def mongo_errors[**P, T](
    function: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(function)
    async def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await function(*args, **kwargs)
        except PyMongoError as exc:
            raise InfrastructureError("MongoDB operation failed") from exc

    return guarded


def qdrant_errors[**P, T](
    function: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(function)
    async def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await function(*args, **kwargs)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise ProviderError("Qdrant operation failed") from exc

    return guarded
