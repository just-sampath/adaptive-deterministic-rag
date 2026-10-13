"""Structured concurrency that cancels siblings and re-raises the first concrete failure."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable
from typing import Any


async def run_pair[A, B](
    first: Coroutine[Any, Any, A],
    second: Coroutine[Any, Any, B],
) -> tuple[A, B]:
    """Run two coroutines concurrently; a failure cancels the sibling."""

    try:
        async with asyncio.TaskGroup() as group:
            first_task = group.create_task(first)
            second_task = group.create_task(second)
    except BaseExceptionGroup as group_error:
        leaf = _first_leaf(group_error)
        # Bare re-raise (no `from None`) keeps the leaf's own __cause__.
        raise leaf  # noqa: B904
    return first_task.result(), second_task.result()


async def run_all[T](coroutines: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """Run coroutines concurrently and return their results in input order."""

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(coroutine) for coroutine in coroutines]
    except BaseExceptionGroup as group_error:
        leaf = _first_leaf(group_error)
        # Bare re-raise (no `from None`) keeps the leaf's own __cause__.
        raise leaf  # noqa: B904
    return [task.result() for task in tasks]


def _first_leaf(error: BaseExceptionGroup[BaseException]) -> BaseException:
    first = error.exceptions[0]
    if isinstance(first, BaseExceptionGroup):
        return _first_leaf(first)
    return first
