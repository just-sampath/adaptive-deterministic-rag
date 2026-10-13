from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from utils.concurrency import run_all, run_pair


class _BoomError(Exception):
    pass


async def _never() -> None:
    await asyncio.sleep(10)


async def _recording_sibling(cancelled: asyncio.Event) -> None:
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        cancelled.set()
        raise


async def _fail_after_a_tick(message: str) -> None:
    await asyncio.sleep(0)
    raise _BoomError(message)


async def _delayed(value: int, delay: float) -> int:
    await asyncio.sleep(delay)
    return value


async def test_run_pair_cancels_the_sibling_and_raises_the_failing_error() -> None:
    cancelled = asyncio.Event()

    with pytest.raises(_BoomError, match="first failure"):
        await run_pair(_recording_sibling(cancelled), _fail_after_a_tick("first failure"))

    assert cancelled.is_set()


async def test_run_pair_returns_both_results_in_argument_order() -> None:
    first, second = await run_pair(_delayed(1, 0.02), _delayed(2, 0.0))

    assert (first, second) == (1, 2)


async def test_run_pair_reraises_a_nested_leaf_with_its_original_cause() -> None:
    async def failing_with_cause() -> None:
        try:
            raise KeyError("root cause")
        except KeyError as root:
            raise _BoomError("leaf") from root

    async def nested_group() -> None:
        try:
            await failing_with_cause()
        except _BoomError as leaf:
            raise ExceptionGroup("nested", [leaf]) from None

    with pytest.raises(_BoomError) as raised:
        await run_pair(nested_group(), _never())

    assert str(raised.value) == "leaf"
    assert isinstance(raised.value.__cause__, KeyError)


async def test_run_pair_propagates_caller_cancellation() -> None:
    task = asyncio.create_task(run_pair(_never(), _never()))
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_run_all_cancels_siblings_when_one_branch_fails() -> None:
    cancelled = asyncio.Event()
    coroutines: list[Coroutine[Any, Any, object]] = [
        _recording_sibling(cancelled),
        _fail_after_a_tick("branch failed"),
    ]

    with pytest.raises(_BoomError, match="branch failed"):
        await run_all(coroutines)

    assert cancelled.is_set()


async def test_run_all_returns_results_in_input_order_not_completion_order() -> None:
    results = await run_all([_delayed(0, 0.03), _delayed(1, 0.0), _delayed(2, 0.01)])

    assert results == [0, 1, 2]


async def test_run_all_of_nothing_is_an_empty_list() -> None:
    results: list[int] = await run_all([])

    assert results == []


async def test_run_all_propagates_caller_cancellation() -> None:
    task = asyncio.create_task(run_all([_never(), _never()]))
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
