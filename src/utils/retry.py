"""Retry a transient provider failure inside the time the caller has left."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping

from utils.errors import TransientProviderError
from utils.request_context import current_deadline, current_request_id
from utils.settings import RetryConfig

logger = logging.getLogger(__name__)

# Services built without the container's policy make one try, as before retries existed.
NO_RETRY = RetryConfig(max_retries=0)


async def retry_transient[T](
    stage: str,
    call: Callable[[int], Awaitable[T]],
    *,
    policy: RetryConfig,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> T:
    """Await ``call(retry_ordinal)`` and retry a TransientProviderError while time allows.

    The first try has ordinal 0. Timeouts, invalid responses, cancellation and every other failure
    propagate at once: a slow provider retried inside the same budget would only time out again.
    """

    retries = 0
    while True:
        try:
            return await call(retries)
        except TransientProviderError as error:
            wait = _retry_wait(error, policy, retries)
            if wait is None:
                raise
            retries += 1
            logger.warning(
                "Retrying a transient %s failure",
                stage,
                extra={"request_id": current_request_id(), "error_code": type(error).__name__},
            )
            await sleep(wait)


def _retry_wait(error: TransientProviderError, policy: RetryConfig, retries: int) -> float | None:
    """Return how long to wait before the next try, or None when no retry fits."""

    if retries >= policy.max_retries:
        return None
    asked = error.retry_after_seconds
    wait = policy.backoff_seconds if asked is None else asked
    deadline = current_deadline()
    if deadline is not None:
        remaining = deadline - asyncio.get_running_loop().time()
        # A wait longer than the time left would only end in the stage's own timeout.
        if wait + policy.min_retry_seconds > remaining:
            return None
    return wait


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Read the wait a response asks for from ``retry-after-ms`` or ``retry-after``.

    ``retry-after`` may hold seconds or an HTTP date. A missing, negative or unreadable value gives
    None, and the caller falls back to its own backoff.
    """

    milliseconds = _seconds(headers.get("retry-after-ms"))
    if milliseconds is not None:
        return milliseconds / 1000
    value = headers.get("retry-after")
    seconds = _seconds(value)
    if seconds is not None or value is None:
        return seconds
    parsed = email.utils.parsedate_tz(value)
    if parsed is None:
        return None
    try:
        when = email.utils.mktime_tz(parsed)
    except (OverflowError, ValueError, OSError):
        return None
    return max(0.0, when - time.time())


def _seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number >= 0 else None
