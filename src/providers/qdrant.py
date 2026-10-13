"""Qdrant client construction."""

from __future__ import annotations

import math

from qdrant_client import AsyncQdrantClient


def build_qdrant_client(
    *, url: str, api_key: str | None, timeout_seconds: float
) -> AsyncQdrantClient:
    """Create one asynchronous Qdrant client for the process."""

    return AsyncQdrantClient(url=url, api_key=api_key, timeout=max(1, math.ceil(timeout_seconds)))
