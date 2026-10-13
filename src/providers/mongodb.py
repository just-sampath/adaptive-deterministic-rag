"""MongoDB client construction."""

from __future__ import annotations

from datetime import UTC
from typing import Any

from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi


def build_mongo_client(
    *,
    uri: str,
    server_selection_timeout_ms: int,
    operation_timeout_ms: int,
) -> AsyncMongoClient[dict[str, Any]]:
    """Create one stable-API asynchronous MongoDB client for the process."""

    return AsyncMongoClient(
        uri,
        server_api=ServerApi("1", strict=True, deprecation_errors=True),
        serverSelectionTimeoutMS=server_selection_timeout_ms,
        timeoutMS=operation_timeout_ms,
        uuidRepresentation="standard",
        # Stored datetimes are UTC; read them back timezone-aware so traces keep their offset.
        tz_aware=True,
        tzinfo=UTC,
    )
