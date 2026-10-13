"""Database readiness checks."""

from __future__ import annotations

from typing import Any

from pymongo import AsyncMongoClient

from managers.errors import mongo_errors


class MongoHealthManager:
    def __init__(self, client: AsyncMongoClient[dict[str, Any]]) -> None:
        self._client = client

    @mongo_errors
    async def health_check(self) -> None:
        await self._client.admin.command("ping")
