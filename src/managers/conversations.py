"""Conversation persistence in MongoDB."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError
from pymongo.asynchronous.database import AsyncDatabase

from managers.errors import mongo_errors
from models.domain import ConversationMessage, Identity, utc_now
from utils.errors import ProviderResponseError


class MongoConversationManager:
    def __init__(self, database: AsyncDatabase[dict[str, Any]]) -> None:
        self._conversations = database["conversations"]

    @mongo_errors
    async def get_messages(
        self,
        identity: Identity,
        conversation_id: str,
        limit: int,
    ) -> tuple[ConversationMessage, ...]:
        conversation = await self._conversations.find_one(
            {
                "tenant_id": identity.tenant_id,
                "user_id": identity.user_id,
                "id": conversation_id,
            },
            {"_id": 0, "messages": {"$slice": -limit}},
        )
        if not conversation:
            return ()
        messages = conversation.get("messages")
        if not isinstance(messages, list):
            raise ProviderResponseError("MongoDB returned an invalid conversation record")
        try:
            return tuple(ConversationMessage.model_validate(item) for item in messages)
        except ValidationError as exc:
            raise ProviderResponseError("MongoDB returned an invalid conversation record") from exc

    @mongo_errors
    async def append_messages(
        self,
        identity: Identity,
        conversation_id: str,
        messages: Sequence[ConversationMessage],
    ) -> None:
        if not messages:
            return
        now = utc_now()
        await self._conversations.update_one(
            {
                "tenant_id": identity.tenant_id,
                "user_id": identity.user_id,
                "id": conversation_id,
            },
            {
                "$setOnInsert": {
                    "tenant_id": identity.tenant_id,
                    "user_id": identity.user_id,
                    "id": conversation_id,
                    "created_at": now,
                },
                "$set": {"updated_at": now},
                "$push": {
                    "messages": {
                        "$each": [message.model_dump(mode="python") for message in messages],
                        "$slice": -100,
                    }
                },
            },
            upsert=True,
        )
