"""Conversation persistence in MongoDB."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bson import BSON
from pydantic import ValidationError
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.common import MAX_BSON_SIZE
from pymongo.errors import DocumentTooLarge, WriteError

from managers.errors import mongo_errors
from models.domain import ConversationMessage, EvidenceRef, Identity, utc_now
from utils.errors import InfrastructureError, ProviderResponseError

# The server's update and upsert stages raise 17419 and 17420 when the resulting document is too
# large; 10334 is BSONObjectTooLarge. A write error means this single update was not applied.
_DOCUMENT_SIZE_ERRORS = frozenset({10334, 17419, 17420})


class MongoConversationManager:
    def __init__(
        self,
        database: AsyncDatabase[dict[str, Any]],
        *,
        max_history_refs: int,
        write_provenance: bool = False,
    ) -> None:
        self._conversations = database["conversations"]
        self._max_history_refs = max_history_refs
        self._write_provenance = write_provenance

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
        refs_seen: dict[EvidenceRef, EvidenceRef] = {}
        newest_first: list[ConversationMessage] = []
        try:
            for item in reversed(messages):
                if isinstance(item, dict) and item.get("evidence_refs") is not None:
                    # Non-null provenance on a user message is an unsafe pair boundary.
                    refs = (
                        self._read_refs(item["evidence_refs"], refs_seen)
                        if item.get("role") == "assistant"
                        else ()
                    )
                    item = {**item, "evidence_refs": refs}
                newest_first.append(ConversationMessage.model_validate(item))
        except ValidationError:
            raise ProviderResponseError("MongoDB returned an invalid conversation record") from None
        return tuple(reversed(newest_first))

    def _read_refs(
        self, raw_refs: object, refs_seen: dict[EvidenceRef, EvidenceRef]
    ) -> tuple[EvidenceRef, ...] | None:
        if not isinstance(raw_refs, list):
            return None
        distinct: dict[EvidenceRef, None] = {}
        for raw in raw_refs:
            if not isinstance(raw, dict) or any(type(value) is not str for value in raw.values()):
                return None
            try:
                ref = EvidenceRef.model_validate(raw, strict=True)
            except ValidationError:
                return None
            if ref not in refs_seen and len(refs_seen) >= self._max_history_refs:
                return None
            distinct[refs_seen.setdefault(ref, ref)] = None
        return tuple(distinct)

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
        key = {"tenant_id": identity.tenant_id, "user_id": identity.user_id, "id": conversation_id}
        rows = [message.model_dump(exclude={"evidence_refs"}) for message in messages]
        update = {
            "$setOnInsert": {**key, "created_at": now},
            "$set": {"updated_at": now},
            "$push": {"messages": {"$each": rows, "$slice": -100}},
        }
        if self._write_provenance:
            for message, row in zip(messages, rows, strict=True):
                if message.role == "assistant" and message.evidence_refs is not None:
                    distinct = dict.fromkeys(message.evidence_refs)
                    if len(distinct) <= self._max_history_refs:
                        row["evidence_refs"] = [ref.model_dump() for ref in distinct]
        if any("evidence_refs" in row for row in rows):
            operation = {"q": key, "u": update, "multi": False, "upsert": True}
            if len(BSON.encode(operation)) <= MAX_BSON_SIZE:
                try:
                    await self._conversations.update_one(key, update, upsert=True)
                    return
                except DocumentTooLarge:
                    pass  # The driver rejected the encoded operation before sending it.
                except WriteError as exc:
                    # The retained document may be full even when the command fits.
                    if exc.code not in _DOCUMENT_SIZE_ERRORS:
                        raise
            for row in rows:
                row.pop("evidence_refs", None)
        try:
            await self._conversations.update_one(key, update, upsert=True)
        except DocumentTooLarge as exc:
            raise InfrastructureError("MongoDB operation failed") from exc
