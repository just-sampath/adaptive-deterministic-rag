"""Authoritative MongoDB reads for RAG documents and chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from pymongo.asynchronous.database import AsyncDatabase

from managers.errors import mongo_errors
from models.domain import Chunk, Identity
from utils.errors import ProviderResponseError

_CHUNK_PROJECTION: dict[str, int] = {"_id": 0, **dict.fromkeys(Chunk.model_fields, 1)}


class _CurrentDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    current_version_id: str


class MongoDocumentManager:
    """Hydrate vector hits through current-version and ACL checks in MongoDB."""

    def __init__(self, database: AsyncDatabase[dict[str, Any]]) -> None:
        self._documents = database["documents"]
        self._chunks = database["chunks"]

    @mongo_errors
    async def hydrate_authorized_chunks(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> dict[str, Chunk]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        raw_chunks = [
            item
            async for item in self._chunks.find(
                {"tenant_id": identity.tenant_id, "id": {"$in": unique_ids}},
                _CHUNK_PROJECTION,
            )
        ]
        if not raw_chunks:
            return {}
        try:
            parsed_chunks = [Chunk.model_validate(item) for item in raw_chunks]
        except ValidationError:
            # Validation errors can include the rejected record in a formatted traceback.
            raise ProviderResponseError("MongoDB returned an invalid chunk record") from None
        document_ids = list({chunk.document_id for chunk in parsed_chunks})
        raw_documents = [
            item
            async for item in self._documents.find(
                {
                    "tenant_id": identity.tenant_id,
                    "id": {"$in": document_ids},
                    "deleted_at": None,
                    "current_version_id": {"$ne": None},
                },
                {"_id": 0, "id": 1, "current_version_id": 1},
            )
        ]
        try:
            current_documents = [_CurrentDocument.model_validate(item) for item in raw_documents]
        except ValidationError as exc:
            raise ProviderResponseError("MongoDB returned an invalid document record") from exc
        current_versions = {item.id: item.current_version_id for item in current_documents}
        return {
            chunk.id: chunk
            for chunk in parsed_chunks
            if _is_current_and_authorized(chunk, identity, current_versions)
        }

    async def get_neighbors(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> dict[str, Chunk]:
        return await self.hydrate_authorized_chunks(chunk_ids, identity)


def _is_current_and_authorized(
    chunk: Chunk,
    identity: Identity,
    current_versions: dict[str, str],
) -> bool:
    return (
        chunk.tenant_id == identity.tenant_id
        and current_versions.get(chunk.document_id) == chunk.document_version_id
        and not chunk.allowed_roles.isdisjoint(identity.roles)
    )
