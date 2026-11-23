"""Idempotent database and vector-index startup checks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from pymongo import AsyncMongoClient, IndexModel
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid, PyMongoError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from db.collections import all_collections
from db.indexes import mongo_indexes
from db.vector_index import VectorIndexSpec
from utils.concurrency import run_pair
from utils.errors import InfrastructureError, ProviderError, ProviderResponseError
from utils.settings import MongoConfig

logger = logging.getLogger(__name__)
_KEYWORD_FIELDS = ("tenant_id", "allowed_roles", "document_id", "document_version_id", "section_id")


class DatabaseInitializer:
    """Create missing Mongo collections and the Qdrant physical index behind a stable alias."""

    def __init__(
        self,
        mongo_client: AsyncMongoClient[dict[str, Any]],
        qdrant_client: AsyncQdrantClient,
        *,
        mongo: MongoConfig,
        index: VectorIndexSpec,
    ) -> None:
        self._mongo_client = mongo_client
        self._qdrant_client = qdrant_client
        self._mongo = mongo
        self._index = index

    async def ensure_ready(self) -> None:
        """Initialize independent stores concurrently; a failure cancels the sibling."""

        await run_pair(self._ensure_mongo(), self._ensure_qdrant())

    async def _ensure_mongo(self) -> None:
        database = self._mongo_client[self._mongo.database]
        try:
            await self._mongo_client.admin.command("ping")
            for name in all_collections(
                trace_collection=self._mongo.trace_collection,
                telemetry_collection=self._mongo.telemetry_collection,
            ):
                with suppress(CollectionInvalid):
                    await database.create_collection(name)
            for name, indexes in mongo_indexes(self._mongo).items():
                await _update_changed_ttls(database, name, indexes)
                await database[name].create_indexes(list(indexes))
        except PyMongoError as exc:
            raise InfrastructureError("MongoDB initialization failed") from exc

    async def _ensure_qdrant(self) -> None:
        try:
            attached = await self._check_alias()
            if not await self._qdrant_client.collection_exists(self._index.physical_collection):
                await self._create_physical_collection()
            info = await self._qdrant_client.get_collection(self._index.physical_collection)
            self._validate_schema(info)
            await self._ensure_payload_indexes()
            if not attached:
                await self._attach_alias()
        except ProviderResponseError:
            raise
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise ProviderError("Qdrant initialization failed") from exc

    async def _check_alias(self) -> bool:
        """Return whether the alias already targets the index; reject any other alias state."""

        alias = self._index.alias
        physical = self._index.physical_collection
        target = await self._alias_target()
        if target is None:
            if await self._qdrant_client.collection_exists(alias):
                raise ProviderResponseError(
                    f"Qdrant collection {alias} uses the legacy layout; migrate its points into "
                    f"{physical}, then create the alias"
                )
            return False
        if target != physical:
            raise ProviderResponseError(
                f"Qdrant alias {alias} points to {target}, but the configured encoder and vector "
                f"layout expect {physical}; reindex into {physical}, then switch the alias"
            )
        return True

    async def _alias_target(self) -> str | None:
        response = await self._qdrant_client.get_aliases()
        return next(
            (
                item.collection_name
                for item in response.aliases
                if item.alias_name == self._index.alias
            ),
            None,
        )

    async def _create_physical_collection(self) -> None:
        index = self._index
        sparse_modifier = models.Modifier.IDF if index.sparse_idf_modifier else None
        try:
            await self._qdrant_client.create_collection(
                collection_name=index.physical_collection,
                vectors_config={
                    index.dense_vector_name: models.VectorParams(
                        size=index.dense_dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    index.sparse_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                        modifier=sparse_modifier,
                    )
                },
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise

    def _validate_schema(self, info: models.CollectionInfo) -> None:
        index = self._index
        params = info.config.params
        vectors = params.vectors
        dense = vectors.get(index.dense_vector_name) if isinstance(vectors, dict) else None
        sparse = (
            params.sparse_vectors.get(index.sparse_vector_name) if params.sparse_vectors else None
        )
        valid = (
            dense is not None
            and dense.size == index.dense_dimension
            and dense.distance is models.Distance.COSINE
            and sparse is not None
            and (sparse.modifier is models.Modifier.IDF) == index.sparse_idf_modifier
        )
        if not valid:
            raise ProviderResponseError(
                "Existing Qdrant collection does not match the configured vector schema"
            )

    async def _ensure_payload_indexes(self) -> None:
        physical = self._index.physical_collection
        for field in _KEYWORD_FIELDS:
            await self._qdrant_client.create_payload_index(
                collection_name=physical,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        await self._qdrant_client.create_payload_index(
            collection_name=physical,
            field_name="active",
            field_schema=models.PayloadSchemaType.BOOL,
            wait=True,
        )

    async def _attach_alias(self) -> None:
        physical = self._index.physical_collection
        operation = models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=physical, alias_name=self._index.alias)
        )
        try:
            await self._qdrant_client.update_collection_aliases([operation])
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise
            # A concurrent startup may have created the same alias; accept only that target.
            if await self._alias_target() != physical:
                raise ProviderResponseError(
                    f"Qdrant alias {self._index.alias} was created concurrently for another "
                    "collection"
                ) from exc


async def _update_changed_ttls(
    database: AsyncDatabase[dict[str, Any]],
    collection: str,
    indexes: Sequence[IndexModel],
) -> None:
    """Apply a longer retention to an existing TTL index in place.

    ``createIndexes`` rejects an existing index name with different options, so a TTL whose key
    is unchanged gets its new ``expireAfterSeconds`` through ``collMod`` first. Only increases
    are applied: a shorter retention deletes stored records, and while instances with old and new
    settings run side by side, applying whichever value started last would flip it back and forth.
    A decrease stops startup and must be applied by hand. Any other difference is left for
    ``createIndexes`` to reject.
    """

    ttls = [index.document for index in indexes if "expireAfterSeconds" in index.document]
    if not ttls:
        return
    cursor = await database[collection].list_indexes()
    existing = {item["name"]: item async for item in cursor}
    for wanted in ttls:
        current = existing.get(wanted["name"])
        if (
            current is None
            or "expireAfterSeconds" not in current
            or list(current["key"].items()) != list(wanted["key"].items())
            or current["expireAfterSeconds"] == wanted["expireAfterSeconds"]
        ):
            continue
        name = wanted["name"]
        old, new = current["expireAfterSeconds"], wanted["expireAfterSeconds"]
        if new < old:
            raise InfrastructureError(
                f"TTL index {name} on {collection} keeps records for {old} seconds, but the "
                f"settings ask for {new}. Shortening retention deletes stored records, so "
                f"startup won't apply it; run collMod on {collection} to lower it, or keep the "
                f"current value"
            )
        await database.command(
            "collMod",
            collection,
            index={"name": name, "expireAfterSeconds": new},
        )
        logger.warning(
            "Raised TTL index %s on %s from %s to %s seconds", name, collection, old, new
        )
