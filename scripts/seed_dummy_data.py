"""Seed one fixed local-development document into MongoDB and Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from db.initializer import DatabaseInitializer
from db.vector_index import build_vector_index_spec
from managers.vector import to_qdrant_sparse
from models.domain import Chunk, Identity
from providers.mongodb import build_mongo_client
from providers.openai import OpenAIEmbeddingProvider, build_openai_client
from providers.qdrant import build_qdrant_client
from providers.sparse import Bm25SparseEncoder
from services.encoding import EmbeddingCache, HybridQueryEncoder
from utils.errors import ConfigurationError
from utils.settings import Environment, Settings, load_settings

_TITLE = "Adaptive RAG Local Handbook"
logger = logging.getLogger(__name__)
_TEXTS = (
    (
        ("Deterministic search",),
        "The backend tries a fixed, bounded sequence of retrieval strategies and abstains "
        "when evidence is insufficient.",
    ),
    (
        ("Observability",),
        "Request telemetry and detailed RAG traces are stored in MongoDB. Provider usage "
        "and request identifiers are retained when returned by the provider.",
    ),
)


async def seed_dummy_data(settings: Settings, identity: Identity) -> dict[str, str | int]:
    """Write the fixed dummy document; this is the repository's only vector writer."""

    if settings.external.app_env is not Environment.LOCAL:
        raise ConfigurationError("Dummy data can only be seeded in APP_ENV=local")
    if settings.ai.provider != "openai":
        raise ConfigurationError("The dummy seed script currently requires the OpenAI provider")
    assert settings.external.openai_api_key is not None

    document_id = str(uuid5(NAMESPACE_URL, f"adaptive-rag:{identity.tenant_id}:dummy"))
    version_id = str(uuid5(NAMESPACE_URL, f"{document_id}:version:1"))
    chunks = _chunks(identity.tenant_id, document_id, version_id)
    mongo_client = build_mongo_client(
        uri=settings.external.mongodb_uri.get_secret_value(),
        server_selection_timeout_ms=settings.mongodb.server_selection_timeout_ms,
        operation_timeout_ms=settings.mongodb.operation_timeout_ms,
    )
    qdrant_client = build_qdrant_client(
        url=settings.external.qdrant_url,
        api_key=(
            settings.external.qdrant_api_key.get_secret_value()
            if settings.external.qdrant_api_key
            else None
        ),
        timeout_seconds=settings.timeouts.retrieval_seconds,
    )
    ai_client = build_openai_client(
        api_key=settings.external.openai_api_key.get_secret_value(),
        base_url=settings.external.openai_base_url,
        max_retries=settings.ai.max_retries,
        timeout_seconds=settings.timeouts.generation_seconds,
    )
    try:
        embeddings = OpenAIEmbeddingProvider(
            ai_client,
            model=settings.ai.embedding_model,
            dimension=settings.ai.embedding_dimension,
        )
        sparse = Bm25SparseEncoder(settings.vector_store.sparse)
        encoder = HybridQueryEncoder(
            dense=embeddings,
            sparse=sparse,
            cache=EmbeddingCache(0),
            embedding_timeout_seconds=settings.timeouts.embedding_seconds,
        )
        index = build_vector_index_spec(
            settings.vector_store,
            encoder_fingerprint=encoder.fingerprint,
            dense_dimension=encoder.dense_dimension,
            sparse_idf_modifier=encoder.requires_idf_modifier,
        )
        await DatabaseInitializer(
            mongo_client,
            qdrant_client,
            mongo=settings.mongodb,
            index=index,
        ).ensure_ready()
        texts = [chunk.text for chunk in chunks]
        dense_results = await embeddings.embed_documents(texts)
        sparse_vectors = sparse.encode_documents(texts)
        now = datetime.now(UTC)
        database = mongo_client[settings.mongodb.database]
        await database["documents"].update_one(
            {"tenant_id": identity.tenant_id, "id": document_id},
            {
                "$set": {
                    "title": _TITLE,
                    "source_type": "dummy",
                    "source_uri": None,
                    "metadata": {"seed": "local"},
                    "current_version_id": version_id,
                    "last_version_number": 1,
                    "deleted_at": None,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        await database["document_versions"].replace_one(
            {"tenant_id": identity.tenant_id, "id": version_id},
            {
                "id": version_id,
                "document_id": document_id,
                "tenant_id": identity.tenant_id,
                "version": 1,
                "checksum": "local-dummy-v1",
                "active": True,
                "created_at": now,
            },
            upsert=True,
        )
        for chunk in chunks:
            item: dict[str, Any] = chunk.model_dump(mode="python")
            item["allowed_roles"] = sorted(chunk.allowed_roles)
            item["heading_path"] = list(chunk.heading_path)
            item["created_at"] = now
            await database["chunks"].replace_one(
                {"tenant_id": identity.tenant_id, "id": chunk.id}, item, upsert=True
            )
        await qdrant_client.upsert(
            collection_name=index.physical_collection,
            points=[
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        index.dense_vector_name: dense.vector,
                        index.sparse_vector_name: to_qdrant_sparse(sparse_vector),
                    },
                    payload={
                        "tenant_id": chunk.tenant_id,
                        "allowed_roles": sorted(chunk.allowed_roles),
                        "document_id": chunk.document_id,
                        "document_version_id": chunk.document_version_id,
                        "section_id": chunk.section_id,
                        "metadata": chunk.metadata,
                        "active": True,
                    },
                )
                for chunk, dense, sparse_vector in zip(
                    chunks, dense_results, sparse_vectors, strict=True
                )
            ],
            wait=True,
        )
        await database["system_state"].update_one(
            {"tenant_id": identity.tenant_id},
            {
                "$setOnInsert": {"tenant_id": identity.tenant_id},
                "$inc": {"corpus_version": 1},
                "$set": {"updated_at": now},
            },
            upsert=True,
        )
        return {
            "document_id": document_id,
            "document_version_id": version_id,
            "chunks_indexed": len(chunks),
        }
    finally:
        close_results = await asyncio.gather(
            ai_client.close(),
            qdrant_client.close(),
            mongo_client.close(),
            return_exceptions=True,
        )
        for result in close_results:
            if isinstance(result, BaseException):
                logger.error(
                    "Dummy seed resource shutdown failed",
                    exc_info=(type(result), result, result.__traceback__),
                )


def _chunks(tenant_id: str, document_id: str, version_id: str) -> tuple[Chunk, ...]:
    allowed_roles = frozenset({"reader", "administrator"})
    chunk_ids = [
        str(uuid5(NAMESPACE_URL, f"{version_id}:chunk:{sequence}"))
        for sequence in range(len(_TEXTS))
    ]
    chunks: list[Chunk] = []
    for sequence, (heading, text) in enumerate(_TEXTS):
        section_id = str(uuid5(NAMESPACE_URL, f"{version_id}:section:{heading[0]}"))
        content = f"Document: {_TITLE}\nSection: {heading[0]}\nVersion: 1\n\n{text}"
        chunks.append(
            Chunk(
                id=chunk_ids[sequence],
                tenant_id=tenant_id,
                document_id=document_id,
                document_version_id=version_id,
                document_version=1,
                document_title=_TITLE,
                section_id=section_id,
                sequence=sequence,
                heading_path=heading,
                text=content,
                token_count=max(1, (len(content) + 3) // 4),
                previous_chunk_id=chunk_ids[sequence - 1] if sequence else None,
                next_chunk_id=(chunk_ids[sequence + 1] if sequence + 1 < len(chunk_ids) else None),
                allowed_roles=allowed_roles,
                metadata={"seed": "local"},
            )
        )
    return tuple(chunks)


async def _main(tenant_id: str, user_id: str) -> None:
    settings = load_settings()
    result = await seed_dummy_data(
        settings,
        Identity(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=frozenset({"administrator"}),
        ),
    )
    print(result)  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", default="local-tenant")
    parser.add_argument("--user-id", default="local-admin")
    args = parser.parse_args()
    asyncio.run(_main(args.tenant_id, args.user_id))


if __name__ == "__main__":
    main()
