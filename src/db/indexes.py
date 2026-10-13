"""Central MongoDB index definitions."""

from __future__ import annotations

from pymongo import ASCENDING, IndexModel

from db.collections import CHUNKS, CONVERSATIONS, DOCUMENT_VERSIONS, DOCUMENTS, SYSTEM_STATE
from utils.settings import MongoConfig


def mongo_indexes(config: MongoConfig) -> dict[str, tuple[IndexModel, ...]]:
    """Return all indexes required for startup idempotence and query safety."""

    return {
        DOCUMENTS: (
            IndexModel(
                [("tenant_id", ASCENDING), ("id", ASCENDING)],
                name="uq_documents_tenant_id",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("current_version_id", ASCENDING)],
                name="ix_documents_current_version",
            ),
        ),
        DOCUMENT_VERSIONS: (
            IndexModel(
                [("tenant_id", ASCENDING), ("id", ASCENDING)],
                name="uq_versions_tenant_id",
                unique=True,
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("document_id", ASCENDING),
                    ("version", ASCENDING),
                ],
                name="uq_versions_document_number",
                unique=True,
            ),
        ),
        CHUNKS: (
            IndexModel(
                [("tenant_id", ASCENDING), ("id", ASCENDING)],
                name="uq_chunks_tenant_id",
                unique=True,
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("document_version_id", ASCENDING),
                    ("sequence", ASCENDING),
                ],
                name="ix_chunks_version_sequence",
            ),
        ),
        CONVERSATIONS: (
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("id", ASCENDING),
                ],
                name="uq_conversations_identity_id",
                unique=True,
            ),
        ),
        SYSTEM_STATE: (
            IndexModel(
                [("tenant_id", ASCENDING)],
                name="uq_system_state_tenant",
                unique=True,
            ),
        ),
        config.trace_collection: (
            IndexModel(
                [("tenant_id", ASCENDING), ("request_id", ASCENDING)],
                name="uq_traces_tenant_request",
                unique=True,
            ),
            IndexModel(
                [("created_at", ASCENDING)],
                name="ttl_traces_created_at",
                expireAfterSeconds=config.trace_retention_days * 86_400,
            ),
        ),
        config.telemetry_collection: (
            IndexModel(
                [("request_id", ASCENDING), ("created_at", ASCENDING)],
                name="ix_telemetry_request_created",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("created_at", ASCENDING)],
                name="ix_telemetry_tenant_created",
            ),
            IndexModel(
                [("created_at", ASCENDING)],
                name="ttl_telemetry_created_at",
                expireAfterSeconds=config.telemetry_retention_days * 86_400,
            ),
        ),
    }
