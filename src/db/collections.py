"""MongoDB collection names owned by this backend."""

DOCUMENTS = "documents"
DOCUMENT_VERSIONS = "document_versions"
CHUNKS = "chunks"
CONVERSATIONS = "conversations"
SYSTEM_STATE = "system_state"


def all_collections(*, trace_collection: str, telemetry_collection: str) -> tuple[str, ...]:
    return (
        DOCUMENTS,
        DOCUMENT_VERSIONS,
        CHUNKS,
        CONVERSATIONS,
        SYSTEM_STATE,
        trace_collection,
        telemetry_collection,
    )
