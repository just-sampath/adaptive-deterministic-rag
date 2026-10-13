from __future__ import annotations

from fastapi import FastAPI

from routes.index import build_router


def test_runtime_routes_exclude_document_mutations_and_dev_seed() -> None:
    app = FastAPI()
    app.include_router(build_router())

    paths = set(app.openapi()["paths"])

    expected = {
        "/health/live",
        "/health/ready",
        "/v1/rag/query",
        "/v1/rag/requests/{request_id}/trace",
    }
    assert expected <= paths
    assert "/v1/documents" not in paths
    assert "/internal/dev/seed" not in paths
