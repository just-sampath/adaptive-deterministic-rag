"""Application route registration."""

from __future__ import annotations

from fastapi import APIRouter

from routes.health.index import router as health_router
from routes.search.index import router as search_router
from routes.traces.index import router as traces_router


def build_router() -> APIRouter:
    router = APIRouter()
    router.include_router(health_router)
    router.include_router(search_router)
    router.include_router(traces_router)
    return router
