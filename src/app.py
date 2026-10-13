"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from container import AppContainer, create_container
from middleware.request_telemetry import RequestTelemetryMiddleware
from routes.exceptions import register_exception_handlers
from routes.index import build_router
from utils.logging import configure_logging
from utils.settings import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an ASGI app with lifespan-owned shared provider clients."""

    resolved = settings or load_settings()
    configure_logging(resolved.app.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container: AppContainer | None = None
        try:
            container = await create_container(resolved)
            app.state.container = container
            await container.start()
            yield
        finally:
            if container is not None:
                await container.close()

    app = FastAPI(
        title="Adaptive Deterministic RAG",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestTelemetryMiddleware)
    app.include_router(build_router())
    register_exception_handlers(app)
    return app
