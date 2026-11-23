"""ASGI entrypoint and local process runner."""

from __future__ import annotations

import uvicorn

from app import create_app
from utils.settings import load_settings

settings = load_settings()
app = create_app(settings)


def main() -> None:
    uvicorn.run(
        app,
        host=settings.app.host,
        port=settings.app.port,
        log_level=settings.app.log_level.lower(),
        # create_app already configured JSON logging; uvicorn's default config would replace it.
        log_config=None,
    )


if __name__ == "__main__":
    main()
