from __future__ import annotations

from routes.dependencies import Context


async def live() -> dict[str, str]:
    return {"status": "alive"}


async def ready(context: Context) -> dict[str, str]:
    await context.readiness_handler.check()
    return {"status": "ready"}
