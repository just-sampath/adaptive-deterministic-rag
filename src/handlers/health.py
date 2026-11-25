"""Readiness use case that runs every dependency health check concurrently."""

from __future__ import annotations

from collections.abc import Sequence

from services.contracts import HealthCheck
from utils.concurrency import run_all
from utils.errors import InfrastructureError


class DependencyReadinessHandler:
    def __init__(self, checks: Sequence[HealthCheck]) -> None:
        self._checks = tuple(checks)

    async def check(self) -> None:
        try:
            await run_all(check.health_check() for check in self._checks)
        except TimeoutError as exc:
            # A dependency that cannot answer in time is not ready; probes expect 503, not 504.
            raise InfrastructureError("A dependency did not answer in time") from exc
