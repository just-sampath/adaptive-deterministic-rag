"""Readiness use case that runs every dependency health check concurrently."""

from __future__ import annotations

from collections.abc import Sequence

from services.contracts import HealthCheck
from utils.concurrency import run_all


class DependencyReadinessHandler:
    def __init__(self, checks: Sequence[HealthCheck]) -> None:
        self._checks = tuple(checks)

    async def check(self) -> None:
        await run_all(check.health_check() for check in self._checks)
