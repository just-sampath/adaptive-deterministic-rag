from __future__ import annotations

from collections.abc import Callable

import pytest

from models.domain import Identity
from utils.settings import Environment, ExternalSettings, Settings


@pytest.fixture
def identity() -> Identity:
    return Identity(tenant_id="tenant-a", user_id="user-a", roles=frozenset({"reader"}))


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    def factory(**values: object) -> Settings:
        values.setdefault(
            "external",
            ExternalSettings(
                _env_file=None,
                app_env=Environment.TEST,
                auth_shared_token="test-auth-token",
                debug_token="test-debug-token",
            ),
        )
        values.setdefault("ai", {"provider": "fake"})
        values.setdefault("reranker", {"provider": "fake"})
        return Settings.model_validate(values)

    return factory
