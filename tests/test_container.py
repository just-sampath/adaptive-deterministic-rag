"""Composition root: implementation names are validated and partial graphs are closed."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

import container
from container import create_container
from db.vector_index import build_vector_index_spec
from handlers.health import DependencyReadinessHandler
from handlers.search import SearchHandler
from handlers.traces import TenantTraceHandler
from middleware.request_telemetry import _TelemetryHost
from providers.sparse import Bm25SparseEncoder
from routes.dependencies import RouteContext
from services.followup import (
    AlwaysFollowupDetector,
    HeuristicFollowupDetector,
    ModelFollowupDetector,
)
from tests.fakes import FakeTextModel
from utils.errors import ConfigurationError
from utils.settings import (
    Environment,
    ExternalSettings,
    FollowupDetectorName,
    Settings,
    load_settings,
)

_PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "config"


class _CloseProbe:
    def __init__(self, name: str, closed: list[str]) -> None:
        self._name = name
        self._closed = closed

    async def close(self) -> None:
        self._closed.append(self._name)


class _FailingCloseProbe:
    async def close(self) -> None:
        raise RuntimeError("close failed")


class _HungCloseProbe:
    async def close(self) -> None:
        await asyncio.Event().wait()


def _shutdown_failures(caplog: pytest.LogCaptureFixture) -> int:
    return sum(1 for record in caplog.records if record.getMessage() == "Resource shutdown failed")


@pytest.fixture
def settings() -> Settings:
    external = ExternalSettings(
        _env_file=None,
        app_env=Environment.LOCAL,
        config_dir=_PROJECT_CONFIG,
        openai_api_key="sk-test",
        cohere_api_key="co-test",
    )
    return load_settings(external)


def _unsupported_ai_provider(settings: Settings) -> Settings:
    return settings.model_copy(update={"ai": settings.ai.model_copy(update={"provider": "bogus"})})


def _unsupported_detector(settings: Settings) -> Settings:
    contextualization = settings.contextualization.model_copy(update={"detector": "bogus"})
    return settings.model_copy(update={"contextualization": contextualization})


def _with_detector(settings: Settings, detector: FollowupDetectorName) -> Settings:
    contextualization = settings.contextualization.model_copy(update={"detector": detector})
    return settings.model_copy(update={"contextualization": contextualization})


def _unsupported_sparse_encoder(settings: Settings) -> Settings:
    sparse = settings.vector_store.sparse.model_copy(update={"encoder": "tfidf"})
    vector_store = settings.vector_store.model_copy(update={"sparse": sparse})
    return settings.model_copy(update={"vector_store": vector_store})


@pytest.mark.parametrize(
    "mutate",
    [_unsupported_ai_provider, _unsupported_detector, _unsupported_sparse_encoder],
)
async def test_unsupported_implementation_fails_before_any_client_is_built(
    settings: Settings,
    mutate: Callable[[Settings], Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    def unexpected_build(*_: object, **__: object) -> None:
        built.append("clients")
        raise AssertionError("no client may be built for an unsupported implementation")

    monkeypatch.setattr(container, "_build_clients", unexpected_build)

    with pytest.raises(ConfigurationError):
        await create_container(mutate(settings))
    assert built == []


async def test_construction_failure_closes_clients_already_built(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    def openai_probe(**_: object) -> _CloseProbe:
        return _CloseProbe("openai", closed)

    def failing_mongo(**_: object) -> object:
        raise RuntimeError("mongo construction failed")

    monkeypatch.setattr(container, "build_openai_client", openai_probe)
    monkeypatch.setattr(container, "build_mongo_client", failing_mongo)

    with pytest.raises(RuntimeError, match="mongo construction failed"):
        await create_container(settings)
    assert closed == ["openai"]


async def test_selected_vocabulary_must_load_before_clients_are_created(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = Settings.model_validate(
        {
            **settings.model_dump(),
            "context": {
                **settings.context.model_dump(),
                "tokenizer_encoding": "o200k_base",
                "tokenizer_verified_for": settings.ai.generation_model,
            },
        }
    )

    def missing_vocabulary() -> None:
        raise ConfigurationError("Vocabulary unavailable")

    def unexpected_client(*_: object, **__: object) -> None:
        pytest.fail("A client was created before the selected vocabulary was verified")

    monkeypatch.setattr(container, "load_o200k_counter", missing_vocabulary)
    monkeypatch.setattr(container, "_build_clients", unexpected_client)
    with pytest.raises(ConfigurationError, match="Vocabulary unavailable"):
        await create_container(configured)


async def test_built_container_wires_handlers_and_closes_its_resources(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_vocabulary() -> None:
        pytest.fail("Byte-mode startup tried to load a vocabulary")

    monkeypatch.setattr(container, "load_o200k_counter", unexpected_vocabulary)
    built = await create_container(settings)
    try:
        assert len(built.external_resources) == 4
        assert isinstance(built.query_handler, SearchHandler)
        assert isinstance(built.trace_handler, TenantTraceHandler)
        assert isinstance(built.readiness_handler, DependencyReadinessHandler)
    finally:
        await built.close()


async def test_built_container_satisfies_the_transport_ports(settings: Settings) -> None:
    built = await create_container(settings)
    try:
        # Annotated assignments are checked by mypy, so AppContainer drift fails the static run.
        route_context: RouteContext = built
        telemetry_host: _TelemetryHost = built
        assert route_context.telemetry is built.telemetry
        assert telemetry_host.telemetry is built.telemetry
    finally:
        await built.close()


@pytest.mark.parametrize(
    ("detector", "detector_class"),
    [
        (FollowupDetectorName.MODEL, ModelFollowupDetector),
        (FollowupDetectorName.HEURISTIC, HeuristicFollowupDetector),
        (FollowupDetectorName.ALWAYS, AlwaysFollowupDetector),
    ],
)
def test_configured_followup_detector_name_selects_its_implementation(
    settings: Settings,
    detector: FollowupDetectorName,
    detector_class: type[object],
) -> None:
    built = container._build_detector(_with_detector(settings, detector), FakeTextModel())

    assert isinstance(built, detector_class)
    assert built.name == detector.value


async def test_start_initializes_storage_before_starting_telemetry(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = await create_container(settings)
    order: list[str] = []

    async def ensure_ready() -> None:
        order.append("storage")

    async def start_telemetry() -> None:
        order.append("telemetry")

    monkeypatch.setattr(built.initializer, "ensure_ready", ensure_ready)
    monkeypatch.setattr(built.telemetry, "start", start_telemetry)
    try:
        await built.start()
    finally:
        await built.close()
    assert order == ["storage", "telemetry"]


async def test_close_releases_external_resources_when_telemetry_close_fails(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = await create_container(settings)
    closed: list[str] = []
    built.external_resources = (*built.external_resources, _CloseProbe("probe", closed))

    async def failing_telemetry_close() -> None:
        raise RuntimeError("telemetry flush failed")

    monkeypatch.setattr(built.telemetry, "close", failing_telemetry_close)

    with pytest.raises(RuntimeError, match="telemetry flush failed"):
        await built.close()
    assert closed == ["probe"]


async def test_close_logs_a_failing_resource_and_still_closes_the_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    closed: list[str] = []

    with caplog.at_level(logging.ERROR):
        await container._close_all((_FailingCloseProbe(), _CloseProbe("probe", closed)))

    assert closed == ["probe"]
    assert _shutdown_failures(caplog) == 1


async def test_hung_resource_close_is_abandoned_so_shutdown_cannot_hang(
    caplog: pytest.LogCaptureFixture,
) -> None:
    closed: list[str] = []

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(
            container._close_all(
                (_HungCloseProbe(), _CloseProbe("probe", closed)),
                timeout_seconds=0.05,
            ),
            timeout=5,
        )

    assert closed == ["probe"]
    assert _shutdown_failures(caplog) == 1


async def test_container_close_does_not_raise_when_a_resource_fails(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    built = await create_container(settings)
    closed: list[str] = []
    built.external_resources = (
        _FailingCloseProbe(),
        _CloseProbe("probe", closed),
        *built.external_resources,
    )

    with caplog.at_level(logging.ERROR):
        await built.close()

    assert closed == ["probe"]
    assert _shutdown_failures(caplog) == 1


async def test_construction_failure_logs_close_errors_and_reraises_the_original(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    closed: list[str] = []

    def failing_close_openai(**_: object) -> _FailingCloseProbe:
        return _FailingCloseProbe()

    def mongo_probe(**_: object) -> _CloseProbe:
        return _CloseProbe("mongo", closed)

    def failing_qdrant(**_: object) -> object:
        raise RuntimeError("qdrant construction failed")

    monkeypatch.setattr(container, "build_openai_client", failing_close_openai)
    monkeypatch.setattr(container, "build_mongo_client", mongo_probe)
    monkeypatch.setattr(container, "build_qdrant_client", failing_qdrant)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="qdrant construction"):
        await create_container(settings)

    assert closed == ["mongo"]
    assert _shutdown_failures(caplog) == 1


async def test_built_query_handler_records_encoder_index_and_detector_versions(
    settings: Settings,
) -> None:
    built = await create_container(settings)
    try:
        assert isinstance(built.query_handler, SearchHandler)
        versions = built.query_handler._runtime_versions
        encoder = (
            f"openai:{settings.ai.embedding_model}:{settings.ai.embedding_dimension}"
            f"|{Bm25SparseEncoder(settings.vector_store.sparse).fingerprint}"
        )
        index = build_vector_index_spec(
            settings.vector_store,
            encoder_fingerprint=encoder,
            dense_dimension=settings.ai.embedding_dimension,
            sparse_idf_modifier=True,
        )

        assert versions["query_encoder"] == encoder
        assert versions["vector_index"] == index.physical_collection
        assert versions["followup_detector"] == FollowupDetectorName.MODEL.value
    finally:
        await built.close()
