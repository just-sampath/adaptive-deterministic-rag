from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from handlers.contracts import (
    QueryHandler,
    ReadinessHandler,
    TelemetryValue,
    TraceQueryHandler,
)
from handlers.health import DependencyReadinessHandler
from handlers.traces import TenantTraceHandler
from middleware.request_telemetry import RequestTelemetryMiddleware
from models.domain import (
    GenerationDraft,
    Identity,
    ModelUsage,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    RequestTrace,
    ResultStatus,
)
from routes.exceptions import register_exception_handlers
from routes.index import build_router
from tests.fakes import FakeRepository
from utils.errors import InfrastructureError, NotFoundError, RequestDeadlineExceededError
from utils.settings import Settings

REQUEST_ID = "5b1d3c6e-2f4a-4e8b-9c7d-0a1b2c3d4e5f"
MISSING_REQUEST_ID = "00000000-0000-4000-8000-000000000000"
TRUSTED_HEADERS = {
    "X-Auth-Token": "test-auth-token",
    "X-Tenant-ID": "tenant-a",
    "X-User-ID": "user-a",
    "X-Roles": "reader",
}
DEBUG_HEADERS = {**TRUSTED_HEADERS, "X-Debug-Token": "test-debug-token"}


@dataclass(frozen=True, slots=True)
class EmittedEvent:
    event_type: str
    request_id: str | None
    tenant_id: str | None
    user_id: str | None
    attributes: Mapping[str, TelemetryValue]


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[EmittedEvent] = []

    def emit(
        self,
        event_type: str,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        attributes: Mapping[str, TelemetryValue] | None = None,
    ) -> None:
        self.events.append(
            EmittedEvent(event_type, request_id, tenant_id, user_id, dict(attributes or {}))
        )

    def named(self, event_type: str) -> list[EmittedEvent]:
        return [event for event in self.events if event.event_type == event_type]


class FakeQueryHandler:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.identities: list[Identity] = []

    async def query(self, identity: Identity, request: QueryRequest) -> QueryResponse:
        del request
        self.identities.append(identity)
        if self._error is not None:
            raise self._error
        return QueryResponse(
            request_id=REQUEST_ID,
            status=ResultStatus.INSUFFICIENT_EVIDENCE,
            answer="The corpus has no supported evidence for this question.",
            sources=(),
            metadata=QueryMetadata(
                strategy=None, generation_repaired=False, attempts=0, latency_ms=1
            ),
        )


class FakeReadiness:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def check(self) -> None:
        if self._error is not None:
            raise self._error


@dataclass(slots=True)
class FakeContext:
    settings: Settings
    query_handler: QueryHandler
    trace_handler: TraceQueryHandler
    readiness_handler: ReadinessHandler
    telemetry: RecordingTelemetry


def make_context(
    settings: Settings,
    *,
    query: QueryHandler | None = None,
    readiness: ReadinessHandler | None = None,
    repository: FakeRepository | None = None,
) -> FakeContext:
    reader = repository if repository is not None else FakeRepository()
    return FakeContext(
        settings=settings,
        query_handler=query if query is not None else FakeQueryHandler(),
        trace_handler=TenantTraceHandler(reader=reader, admin_role=settings.auth.admin_role),
        readiness_handler=readiness if readiness is not None else FakeReadiness(),
        telemetry=RecordingTelemetry(),
    )


def make_app(context: FakeContext) -> FastAPI:
    """Wire the transport layers as app.create_app does, without the lifespan."""

    app = FastAPI()
    app.state.container = context
    app.add_middleware(RequestTelemetryMiddleware)
    app.include_router(build_router())
    register_exception_handlers(app)
    return app


def create_lifespan_app(settings: Settings) -> FastAPI:
    # Deferred: the composition root imports every provider module, which the HTTP tests avoid.
    from app import create_app

    return create_app(settings)


def make_trace(
    request_id: str,
    *,
    tenant_id: str = "tenant-a",
    user_id: str = "user-a",
) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        tenant_id=tenant_id,
        user_id=user_id,
        literal_question_observation={},
        canonical_question_observation={},
        attempted_strategies=(),
        generation_repair_used=False,
        attempts=(),
        final_status=None,
        final_source_ids=(),
        latency_ms=1,
        token_usage=ModelUsage(),
        prompt_versions={},
        runtime_versions={},
        configuration_fingerprint="test",
    )


def make_identity(
    *,
    user_id: str = "user-a",
    roles: frozenset[str] = frozenset({"reader"}),
) -> Identity:
    return Identity(tenant_id="tenant-a", user_id=user_id, roles=roles)


def test_search_passes_trusted_identity_and_records_completion(
    make_settings: Callable[..., Settings],
) -> None:
    query = FakeQueryHandler()
    context = make_context(make_settings(), query=query)

    response = TestClient(make_app(context)).post(
        "/v1/rag/query",
        json={"message": "question"},
        headers=TRUSTED_HEADERS,
    )

    assert response.status_code == 200
    assert query.identities == [make_identity()]
    completed = context.telemetry.named("http.request.completed")
    assert len(completed) == 1
    assert completed[0].request_id == response.headers["x-request-id"]
    assert completed[0].tenant_id == "tenant-a"
    assert completed[0].user_id == "user-a"
    assert completed[0].attributes["path"] == "/v1/rag/query"
    assert completed[0].attributes["status_code"] == 200


@pytest.mark.parametrize(
    "body",
    [
        b'{"message": "   "}',
        b'{"message": "question", "filters": {"score": NaN}}',
    ],
    ids=["whitespace-message", "nan-filter"],
)
def test_invalid_query_returns_location_and_type_without_echoing_input(
    make_settings: Callable[..., Settings],
    body: bytes,
) -> None:
    query = FakeQueryHandler()
    context = make_context(make_settings(), query=query)

    response = TestClient(make_app(context)).post(
        "/v1/rag/query",
        content=body,
        headers={**TRUSTED_HEADERS, "Content-Type": "application/json"},
    )

    payload = response.json()
    assert response.status_code == 422
    assert payload["error"] == "INVALID_REQUEST"
    assert payload["details"]
    for detail in payload["details"]:
        assert set(detail) == {"loc", "msg", "type"}
    assert len(response.headers.get_list("x-request-id")) == 1
    assert query.identities == []


def test_unhandled_error_is_500_with_request_id_in_response_and_events(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(make_settings())
    app = make_app(context)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("unexpected")

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    request_id = response.headers["x-request-id"]
    assert response.status_code == 500
    assert response.json()["error"] == "INTERNAL_ERROR"
    failed = context.telemetry.named("http.request.failed")
    assert [(event.request_id, dict(event.attributes)) for event in failed] == [
        (request_id, {"error_code": "RuntimeError"})
    ]
    completed = context.telemetry.named("http.request.completed")
    assert completed[0].request_id == request_id
    assert completed[0].attributes["status_code"] == 500
    assert completed[0].attributes["error_code"] == "RuntimeError"


def test_validation_failure_is_logged_by_type_without_the_model_text(
    make_settings: Callable[..., Settings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = make_app(make_context(make_settings()))

    @app.get("/parse")
    async def parse() -> None:
        GenerationDraft.model_validate_json('{"answer": "Acme bills 4')

    with caplog.at_level(logging.ERROR):
        response = TestClient(app, raise_server_exceptions=False).get("/parse")

    failures = [record for record in caplog.records if record.getMessage() == "Request failed"]
    assert response.status_code == 500
    assert len(failures) == 1
    assert failures[0].exc_info is None
    assert "Acme" not in caplog.text


def test_deadline_error_is_504_and_records_failure(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(
        make_settings(),
        query=FakeQueryHandler(RequestDeadlineExceededError("deadline")),
    )

    response = TestClient(make_app(context)).post(
        "/v1/rag/query",
        json={"message": "question"},
        headers=TRUSTED_HEADERS,
    )

    failed = context.telemetry.named("http.request.failed")
    assert response.status_code == 504
    assert response.json()["error"] == "REQUEST_DEADLINE_EXCEEDED"
    assert [dict(event.attributes) for event in failed] == [
        {"error_code": "RequestDeadlineExceededError"}
    ]
    assert failed[0].request_id == response.headers["x-request-id"]


def test_provider_timeout_is_504_operation_timeout_not_a_deadline_and_records_failure(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(
        make_settings(),
        query=FakeQueryHandler(TimeoutError("provider timed out")),
    )

    response = TestClient(make_app(context)).post(
        "/v1/rag/query",
        json={"message": "question"},
        headers=TRUSTED_HEADERS,
    )

    failed = context.telemetry.named("http.request.failed")
    assert response.status_code == 504
    assert response.json()["error"] == "OPERATION_TIMEOUT"
    assert [dict(event.attributes) for event in failed] == [{"error_code": "TimeoutError"}]
    assert failed[0].request_id == response.headers["x-request-id"]


def test_ready_route_reports_ready_when_dependencies_pass(
    make_settings: Callable[..., Settings],
) -> None:
    response = TestClient(make_app(make_context(make_settings()))).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_route_maps_dependency_failure_to_503_and_records_it(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(
        make_settings(),
        readiness=FakeReadiness(InfrastructureError("mongo down")),
    )

    response = TestClient(make_app(context)).get("/health/ready")

    failed = context.telemetry.named("http.request.failed")
    assert response.status_code == 503
    assert response.json()["error"] == "SERVICE_UNAVAILABLE"
    assert [dict(event.attributes) for event in failed] == [{"error_code": "InfrastructureError"}]
    assert failed[0].request_id == response.headers["x-request-id"]


def test_trace_route_shows_owner_and_administrator_but_hides_other_users(
    make_settings: Callable[..., Settings],
) -> None:
    repository = FakeRepository()
    repository.traces[REQUEST_ID] = make_trace(REQUEST_ID)
    client = TestClient(make_app(make_context(make_settings(), repository=repository)))
    url = f"/v1/rag/requests/{REQUEST_ID}/trace"

    owner = client.get(url, headers=DEBUG_HEADERS)
    other_user = client.get(url, headers={**DEBUG_HEADERS, "X-User-ID": "user-b"})
    administrator = client.get(
        url,
        headers={**DEBUG_HEADERS, "X-User-ID": "user-b", "X-Roles": "administrator"},
    )

    assert owner.status_code == 200
    assert owner.json()["request_id"] == REQUEST_ID
    assert other_user.status_code == 404
    assert other_user.json()["error"] == "NOT_FOUND"
    assert administrator.status_code == 200


def test_trace_route_requires_debug_token(make_settings: Callable[..., Settings]) -> None:
    client = TestClient(make_app(make_context(make_settings())))

    response = client.get(f"/v1/rag/requests/{REQUEST_ID}/trace", headers=TRUSTED_HEADERS)

    assert response.status_code == 403


def test_completed_event_records_route_template_or_unmatched_marker(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(make_settings())
    client = TestClient(make_app(context))

    client.get(f"/v1/rag/requests/{REQUEST_ID}/trace", headers=DEBUG_HEADERS)
    client.get(f"/unknown/{REQUEST_ID}")

    paths = [
        event.attributes["path"] for event in context.telemetry.named("http.request.completed")
    ]
    assert paths == ["/v1/rag/requests/{request_id}/trace", "<unmatched>"]


async def test_owner_reads_their_trace() -> None:
    handler = make_trace_handler(make_trace(REQUEST_ID))

    trace = await handler.get_trace(REQUEST_ID, make_identity())

    assert trace.request_id == REQUEST_ID


async def test_administrator_reads_another_users_trace_in_the_same_tenant() -> None:
    handler = make_trace_handler(make_trace(REQUEST_ID))
    administrator = make_identity(user_id="user-b", roles=frozenset({"administrator"}))

    trace = await handler.get_trace(REQUEST_ID, administrator)

    assert trace.user_id == "user-a"


async def test_other_user_gets_not_found_for_another_users_trace() -> None:
    handler = make_trace_handler(make_trace(REQUEST_ID))

    with pytest.raises(NotFoundError, match="Trace not found"):
        await handler.get_trace(REQUEST_ID, make_identity(user_id="user-b"))


async def test_other_tenant_and_missing_traces_are_not_found() -> None:
    handler = make_trace_handler(make_trace(REQUEST_ID, tenant_id="tenant-b"))
    administrator = make_identity(user_id="user-a", roles=frozenset({"administrator"}))

    with pytest.raises(NotFoundError, match="Trace not found"):
        await handler.get_trace(REQUEST_ID, administrator)
    with pytest.raises(NotFoundError, match="Trace not found"):
        await handler.get_trace(MISSING_REQUEST_ID, administrator)


def make_trace_handler(trace: RequestTrace) -> TenantTraceHandler:
    repository = FakeRepository()
    repository.traces[trace.request_id] = trace
    return TenantTraceHandler(reader=repository, admin_role="administrator")


class RecordingCheck:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.calls = 0

    async def health_check(self) -> None:
        self.calls += 1
        if self._error is not None:
            raise self._error


async def test_readiness_runs_every_dependency_check() -> None:
    checks = [RecordingCheck(), RecordingCheck()]

    await DependencyReadinessHandler(checks).check()

    assert [check.calls for check in checks] == [1, 1]


async def test_readiness_reports_a_failed_dependency() -> None:
    checks = [RecordingCheck(), RecordingCheck(InfrastructureError("qdrant down"))]

    with pytest.raises(InfrastructureError, match="qdrant down"):
        await DependencyReadinessHandler(checks).check()


class LifecycleContainer:
    def __init__(self, start_error: Exception | None = None) -> None:
        self._start_error = start_error
        self.closed = False

    async def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error

    async def close(self) -> None:
        self.closed = True


def stub_container_factory(
    monkeypatch: pytest.MonkeyPatch,
    container: LifecycleContainer,
    error: Exception | None = None,
) -> None:
    async def create(settings: Settings) -> LifecycleContainer:
        del settings
        if error is not None:
            raise error
        return container

    monkeypatch.setattr("app.create_container", create)


def test_lifespan_starts_then_closes_the_container(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = LifecycleContainer()
    stub_container_factory(monkeypatch, container)
    app = create_lifespan_app(make_settings())

    with TestClient(app):
        assert app.state.container is container
        assert not container.closed

    assert container.closed


def test_lifespan_closes_a_constructed_container_when_startup_fails(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = LifecycleContainer(start_error=RuntimeError("start failed"))
    stub_container_factory(monkeypatch, container)

    with (
        pytest.raises(RuntimeError, match="start failed"),
        TestClient(create_lifespan_app(make_settings())),
    ):
        pass
    assert container.closed


def test_lifespan_reports_construction_failure_without_masking_it(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_container_factory(
        monkeypatch,
        LifecycleContainer(),
        error=RuntimeError("construction failed"),
    )

    with (
        pytest.raises(RuntimeError, match="construction failed"),
        TestClient(create_lifespan_app(make_settings())),
    ):
        pass
