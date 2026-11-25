from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import create_autospec

import httpx
import pytest
from bson import BSON
from fastapi import FastAPI
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from managers.telemetry import MongoTelemetrySink
from middleware.request_telemetry import RequestTelemetryMiddleware
from models.domain import (
    AttemptOutcome,
    AttemptTrace,
    AttributedFaithfulness,
    AttributionStatus,
    CitationAssessment,
    ContextualizationTrace,
    ModelUsage,
    RequestTrace,
    RewriteDecision,
    TelemetryEvent,
)
from routes.dependencies import RequestIdentity
from services.telemetry import TelemetryService
from tests.fakes import FakeRepository
from tests.test_redaction import make_attempt
from utils.errors import InfrastructureError
from utils.logging import JsonFormatter
from utils.request_context import current_request_id
from utils.settings import ContentMode, Environment, ExternalSettings, Settings, TelemetryConfig

_HASH_KEY = b"test-telemetry-key"


class ScriptedSink:
    """Telemetry sink whose writes can hang or fail on demand and that never serves reads."""

    def __init__(self, *, hang: bool = False, error: Exception | None = None) -> None:
        self._hang = hang
        self._error = error
        self.written: list[str] = []
        self.cancelled = False

    async def write_events(self, events: Sequence[TelemetryEvent]) -> None:
        del events
        await self._write()

    async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
        await self._write()
        self.written.extend(trace.request_id for trace in traces)

    async def get_trace(self, request_id: str, tenant_id: str) -> RequestTrace | None:
        del request_id, tenant_id
        return None

    async def _write(self) -> None:
        try:
            if self._hang:
                await asyncio.Event().wait()
            if self._error is not None:
                raise self._error
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def make_trace(request_id: str) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        tenant_id="tenant",
        user_id="user",
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


def rejected_rewrite_trace() -> RequestTrace:
    context = ContextualizationTrace(
        enabled=True,
        detector="model",
        history_messages=2,
        selected_messages=1,
        proposed_drift_score=0.25,
        rewrite_decision=RewriteDecision.REJECTED_ANCHOR,
    )
    return make_trace("rewrite").model_copy(update={"contextualization": context})


@pytest.mark.parametrize("enriched", [False, True], ids=["legacy", "enriched"])
async def test_mongo_trace_layout_and_read_after_disabling_writes(enriched: bool) -> None:
    collection = create_autospec(AsyncCollection, instance=True)
    database = cast(AsyncDatabase[dict[str, Any]], {"traces": collection, "events": None})
    reader = MongoTelemetrySink(database, trace_collection="traces", event_collection="events")
    writer = reader
    if enriched:
        writer = MongoTelemetrySink(
            database,
            trace_collection="traces",
            event_collection="events",
            write_query_intent_fields=True,
        )

    await writer.write_traces([rejected_rewrite_trace()])

    document = BSON(BSON.encode(collection.insert_many.await_args.args[0][0])).decode()
    expected = {
        "enabled": True,
        "detector": "model",
        "followup_probability": None,
        "fail_open": False,
        "history_messages": 2,
        "selected_messages": 1,
        "rewritten": False,
    }
    if enriched:
        expected.update(proposed_drift_score=0.25, rewrite_decision="rejected_anchor")
    assert document["contextualization"] == expected
    collection.find_one.return_value = document
    trace = await reader.get_trace("rewrite", "tenant")
    assert trace is not None and trace.contextualization is not None
    assert trace.contextualization.proposed_drift_score == (0.25 if enriched else None)
    assert trace.contextualization.rewrite_decision == ("rejected_anchor" if enriched else None)


@pytest.mark.parametrize("enriched", [False, True], ids=["legacy", "enriched"])
@pytest.mark.parametrize(
    "mode", [ContentMode.FULL, ContentMode.REDACTED, ContentMode.METADATA_ONLY]
)
async def test_pending_trace_exposes_the_configured_query_intent_fields(
    enriched: bool, mode: ContentMode
) -> None:
    service = TelemetryService(
        FakeRepository(),
        TelemetryConfig(content_mode=mode, write_query_intent_fields=enriched),
        hash_key=_HASH_KEY,
    )
    await service.start()
    try:
        await service.save_trace(rejected_rewrite_trace())
        pending = await service.get_trace("rewrite", "tenant")
    finally:
        await service.close()

    assert pending is not None and pending.contextualization is not None
    assert pending.contextualization.history_messages == 2
    assert pending.contextualization.proposed_drift_score == (0.25 if enriched else None)
    assert pending.contextualization.rewrite_decision == ("rejected_anchor" if enriched else None)
    assert await service.get_trace("rewrite", "tenant") == pending


def log_field(record: logging.LogRecord, name: str) -> object:
    return record.__dict__[name]


def error_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.ERROR]


async def test_telemetry_shutdown_flushes_queued_mongo_events() -> None:
    sink = FakeRepository()
    telemetry = TelemetryService(
        sink, TelemetryConfig(queue_capacity=4, batch_size=2), hash_key=_HASH_KEY
    )
    await telemetry.start()

    telemetry.emit("first", request_id="request")
    telemetry.emit("second", request_id="request")
    trace = make_trace("request")
    await telemetry.save_trace(trace)

    assert await telemetry.get_trace("request", "tenant") == trace
    await telemetry.close()

    assert [event.event_type for event in sink.events] == ["first", "second"]
    assert set(sink.traces) == {"request"}


async def test_request_context_is_isolated_between_concurrent_requests() -> None:
    sink = FakeRepository()
    telemetry = TelemetryService(
        sink, TelemetryConfig(queue_capacity=10, batch_size=10), hash_key=_HASH_KEY
    )
    await telemetry.start()
    app = FastAPI()
    app.state.container = SimpleNamespace(telemetry=telemetry)
    app.add_middleware(RequestTelemetryMiddleware)

    @app.get("/probe/{delay}")
    async def probe(delay: float) -> dict[str, str | None]:
        await asyncio.sleep(delay)
        return {"request_id": current_request_id()}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        slow, fast = await asyncio.gather(client.get("/probe/0.01"), client.get("/probe/0"))
    await telemetry.close()

    slow_id = slow.json()["request_id"]
    fast_id = fast.json()["request_id"]
    assert slow_id == slow.headers["x-request-id"]
    assert fast_id == fast.headers["x-request-id"]
    assert slow_id != fast_id


async def test_async_identity_context_reaches_request_telemetry() -> None:
    sink = FakeRepository()
    telemetry = TelemetryService(
        sink, TelemetryConfig(queue_capacity=10, batch_size=10), hash_key=_HASH_KEY
    )
    await telemetry.start()
    settings = Settings(
        ai={"provider": "fake"},
        reranker={"provider": "fake"},
        external=ExternalSettings(
            _env_file=None,
            app_env=Environment.TEST,
            auth_shared_token="test-auth-token",
            debug_token="test-debug-token",
        ),
    )
    app = FastAPI()
    app.state.container = SimpleNamespace(telemetry=telemetry, settings=settings)
    app.add_middleware(RequestTelemetryMiddleware)

    @app.get("/identity")
    async def identity_probe(identity: RequestIdentity) -> dict[str, str]:
        return {"tenant_id": identity.tenant_id}

    headers = {
        "X-Auth-Token": "test-auth-token",
        "X-Tenant-ID": "tenant-a",
        "X-User-ID": "user-a",
        "X-Roles": "reader",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/identity", headers=headers)
    await telemetry.close()

    assert response.status_code == 200
    completed = next(event for event in sink.events if event.event_type == "http.request.completed")
    assert completed.tenant_id == "tenant-a"
    assert completed.user_id == "user-a"


async def test_trace_saved_without_running_writer_is_written_directly() -> None:
    sink = ScriptedSink()
    telemetry = TelemetryService(sink, TelemetryConfig(), hash_key=_HASH_KEY)

    await telemetry.save_trace(make_trace("request"))

    assert sink.written == ["request"]


async def test_direct_write_is_bounded_and_logged_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = ScriptedSink(hang=True)
    config = TelemetryConfig(direct_write_timeout_seconds=0.05)
    telemetry = TelemetryService(sink, config, hash_key=_HASH_KEY)

    await asyncio.wait_for(telemetry.save_trace(make_trace("request")), timeout=2)

    record = error_records(caplog)[0]
    assert log_field(record, "request_id") == "request"
    assert log_field(record, "tenant_id") == "tenant"
    assert log_field(record, "error_code") == "TimeoutError"
    assert sink.written == []


async def test_direct_write_failure_is_logged_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = ScriptedSink(error=InfrastructureError("MongoDB operation failed"))
    telemetry = TelemetryService(sink, TelemetryConfig(), hash_key=_HASH_KEY)

    await telemetry.save_trace(make_trace("request"))

    record = error_records(caplog)[0]
    assert log_field(record, "error_code") == "InfrastructureError"
    assert log_field(record, "request_id") == "request"


async def test_pending_trace_is_kept_when_batch_write_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = ScriptedSink(error=InfrastructureError("MongoDB operation failed"))
    telemetry = TelemetryService(sink, TelemetryConfig(), hash_key=_HASH_KEY)
    await telemetry.start()
    trace = make_trace("request")

    await telemetry.save_trace(trace)
    await telemetry.close()

    assert await telemetry.get_trace("request", "tenant") == trace
    record = error_records(caplog)[0]
    assert log_field(record, "error_code") == "InfrastructureError"
    assert log_field(record, "request_ids") == ["request"]


async def test_successful_batch_write_releases_pending_trace() -> None:
    sink = ScriptedSink()
    telemetry = TelemetryService(sink, TelemetryConfig(), hash_key=_HASH_KEY)
    await telemetry.start()
    await telemetry.save_trace(make_trace("request"))
    assert await telemetry.get_trace("request", "tenant") is not None

    await telemetry.close()

    assert sink.written == ["request"]
    assert await telemetry.get_trace("request", "tenant") is None


async def test_pending_traces_are_capped_at_queue_capacity() -> None:
    config = TelemetryConfig(queue_capacity=2, batch_size=1, shutdown_timeout_seconds=0.05)
    telemetry = TelemetryService(ScriptedSink(hang=True), config, hash_key=_HASH_KEY)
    await telemetry.start()
    await telemetry.save_trace(make_trace("first"))
    await asyncio.sleep(0)  # the writer takes "first" and blocks inside the sink

    await telemetry.save_trace(make_trace("second"))
    await telemetry.save_trace(make_trace("third"))

    assert await telemetry.get_trace("first", "tenant") is None
    assert await telemetry.get_trace("second", "tenant") is not None
    assert await telemetry.get_trace("third", "tenant") is not None
    await telemetry.close()


async def test_failed_event_batch_is_logged_and_the_writer_finishes_cleanly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = ScriptedSink(error=InfrastructureError("MongoDB operation failed"))
    telemetry = TelemetryService(sink, TelemetryConfig(), hash_key=_HASH_KEY)
    await telemetry.start()
    telemetry.emit("http.request.completed", request_id="request")

    await telemetry.close()

    record = error_records(caplog)[0]
    assert record.getMessage() == "Failed to persist telemetry event batch"
    assert log_field(record, "error_code") == "InfrastructureError"
    assert log_field(record, "request_ids") == ["request"]


async def test_full_event_queue_drops_new_events_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = TelemetryConfig(queue_capacity=1, batch_size=1, shutdown_timeout_seconds=0.05)
    telemetry = TelemetryService(ScriptedSink(hang=True), config, hash_key=_HASH_KEY)
    await telemetry.start()
    telemetry.emit("first")
    await asyncio.sleep(0)  # the writer takes "first" and blocks inside the sink
    telemetry.emit("second")
    telemetry.emit("third")

    dropped = [r for r in caplog.records if "event was not persisted" in r.getMessage()]
    assert len(dropped) == 1
    await telemetry.close()


async def test_full_trace_queue_drops_the_trace_without_remembering_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = TelemetryConfig(queue_capacity=1, batch_size=1, shutdown_timeout_seconds=0.05)
    telemetry = TelemetryService(ScriptedSink(hang=True), config, hash_key=_HASH_KEY)
    await telemetry.start()
    await telemetry.save_trace(make_trace("first"))
    await asyncio.sleep(0)  # the writer takes "first" and blocks inside the sink
    await telemetry.save_trace(make_trace("second"))
    await telemetry.save_trace(make_trace("third"))

    assert await telemetry.get_trace("third", "tenant") is None
    assert any("trace was not persisted" in r.getMessage() for r in caplog.records)
    await telemetry.close()


async def test_start_runs_a_single_writer_even_when_called_twice() -> None:
    telemetry = TelemetryService(ScriptedSink(), TelemetryConfig(), hash_key=_HASH_KEY)

    await telemetry.start()
    await telemetry.start()

    writers = [task for task in asyncio.all_tasks() if task.get_name() == "mongo-telemetry-writer"]
    assert len(writers) == 1
    await telemetry.close()


async def test_disabled_telemetry_starts_no_writer_and_persists_nothing() -> None:
    sink = ScriptedSink()
    telemetry = TelemetryService(sink, TelemetryConfig(enabled=False), hash_key=_HASH_KEY)

    await telemetry.start()
    await telemetry.save_trace(make_trace("request"))

    assert sink.written == []
    assert await telemetry.get_trace("request", "tenant") is None


async def test_close_is_bounded_when_writer_hangs() -> None:
    sink = ScriptedSink(hang=True)
    telemetry = TelemetryService(
        sink, TelemetryConfig(shutdown_timeout_seconds=0.05), hash_key=_HASH_KEY
    )
    await telemetry.start()
    await telemetry.save_trace(make_trace("request"))
    await asyncio.sleep(0)

    await asyncio.wait_for(telemetry.close(), timeout=2)
    await asyncio.sleep(0)

    assert sink.cancelled


async def test_close_returns_when_writer_task_has_died() -> None:
    telemetry = TelemetryService(
        ScriptedSink(), TelemetryConfig(queue_capacity=1), hash_key=_HASH_KEY
    )
    await telemetry.start()
    telemetry.emit("queued")
    writer = next(
        task for task in asyncio.all_tasks() if task.get_name() == "mongo-telemetry-writer"
    )
    writer.cancel()
    await asyncio.sleep(0)

    await asyncio.wait_for(telemetry.close(), timeout=1)

    assert writer.cancelled()


def test_json_formatter_reports_structured_failure_fields() -> None:
    record = logging.makeLogRecord(
        {
            "name": "services.telemetry",
            "levelno": logging.ERROR,
            "levelname": "ERROR",
            "msg": "Failed to persist RAG trace batch",
            "request_ids": ["first", "second"],
            "error_code": "InfrastructureError",
        }
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "Failed to persist RAG trace batch"
    assert payload["request_ids"] == ["first", "second"]
    assert payload["error_code"] == "InfrastructureError"


@pytest.mark.parametrize(("length", "kept"), [(255, 255), (256, 256), (300, 256)])
def test_long_text_attributes_are_capped_at_256_characters(length: int, kept: int) -> None:
    event = TelemetryEvent(event_type="rag.test", attributes={"text": "x" * length})

    assert event.attributes == {"text": "x" * kept}


def test_truncation_leaves_non_text_attributes_unchanged() -> None:
    attributes = {"count": 7, "ratio": 0.5, "flag": True, "missing": None, "short": "ok"}

    event = TelemetryEvent(event_type="rag.test", attributes=attributes)

    assert event.attributes == attributes


# The attempt layout written before answer verification, pinned independently of the model.
_LEGACY_ATTEMPT_KEYS = {
    "attempt_number",
    "strategy",
    "retrieval_queries",
    "candidate_k",
    "generation_k",
    "dense_result_count",
    "sparse_result_count",
    "fused_result_count",
    "candidates",
    "packed_chunk_ids",
    "faithfulness",
    "completeness",
    "failure_types",
    "generation_repair_attempted",
    "latency_ms",
    "token_usage",
    "model_calls",
    "answer_observation",
    "decision",
}
_LEGACY_FAITHFULNESS_KEYS = {
    "passed",
    "failure_type",
    "verified_sources",
    "unsupported_claims",
    "contradicted_claims",
    "correcting_evidence_available",
    "correcting_chunks",
    "confidence",
}


def _attempt_with(outcome: AttemptOutcome | None) -> AttemptTrace:
    legacy = make_attempt()
    if outcome is None:
        return legacy
    update: dict[str, Any] = {"outcome": outcome, "faithfulness": None, "completeness": None}
    if outcome is AttemptOutcome.EVALUATED:
        assert legacy.faithfulness is not None
        update["faithfulness"] = AttributedFaithfulness(
            **legacy.faithfulness.model_dump(),
            occurrence_assessments=(
                CitationAssessment(
                    occurrence_id=1,
                    supported_labels=(),
                    attribution_status=AttributionStatus.UNSUPPORTED,
                ),
            ),
            has_uncited_material_claims=True,
        )
        update["completeness"] = legacy.completeness
    return AttemptTrace.model_validate({**legacy.model_dump(), **update})


@pytest.mark.parametrize(
    "outcome", [None, AttemptOutcome.EVALUATED, AttemptOutcome.EMPTY_DRAFT], ids=str
)
async def test_attempt_outcome_layout_and_reads(outcome: AttemptOutcome | None) -> None:
    collection = create_autospec(AsyncCollection, instance=True)
    database = cast(AsyncDatabase[dict[str, Any]], {"traces": collection, "events": None})
    sink = MongoTelemetrySink(database, trace_collection="traces", event_collection="events")
    trace = make_trace("attempt").model_copy(update={"attempts": (_attempt_with(outcome),)})

    await sink.write_traces([trace])

    document = BSON(BSON.encode(collection.insert_many.await_args.args[0][0])).decode()
    stored = document["attempts"][0]
    if outcome is None:
        # A writer with verification disabled emits exactly the previous attempt layout.
        assert set(stored) == _LEGACY_ATTEMPT_KEYS
        assert set(stored["faithfulness"]) == _LEGACY_FAITHFULNESS_KEYS
    else:
        assert stored["outcome"] == outcome.value
    collection.find_one.return_value = document
    read = await sink.get_trace("attempt", "tenant")
    assert read is not None
    assert read.attempts == trace.attempts
    assert type(read.attempts[0].faithfulness) is type(trace.attempts[0].faithfulness)


@pytest.mark.parametrize(
    "update",
    [
        {"outcome": AttemptOutcome.EMPTY_DRAFT},
        {"outcome": AttemptOutcome.EVALUATED},
        {"outcome": None, "faithfulness": None, "completeness": None},
    ],
    ids=["verdicts-without-evaluation", "evaluated-without-assessments", "legacy-without-verdicts"],
)
def test_attempt_verdicts_must_match_the_outcome(update: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"verdicts|assessments"):
        AttemptTrace.model_validate({**make_attempt().model_dump(), **update})
