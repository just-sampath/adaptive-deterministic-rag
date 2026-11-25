from __future__ import annotations

import hashlib
import hmac
import importlib
import logging
from collections.abc import Iterator

import pytest
from openai._utils import _logs

from models.domain import (
    AttemptOutcome,
    AttemptTrace,
    CompletenessEvaluation,
    CompletenessStatus,
    ContradictedClaim,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    ModelUsage,
    RequestTrace,
    RequirementEvaluation,
    RequirementStatus,
    RetrievalStrategy,
    RetryDecision,
    RetryHypothesis,
    UnsupportedClaim,
)
from services.telemetry import TelemetryService
from tests.fakes import FakeRepository
from utils.logging import configure_logging
from utils.settings import ContentMode, TelemetryConfig

_HASH_KEY = b"test-telemetry-key"
_PAYLOAD_LOGGERS = (
    "openai",
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "pymongo",
    "qdrant_client",
)
_SERVER_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def make_service(mode: ContentMode, key: bytes = _HASH_KEY) -> TelemetryService:
    return TelemetryService(FakeRepository(), TelemetryConfig(content_mode=mode), hash_key=key)


def make_attempt() -> AttemptTrace:
    return AttemptTrace(
        attempt_number=1,
        strategy=RetrievalStrategy.HAPPY_PATH,
        retrieval_queries=("what is the secret plan", "secret keyword"),
        candidate_k=30,
        generation_k=8,
        dense_result_count=0,
        sparse_result_count=0,
        fused_result_count=0,
        candidates=(),
        packed_chunk_ids=(),
        faithfulness=FaithfulnessEvaluation(
            passed=False,
            failure_type=FaithfulnessStatus.UNSUPPORTED_CLAIMS,
            unsupported_claims=(UnsupportedClaim(claim="unsupported secret"),),
            contradicted_claims=(ContradictedClaim(claim="contradicted secret"),),
            confidence=0.5,
        ),
        completeness=CompletenessEvaluation(
            passed=False,
            failure_type=CompletenessStatus.MISSING_ANSWER_CONTENT,
            requirements=(
                RequirementEvaluation(
                    requirement="requirement secret",
                    status=RequirementStatus.MISSING,
                ),
            ),
            confidence=0.5,
        ),
        failure_types=(),
        generation_repair_attempted=False,
        latency_ms=1,
        token_usage=ModelUsage(),
        answer_observation={},
        decision=RetryDecision(
            hypothesis=RetryHypothesis.LOW_RECALL,
            next_strategy=RetrievalStrategy.RECALL_EXPANSION,
            drift_score=0.4,
            reason="secret reason",
        ),
    )


def make_trace() -> RequestTrace:
    return RequestTrace(
        request_id="request",
        tenant_id="tenant",
        user_id="user",
        literal_question_observation={},
        canonical_question_observation={},
        attempted_strategies=(),
        generation_repair_used=False,
        attempts=(make_attempt(),),
        final_status=None,
        final_source_ids=(),
        latency_ms=1,
        token_usage=ModelUsage(),
        prompt_versions={},
        runtime_versions={},
        configuration_fingerprint="test",
    )


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_payload_levels = {name: logging.getLogger(name).level for name in _PAYLOAD_LOGGERS}
    saved_server_loggers = {
        name: (logging.getLogger(name).handlers[:], logging.getLogger(name).propagate)
        for name in _SERVER_LOGGERS
    }
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    for name, level in saved_payload_levels.items():
        logging.getLogger(name).setLevel(level)
    for name, (handlers, propagate) in saved_server_loggers.items():
        logging.getLogger(name).handlers[:] = handlers
        logging.getLogger(name).propagate = propagate


@pytest.mark.parametrize("mode", [ContentMode.REDACTED, ContentMode.METADATA_ONLY])
def test_projection_removes_every_free_text_field(mode: ContentMode) -> None:
    service = make_service(mode)

    projected = service.project_trace(make_trace())

    serialized = projected.model_dump_json()
    attempt = projected.attempts[0]
    assert "secret" not in serialized
    assert attempt.retrieval_queries == (
        service.redact_text("what is the secret plan"),
        service.redact_text("secret keyword"),
    )
    assert attempt.faithfulness and attempt.completeness
    assert attempt.faithfulness.unsupported_claims[0].claim == service.redact_text(
        "unsupported secret"
    )
    assert attempt.faithfulness.contradicted_claims[0].claim == service.redact_text(
        "contradicted secret"
    )
    assert attempt.completeness.requirements[0].requirement == service.redact_text(
        "requirement secret"
    )
    assert attempt.decision is not None
    assert attempt.decision.reason == service.redact_text("secret reason")


def test_full_mode_projection_keeps_trace_text() -> None:
    trace = make_trace()

    assert make_service(ContentMode.FULL).project_trace(trace) == trace


def test_redacted_text_is_keyed_digest_with_length() -> None:
    service = make_service(ContentMode.REDACTED)
    digest = hmac.new(_HASH_KEY, b"hello", hashlib.sha256).hexdigest()

    assert service.redact_text("hello") == f"hmac-sha256:{digest[:32]}:len=5"
    assert service.observe_text("hello") == {
        "mode": "redacted",
        "length": 5,
        "hmac_sha256": digest,
    }


def test_metadata_only_keeps_length_and_no_digest() -> None:
    service = make_service(ContentMode.METADATA_ONLY)

    assert service.redact_text("hello") == "len=5"
    assert service.observe_text("hello") == {"mode": "metadata_only", "length": 5}


def test_full_mode_observation_keeps_text() -> None:
    service = make_service(ContentMode.FULL)

    assert service.redact_text("hello") == "hello"
    assert service.observe_text("hello") == {"mode": "full", "content": "hello", "length": 5}


def test_hash_key_changes_redacted_digest() -> None:
    first = make_service(ContentMode.REDACTED, b"first-key")
    second = make_service(ContentMode.REDACTED, b"second-key")
    repeat = make_service(ContentMode.REDACTED, b"first-key")

    assert first.redact_text("hello") != second.redact_text("hello")
    assert first.observe_text("hello")["hmac_sha256"] != second.observe_text("hello")["hmac_sha256"]
    assert first.redact_text("hello") == repeat.redact_text("hello")


def test_telemetry_refuses_an_empty_hash_key_so_digests_stay_keyed() -> None:
    with pytest.raises(ValueError, match="hash key must not be empty"):
        make_service(ContentMode.REDACTED, b"")


async def test_pending_and_persisted_traces_are_both_projected() -> None:
    sink = FakeRepository()
    config = TelemetryConfig(content_mode=ContentMode.METADATA_ONLY)
    telemetry = TelemetryService(sink, config, hash_key=_HASH_KEY)
    await telemetry.start()
    await telemetry.save_trace(make_trace())

    pending = await telemetry.get_trace("request", "tenant")
    await telemetry.close()

    persisted = sink.traces["request"]
    assert pending == persisted
    assert "secret" not in persisted.model_dump_json()
    assert persisted.attempts[0].retrieval_queries[0] == "len=23"


@pytest.mark.usefixtures("restore_logging")
def test_configure_logging_keeps_sdk_payload_loggers_at_warning() -> None:
    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.DEBUG
    for name in _PAYLOAD_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


@pytest.mark.usefixtures("restore_logging")
def test_configure_logging_mutes_the_transport_logger_the_installed_openai_sdk_uses() -> None:
    configure_logging("DEBUG")

    for sdk_logger in (_logs.logger, _logs.httpx2_logger):
        assert sdk_logger.name in _PAYLOAD_LOGGERS
        assert logging.getLogger(sdk_logger.name).getEffectiveLevel() == logging.WARNING


@pytest.mark.usefixtures("restore_logging")
def test_configure_logging_routes_uvicorn_logs_through_the_json_handler() -> None:
    plain = logging.StreamHandler()
    for name in _SERVER_LOGGERS:
        logging.getLogger(name).addHandler(plain)
        logging.getLogger(name).propagate = False

    configure_logging("INFO")

    for name in _SERVER_LOGGERS:
        assert logging.getLogger(name).handlers == []
        assert logging.getLogger(name).propagate is True


# httpx2 sends requests through these httpcore2 modules; each logs request traces at DEBUG.
@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param("httpcore2._async.connection", id="connection"),
        pytest.param("httpcore2._async.http11", id="http11"),
        pytest.param("httpcore2._async.http2", id="http2"),
        pytest.param("httpcore2._async.http_proxy", id="http-proxy"),
    ],
)
@pytest.mark.usefixtures("restore_logging")
def test_configure_logging_mutes_the_installed_httpcore2_transport_loggers(
    module_name: str,
) -> None:
    transport_logger = importlib.import_module(module_name).logger

    configure_logging("DEBUG")

    assert transport_logger.name.partition(".")[0] in _PAYLOAD_LOGGERS
    assert transport_logger.getEffectiveLevel() == logging.WARNING


@pytest.mark.parametrize(
    "mode", [ContentMode.FULL, ContentMode.REDACTED, ContentMode.METADATA_ONLY]
)
def test_projection_keeps_an_unevaluated_attempt_without_verdicts(mode: ContentMode) -> None:
    attempt = AttemptTrace.model_validate(
        {
            **make_attempt().model_dump(),
            "outcome": AttemptOutcome.EMPTY_DRAFT,
            "faithfulness": None,
            "completeness": None,
        }
    )
    trace = make_trace().model_copy(update={"attempts": (attempt,)})

    (projected,) = make_service(mode).project_trace(trace).attempts

    assert projected.outcome is AttemptOutcome.EMPTY_DRAFT
    assert projected.faithfulness is None and projected.completeness is None
    if mode is not ContentMode.FULL:
        assert "secret" not in projected.model_dump_json()
