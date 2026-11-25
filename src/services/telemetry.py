"""Central, provider-pluggable telemetry pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto

from models.domain import (
    AttemptTrace,
    CompletenessEvaluation,
    FaithfulnessEvaluation,
    HandlerOutcome,
    RequestTrace,
    RetryDecision,
    TelemetryEvent,
    UsageCoverage,
)
from services.contracts import TelemetrySink
from utils.errors import TraceConflictError, TransientTraceWriteError
from utils.settings import ContentMode, TelemetryConfig

logger = logging.getLogger(__name__)

TelemetryValue = str | int | float | bool | None
_DIGEST_PREFIX_LENGTH = 32


class _WriterState(Enum):
    NEW = auto()
    RUNNING = auto()
    CLOSING = auto()
    CLOSED = auto()


@dataclass(slots=True)
class DeliveryCounts:
    dropped_events: int = 0
    dropped_traces: int = 0
    trace_retries: int = 0
    exhausted_trace_batches: int = 0
    shutdown_abandoned: int = 0


class TelemetryService:
    """Queue low-cardinality events and persist request traces without blocking RAG."""

    def __init__(
        self,
        sink: TelemetrySink,
        config: TelemetryConfig,
        *,
        hash_key: bytes,
        store_timeout_seconds: float | None = None,
    ) -> None:
        if not hash_key:
            # An empty key makes HMAC-SHA256 a public, unkeyed digest of redacted content.
            raise ValueError("Telemetry hash key must not be empty")
        self._sink = sink
        self._config = config
        self._hash_key = hash_key
        self._store_timeout = (
            config.direct_write_timeout_seconds
            if store_timeout_seconds is None
            else store_timeout_seconds
        )
        self._events: deque[TelemetryEvent] = deque()
        self._traces: deque[RequestTrace] = deque()
        self._ready = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._state = _WriterState.NEW
        self._closed = asyncio.Event()
        self._shutdown_deadline: float | None = None
        self._in_flight: tuple[TelemetryEvent | RequestTrace, ...] = ()
        self._direct_writes: set[asyncio.Task[None]] = set()
        self._counts = DeliveryCounts()
        self._pending_traces: dict[tuple[str, str], RequestTrace] = {}

    @property
    def delivery_counts(self) -> DeliveryCounts:
        return replace(self._counts)

    async def start(self) -> None:
        if not self._config.enabled or self._state is not _WriterState.NEW:
            return
        self._state = _WriterState.RUNNING
        self._worker = asyncio.create_task(self._run(), name="mongo-telemetry-writer")

    async def close(self) -> None:
        if self._state is _WriterState.CLOSED:
            return
        if self._state is _WriterState.CLOSING:
            await self._closed.wait()
            return
        self._state = _WriterState.CLOSING
        loop = asyncio.get_running_loop()
        self._shutdown_deadline = loop.time() + self._config.shutdown_timeout_seconds
        self._ready.set()
        tasks = set(self._direct_writes)
        if self._worker is not None:
            tasks.add(self._worker)
        try:
            if tasks:
                await asyncio.wait(tasks, timeout=max(0, self._shutdown_deadline - loop.time()))
        finally:
            self._state = _WriterState.CLOSED
            items = (*self._in_flight, *self._traces, *self._events)
            abandoned = len(items) + sum(not task.done() for task in self._direct_writes)
            self._events.clear()
            self._traces.clear()
            if abandoned:
                self._counts.shutdown_abandoned += abandoned
                logger.error(
                    "Telemetry shutdown abandoned %d items",
                    abandoned,
                    extra={"request_ids": _request_ids(items)},
                )
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            try:
                if pending:
                    # Let cooperative writers observe cancellation. A sink can suppress it,
                    # so the owning container must be free to close its clients afterwards.
                    await asyncio.sleep(0)
            finally:
                self._closed.set()

    def emit(
        self,
        event_type: str,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        attributes: Mapping[str, TelemetryValue] | None = None,
    ) -> None:
        if not self._is_accepting():
            return
        event = TelemetryEvent(
            event_type=event_type,
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            attributes=dict(attributes or {}),
        )
        if self._waiting_count() >= self._config.queue_capacity:
            self._drop_event(event)
        else:
            self._events.append(event)
            self._ready.set()

    async def save_trace(self, trace: RequestTrace) -> None:
        if not self._config.enabled:
            return
        if self._state is not _WriterState.NEW and not self._is_accepting():
            self._drop_trace(trace, "Telemetry writer is unavailable; trace was not persisted")
            return
        projected = self.project_trace(trace)
        if self._is_accepting():
            self._enqueue_trace(projected)
            return
        await self._write_trace_directly(projected)

    async def get_trace(self, request_id: str, tenant_id: str) -> RequestTrace | None:
        pending = self._pending_traces.get((tenant_id, request_id))
        if pending is not None:
            return pending
        return await self._sink.get_trace(request_id, tenant_id)

    def observe_text(self, text: str) -> dict[str, str | int]:
        if self._config.content_mode is ContentMode.FULL:
            return {"mode": "full", "content": text, "length": len(text)}
        if self._config.content_mode is ContentMode.REDACTED:
            return {"mode": "redacted", "length": len(text), "hmac_sha256": self._digest(text)}
        return {"mode": "metadata_only", "length": len(text)}

    def redact_text(self, text: str) -> str:
        if self._config.content_mode is ContentMode.FULL:
            return text
        if self._config.content_mode is ContentMode.REDACTED:
            return f"hmac-sha256:{self._digest(text)[:_DIGEST_PREFIX_LENGTH]}:len={len(text)}"
        return f"len={len(text)}"

    def project_trace(self, trace: RequestTrace) -> RequestTrace:
        """Apply the configured content mode and expose the same values before and after storage."""

        updates: dict[str, object] = {}
        if trace.contextualization is not None and not self._config.write_query_intent_fields:
            updates["contextualization"] = trace.contextualization.model_copy(
                update={"proposed_drift_score": None, "rewrite_decision": None}
            )
        if not self._config.write_accounting_fields:
            # The sink omits these keys, so a pending read shows what the stored record reads as.
            updates.update(
                operations=None,
                usage_coverage=UsageCoverage.UNKNOWN,
                handler_outcome=HandlerOutcome.UNKNOWN,
                returned_status=None,
                returned_document_ids=None,
            )
        if self._config.content_mode is not ContentMode.FULL:
            updates["attempts"] = tuple(self._project_attempt(item) for item in trace.attempts)
        return trace.model_copy(update=updates) if updates else trace

    def _project_attempt(self, attempt: AttemptTrace) -> AttemptTrace:
        return attempt.model_copy(
            update={
                "retrieval_queries": tuple(
                    self.redact_text(query) for query in attempt.retrieval_queries
                ),
                "faithfulness": self._project_faithfulness(attempt.faithfulness)
                if attempt.faithfulness
                else None,
                "completeness": self._project_completeness(attempt.completeness)
                if attempt.completeness
                else None,
                "decision": self._project_decision(attempt.decision),
            }
        )

    def _project_decision(self, decision: RetryDecision | None) -> RetryDecision | None:
        if decision is None:
            return None
        return decision.model_copy(update={"reason": self.redact_text(decision.reason)})

    def _project_faithfulness(self, evaluation: FaithfulnessEvaluation) -> FaithfulnessEvaluation:
        return evaluation.model_copy(
            update={
                "unsupported_claims": tuple(
                    item.model_copy(update={"claim": self.redact_text(item.claim)})
                    for item in evaluation.unsupported_claims
                ),
                "contradicted_claims": tuple(
                    item.model_copy(update={"claim": self.redact_text(item.claim)})
                    for item in evaluation.contradicted_claims
                ),
            }
        )

    def _project_completeness(self, evaluation: CompletenessEvaluation) -> CompletenessEvaluation:
        return evaluation.model_copy(
            update={
                "requirements": tuple(
                    item.model_copy(update={"requirement": self.redact_text(item.requirement)})
                    for item in evaluation.requirements
                )
            }
        )

    def _digest(self, text: str) -> str:
        message = text.encode("utf-8", errors="surrogatepass")
        return hmac.new(self._hash_key, message, hashlib.sha256).hexdigest()

    def _is_accepting(self) -> bool:
        return (
            self._state is _WriterState.RUNNING
            and self._worker is not None
            and not self._worker.done()
        )

    def _waiting_count(self) -> int:
        return len(self._events) + len(self._traces)

    def _drop_event(self, event: TelemetryEvent) -> None:
        self._counts.dropped_events += 1
        logger.warning(
            "Telemetry queue is full; event was not persisted",
            extra={
                "request_id": event.request_id,
                "tenant_id": event.tenant_id,
                "event_type": event.event_type,
            },
        )

    def _drop_trace(self, trace: RequestTrace, reason: str) -> None:
        self._counts.dropped_traces += 1
        logger.warning(reason, extra={"request_id": trace.request_id, "tenant_id": trace.tenant_id})

    def _enqueue_trace(self, trace: RequestTrace) -> None:
        pending = self._pending_traces.get((trace.tenant_id, trace.request_id))
        if pending is not None:
            if pending != trace:
                self._drop_trace(trace, "Conflicting immutable telemetry trace was not persisted")
            return
        if self._waiting_count() >= self._config.queue_capacity:
            if not self._events:
                self._drop_trace(trace, "Telemetry queue is full; trace was not persisted")
                return
            self._drop_event(self._events.pop())
        self._traces.append(trace)
        self._remember_pending(trace)
        self._ready.set()

    def _remember_pending(self, trace: RequestTrace) -> None:
        key = (trace.tenant_id, trace.request_id)
        self._pending_traces.pop(key, None)
        self._pending_traces[key] = trace
        while len(self._pending_traces) > self._config.queue_capacity:
            del self._pending_traces[next(iter(self._pending_traces))]

    def _forget_pending(self, trace: RequestTrace) -> None:
        key = (trace.tenant_id, trace.request_id)
        if self._pending_traces.get(key) is trace:
            del self._pending_traces[key]

    async def _write_trace_directly(self, trace: RequestTrace) -> None:
        if len(self._direct_writes) >= self._config.queue_capacity:
            self._drop_trace(
                trace, "Direct telemetry writes are at capacity; trace was not persisted"
            )
            return
        task = asyncio.create_task(self._sink.write_traces((trace,)))
        self._direct_writes.add(task)
        task.add_done_callback(self._direct_finished)
        try:
            _, pending = await asyncio.wait(
                {task}, timeout=self._config.direct_write_timeout_seconds
            )
            if pending:
                task.cancel()
                raise TimeoutError
            if not task.cancelled():
                task.result()
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            self._counts.exhausted_trace_batches += 1
            logger.exception(
                "Failed to persist RAG request trace directly",
                extra={
                    "request_id": trace.request_id,
                    "tenant_id": trace.tenant_id,
                    "error_code": type(exc).__name__,
                },
            )

    def _direct_finished(self, task: asyncio.Task[None]) -> None:
        self._direct_writes.discard(task)
        if not task.cancelled():
            # The caller may already have timed out before the driver finishes.
            task.exception()

    def _can_write(self) -> bool:
        return self._state is not _WriterState.CLOSED and (
            self._shutdown_deadline is None
            or asyncio.get_running_loop().time() < self._shutdown_deadline
        )

    async def _run(self) -> None:
        while self._can_write():
            if not self._waiting_count():
                if self._state is _WriterState.CLOSING:
                    return
                self._ready.clear()
                await self._ready.wait()
                continue
            # Dequeue a single kind at a time. Every new batch first serves waiting traces.
            try:
                if self._traces:
                    traces = self._take_batch(self._traces)
                    self._in_flight = traces
                    await self._write_traces(traces)
                else:
                    events = self._take_batch(self._events)
                    self._in_flight = events
                    await self._write_events(events)
            finally:
                self._in_flight = ()

    def _take_batch[T](self, queue: deque[T]) -> tuple[T, ...]:
        return tuple(queue.popleft() for _ in range(min(len(queue), self._config.batch_size)))

    async def _write_events(self, events: tuple[TelemetryEvent, ...]) -> None:
        try:
            async with asyncio.timeout(self._store_timeout):
                await self._sink.write_events(events)
        except Exception as exc:
            logger.exception(
                "Failed to persist telemetry event batch",
                extra={"error_code": type(exc).__name__, "request_ids": _request_ids(events)},
            )

    async def _write_traces(self, traces: tuple[RequestTrace, ...]) -> None:
        for attempt in range(2):
            try:
                async with asyncio.timeout(self._store_timeout):
                    await self._sink.write_traces(traces)
            except Exception as exc:
                if self._state is _WriterState.CLOSED:
                    # close() already counted this in-flight batch as abandoned. A sink that
                    # suppressed its cancellation must not be counted a second time.
                    return
                retryable = isinstance(exc, (TransientTraceWriteError, TimeoutError))
                if not attempt and retryable and self._can_write():
                    self._counts.trace_retries += 1
                    logger.warning(
                        "Retrying transient RAG trace write",
                        extra={
                            "error_code": type(exc).__name__,
                            "request_ids": _request_ids(traces),
                        },
                    )
                    continue
                if not attempt and retryable:
                    # The retry was owed but the shutdown deadline had passed: abandoned, not
                    # exhausted, so the two counts keep their separate meanings.
                    self._counts.shutdown_abandoned += len(traces)
                    logger.error(
                        "Telemetry shutdown abandoned a RAG trace batch before its retry",
                        extra={
                            "error_code": type(exc).__name__,
                            "request_ids": _request_ids(traces),
                        },
                    )
                    return
                self._counts.exhausted_trace_batches += 1
                logger.exception(
                    "Failed to persist RAG trace batch",
                    extra={"error_code": type(exc).__name__, "request_ids": _request_ids(traces)},
                )
                if isinstance(exc, TraceConflictError):
                    # A lookup must return the immutable record already in Mongo.
                    for trace in traces:
                        self._forget_pending(trace)
            else:
                for trace in traces:
                    self._forget_pending(trace)
            return


def _request_ids(items: Sequence[TelemetryEvent | RequestTrace]) -> list[str]:
    """Return the distinct request IDs of a batch in first-seen order."""

    return list(dict.fromkeys(item.request_id for item in items if item.request_id is not None))
