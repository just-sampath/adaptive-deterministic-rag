"""Central, provider-pluggable telemetry pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from collections.abc import Mapping, Sequence

from models.domain import (
    AttemptTrace,
    CompletenessEvaluation,
    FaithfulnessEvaluation,
    RequestTrace,
    RetryDecision,
    TelemetryEvent,
)
from services.contracts import TelemetrySink
from utils.settings import ContentMode, TelemetryConfig

logger = logging.getLogger(__name__)

TelemetryValue = str | int | float | bool | None
_DIGEST_PREFIX_LENGTH = 32


class TelemetryService:
    """Queue low-cardinality events and persist request traces without blocking RAG."""

    def __init__(self, sink: TelemetrySink, config: TelemetryConfig, *, hash_key: bytes) -> None:
        if not hash_key:
            # An empty key makes HMAC-SHA256 a public, unkeyed digest of redacted content.
            raise ValueError("Telemetry hash key must not be empty")
        self._sink = sink
        self._config = config
        self._hash_key = hash_key
        self._queue: asyncio.Queue[TelemetryEvent | RequestTrace | None] = asyncio.Queue(
            maxsize=config.queue_capacity
        )
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False
        self._pending_traces: dict[tuple[str, str], RequestTrace] = {}

    async def start(self) -> None:
        if not self._config.enabled or self._worker is not None:
            return
        self._accepting = True
        self._worker = asyncio.create_task(self._run(), name="mongo-telemetry-writer")

    async def close(self) -> None:
        self._accepting = False
        worker, self._worker = self._worker, None
        if worker is None or worker.done():
            return
        timeout = self._config.shutdown_timeout_seconds
        try:
            await asyncio.wait_for(self._queue.put(None), timeout=timeout)
        except TimeoutError:
            self._abandon(worker)
            return
        _, still_running = await asyncio.wait({worker}, timeout=timeout)
        if still_running:
            self._abandon(worker)

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
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "Telemetry queue is full; event was not persisted",
                extra={"request_id": request_id, "tenant_id": tenant_id, "event_type": event_type},
            )

    async def save_trace(self, trace: RequestTrace) -> None:
        if not self._config.enabled:
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
        if self._config.content_mode is not ContentMode.FULL:
            updates["attempts"] = tuple(self._project_attempt(item) for item in trace.attempts)
        return trace.model_copy(update=updates) if updates else trace

    def _project_attempt(self, attempt: AttemptTrace) -> AttemptTrace:
        return attempt.model_copy(
            update={
                "retrieval_queries": tuple(
                    self.redact_text(query) for query in attempt.retrieval_queries
                ),
                "faithfulness": self._project_faithfulness(attempt.faithfulness),
                "completeness": self._project_completeness(attempt.completeness),
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
        return self._accepting and self._worker is not None and not self._worker.done()

    def _enqueue_trace(self, trace: RequestTrace) -> None:
        try:
            self._queue.put_nowait(trace)
        except asyncio.QueueFull:
            logger.warning(
                "Telemetry queue is full; trace was not persisted",
                extra={"request_id": trace.request_id, "tenant_id": trace.tenant_id},
            )
            return
        self._remember_pending(trace)

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
        try:
            await asyncio.wait_for(
                self._sink.write_traces((trace,)),
                timeout=self._config.direct_write_timeout_seconds,
            )
        except Exception as exc:
            logger.exception(
                "Failed to persist RAG request trace directly",
                extra={
                    "request_id": trace.request_id,
                    "tenant_id": trace.tenant_id,
                    "error_code": type(exc).__name__,
                },
            )

    def _abandon(self, worker: asyncio.Task[None]) -> None:
        worker.cancel()
        logger.error(
            "Telemetry writer exceeded the shutdown timeout and was cancelled",
            extra={
                "error_code": "TimeoutError",
                "request_ids": [request_id for _, request_id in self._pending_traces],
            },
        )

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            batch: list[TelemetryEvent | RequestTrace] = [first]
            stopping = self._fill_batch(batch)
            try:
                await self._write_batch(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()
            if stopping:
                return

    def _fill_batch(self, batch: list[TelemetryEvent | RequestTrace]) -> bool:
        """Append queued items up to the batch size; return True at the shutdown sentinel."""

        while len(batch) < self._config.batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return False
            if item is None:
                self._queue.task_done()
                return True
            batch.append(item)
        return False

    async def _write_batch(self, items: Sequence[TelemetryEvent | RequestTrace]) -> None:
        events = [item for item in items if isinstance(item, TelemetryEvent)]
        traces = [item for item in items if isinstance(item, RequestTrace)]
        if events:
            await self._write_events(events)
        if traces:
            await self._write_traces(traces)

    async def _write_events(self, events: list[TelemetryEvent]) -> None:
        try:
            await self._sink.write_events(events)
        except Exception as exc:
            logger.exception(
                "Failed to persist telemetry event batch",
                extra={"error_code": type(exc).__name__, "request_ids": _request_ids(events)},
            )

    async def _write_traces(self, traces: list[RequestTrace]) -> None:
        try:
            await self._sink.write_traces(traces)
        except Exception as exc:
            logger.exception(
                "Failed to persist RAG trace batch",
                extra={"error_code": type(exc).__name__, "request_ids": _request_ids(traces)},
            )
            return
        for trace in traces:
            self._forget_pending(trace)


def _request_ids(items: Sequence[TelemetryEvent | RequestTrace]) -> list[str]:
    """Return the distinct request IDs of a batch in first-seen order."""

    return list(dict.fromkeys(item.request_id for item in items if item.request_id is not None))
