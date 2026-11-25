from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from models.domain import RequestTrace, TelemetryEvent
from services.telemetry import TelemetryService
from tests.fakes import FakeRepository
from tests.test_telemetry import make_trace
from utils.errors import InfrastructureError
from utils.settings import TelemetryConfig


class DeliverySink(FakeRepository):
    def __init__(self, *, errors: Sequence[Exception] = (), blocked: bool = False) -> None:
        super().__init__()
        self.errors = list(errors)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.trace_batches: list[tuple[RequestTrace, ...]] = []
        self.order: list[str] = []

    async def write_events(self, events: Sequence[TelemetryEvent]) -> None:
        self.order.extend(event.event_type for event in events)
        await super().write_events(events)

    async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
        self.trace_batches.append(tuple(traces))
        self.order.extend(trace.request_id for trace in traces)
        self.entered.set()
        await self.release.wait()
        if self.errors:
            raise self.errors.pop(0)
        await super().write_traces(traces)


def service(sink: DeliverySink, *, capacity: int = 2) -> TelemetryService:
    return TelemetryService(
        sink,
        TelemetryConfig(queue_capacity=capacity, batch_size=2, shutdown_timeout_seconds=0.1),
        hash_key=b"test-telemetry",
    )


@pytest.mark.parametrize("capacity", [1, 2])
async def test_traces_evict_routine_events_and_receive_service_first(capacity: int) -> None:
    sink = DeliverySink()
    telemetry = service(sink, capacity=capacity)
    await telemetry.start()
    for i in range(capacity):
        telemetry.emit(f"event-{i}")
    await telemetry.save_trace(make_trace("trace"))
    await telemetry.close()
    assert sink.order == ["trace", *(["event-0"] if capacity == 2 else [])]
    assert telemetry.delivery_counts.dropped_events == 1
    assert telemetry.delivery_counts.dropped_traces == 0


async def test_all_trace_overload_is_counted_and_keeps_first_owner() -> None:
    sink = DeliverySink()
    telemetry = service(sink, capacity=1)
    await telemetry.start()
    original = make_trace("same")
    await telemetry.save_trace(original)
    await telemetry.save_trace(original.model_copy(update={"user_id": "different"}))
    await telemetry.save_trace(make_trace("overload"))
    assert await telemetry.get_trace("same", "tenant") == original
    await telemetry.close()
    assert sink.order == ["same"]
    assert telemetry.delivery_counts.dropped_traces == 2


@pytest.mark.parametrize("transient_failures", [0, 1, 2])
async def test_only_transient_trace_failures_retry_the_identical_projected_batch(
    transient_failures: int,
) -> None:
    from utils.errors import TransientTraceWriteError

    errors: list[Exception] = [TransientTraceWriteError("transient")] * transient_failures
    if not transient_failures:
        errors.append(InfrastructureError("permanent"))
    sink = DeliverySink(errors=errors)
    telemetry = service(sink)
    await telemetry.start()
    await telemetry.save_trace(make_trace("first"))
    await telemetry.save_trace(make_trace("second"))
    await telemetry.close()
    assert len(sink.trace_batches) == (2 if transient_failures else 1)
    if transient_failures:
        assert all(a is b for a, b in zip(*sink.trace_batches, strict=True))
    assert telemetry.delivery_counts.trace_retries == int(bool(transient_failures))
    assert telemetry.delivery_counts.exhausted_trace_batches == int(transient_failures != 1)
    assert set(sink.traces) == ({"first", "second"} if transient_failures == 1 else set())


async def test_close_waits_for_prestart_direct_write_and_prevents_late_io() -> None:
    sink = DeliverySink(blocked=True)
    telemetry = service(sink)
    direct = asyncio.create_task(telemetry.save_trace(make_trace("admitted")))
    await sink.entered.wait()
    closing = asyncio.create_task(telemetry.close())
    await asyncio.sleep(0)
    assert not closing.done()
    await telemetry.save_trace(make_trace("closing"))
    sink.release.set()
    await direct
    await closing
    await telemetry.start()
    await telemetry.save_trace(make_trace("closed"))
    await telemetry.close()
    assert sink.order == ["admitted"]
    assert telemetry.delivery_counts.dropped_traces == 2


async def test_close_abandons_even_a_sink_that_swallows_cancellation() -> None:
    class StuckSink(DeliverySink):
        async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
            try:
                await super().write_traces(traces)
            except asyncio.CancelledError:
                self.entered.clear()
                await self.release.wait()

    sink = StuckSink(blocked=True)
    telemetry = service(sink, capacity=1)
    await telemetry.start()
    await telemetry.save_trace(make_trace("in-flight"))
    await sink.entered.wait()
    await telemetry.save_trace(make_trace("waiting"))
    try:
        await asyncio.wait_for(telemetry.close(), timeout=1)
        assert not sink.entered.is_set()
        assert telemetry.delivery_counts.shutdown_abandoned == 2
    finally:
        sink.release.set()
        await asyncio.sleep(0)
    assert sink.order == ["in-flight"]


async def test_a_batch_abandoned_at_close_is_counted_once_when_the_sink_fails_late(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from utils.errors import TransientTraceWriteError

    class LateFailureSink(DeliverySink):
        async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
            try:
                await super().write_traces(traces)
            except asyncio.CancelledError:
                await self.release.wait()
                raise TransientTraceWriteError("late") from None

    sink = LateFailureSink(blocked=True)
    telemetry = service(sink, capacity=1)
    await telemetry.start()
    await telemetry.save_trace(make_trace("in-flight"))
    await sink.entered.wait()
    await asyncio.wait_for(telemetry.close(), timeout=1)
    sink.release.set()
    for _ in range(3):
        await asyncio.sleep(0)

    counts = telemetry.delivery_counts
    assert (counts.shutdown_abandoned, counts.trace_retries, counts.exhausted_trace_batches) == (
        1,
        0,
        0,
    )
    (record,) = [item for item in caplog.records if "shutdown abandoned" in item.getMessage()]
    assert record.__dict__["request_ids"] == ["in-flight"]


async def test_expired_shutdown_deadline_allows_no_retry_or_next_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from utils.errors import TransientTraceWriteError

    sink = DeliverySink(errors=[TransientTraceWriteError("transient")], blocked=True)
    telemetry = service(sink, capacity=1)
    await telemetry.start()
    await telemetry.save_trace(make_trace("first"))
    await sink.entered.wait()
    await telemetry.save_trace(make_trace("second"))
    closing = asyncio.create_task(telemetry.close())
    await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    actual_time = loop.time
    monkeypatch.setattr(loop, "time", lambda: actual_time() + 0.2)
    sink.release.set()
    await closing
    assert sink.order == ["first"]
    assert telemetry.delivery_counts.trace_retries == 0
    assert telemetry.delivery_counts.exhausted_trace_batches == 0
    assert telemetry.delivery_counts.shutdown_abandoned == 2


async def test_worker_retries_a_store_timeout_while_running() -> None:
    class TimeoutSink(DeliverySink):
        async def write_traces(self, traces: Sequence[RequestTrace]) -> None:
            if not self.entered.is_set():
                self.entered.set()
                await asyncio.Event().wait()
            await super().write_traces(traces)
            self.release.set()

    sink = TimeoutSink()
    telemetry = TelemetryService(
        sink, TelemetryConfig(), hash_key=b"test", store_timeout_seconds=0.04
    )
    await telemetry.start()
    await telemetry.save_trace(make_trace("trace"))
    await sink.entered.wait()
    await telemetry.close()
    assert sink.order == ["trace"]
    assert telemetry.delivery_counts.trace_retries == 1
