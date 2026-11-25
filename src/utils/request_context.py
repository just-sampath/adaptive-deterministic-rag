"""Task-local request and identity context for concurrent ASGI requests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from models.domain import (
    EmbeddingResult,
    Identity,
    ModelCallTrace,
    ModelResult,
    ModelUsage,
    OperationStatus,
    ProviderOperation,
    RerankMetering,
    RetrievalStrategy,
    UsageCoverage,
)

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_identity: ContextVar[Identity | None] = ContextVar("identity", default=None)


@dataclass(frozen=True, slots=True)
class RequestContextTokens:
    request_id: Token[str | None]
    identity: Token[Identity | None]


def bind_request(request_id: str) -> RequestContextTokens:
    return RequestContextTokens(
        request_id=_request_id.set(request_id),
        identity=_identity.set(None),
    )


def bind_identity(identity: Identity) -> None:
    _identity.set(identity)


def reset_request(tokens: RequestContextTokens) -> None:
    _identity.reset(tokens.identity)
    _request_id.reset(tokens.request_id)


def current_request_id() -> str | None:
    return _request_id.get()


def current_identity() -> Identity | None:
    return _identity.get()


@dataclass(slots=True)
class _Slot:
    row: ProviderOperation
    received: bool = False


class OperationCollector:
    """A single request owns its slots. Parallel children update only their reserved slot."""

    def __init__(self) -> None:
        self.slots: list[_Slot] = []
        self.strategy: RetrievalStrategy | None = None
        self.strategy_try: int | None = None
        self.repair_ordinal = 0

    def snapshot(self) -> tuple[ProviderOperation, ...]:
        return tuple(slot.row for slot in self.slots)

    def summary(self) -> tuple[ModelUsage, UsageCoverage, tuple[ModelCallTrace, ...]]:
        total = ModelUsage()
        calls: list[ModelCallTrace] = []
        known = unknown = 0
        for slot in self.slots:
            row = slot.row
            if row.status is OperationStatus.REUSED:
                continue
            if row.usage is not None:
                total += row.usage
            if row.usage is not None or row.rerank_metering is not None:
                known += 1
            else:
                unknown += 1
            if slot.received and row.provider is not None and row.model is not None:
                calls.append(
                    ModelCallTrace(
                        operation=row.operation,
                        provider=row.provider,
                        model=row.model,
                        request_id=row.request_id,
                        usage=row.usage or ModelUsage(),
                    )
                )
        coverage = UsageCoverage.COMPLETE
        if unknown:
            coverage = UsageCoverage.PARTIAL if known else UsageCoverage.UNKNOWN
        return total, coverage, tuple(calls)


_operations: ContextVar[OperationCollector | None] = ContextVar("operations", default=None)
_operation_slot: ContextVar[_Slot | None] = ContextVar("operation_slot", default=None)


@contextmanager
def collect_operations() -> Iterator[OperationCollector]:
    collector = OperationCollector()
    token = _operations.set(collector)
    slot_token = _operation_slot.set(None)
    try:
        yield collector
    finally:
        _operation_slot.reset(slot_token)
        _operations.reset(token)


def current_operations() -> OperationCollector | None:
    return _operations.get()


@contextmanager
def operation(
    stage: str,
    model: str,
    *,
    reused: bool = False,
    provider: str | None = None,
    adapter: bool = False,
) -> Iterator[None]:
    collector = current_operations()
    parent = _operation_slot.get()
    if collector is None or (adapter and parent is not None):
        if adapter and parent is not None:
            parent.row = parent.row.model_copy(update={"provider": provider, "model": model})
        yield
        return
    slot = _Slot(
        ProviderOperation(
            slot=len(collector.slots) + 1,
            operation=stage,
            model=model,
            provider=provider,
            strategy=collector.strategy,
            strategy_try=collector.strategy_try,
            repair_ordinal=collector.repair_ordinal,
            status=OperationStatus.REUSED if reused else OperationStatus.COMPLETED,
        )
    )
    collector.slots.append(slot)
    token = _operation_slot.set(slot)
    try:
        yield
    except (Exception, asyncio.CancelledError) as exc:
        if not reused:
            status = (
                OperationStatus.CANCELLED
                if isinstance(exc, asyncio.CancelledError)
                else OperationStatus.FAILED
            )
            slot.row = slot.row.model_copy(update={"status": status})
        raise
    finally:
        _operation_slot.reset(token)


def observe_response(
    *,
    provider: str,
    model: str,
    request_id: str | None,
    usage: ModelUsage | None,
    metering: RerankMetering | None = None,
) -> None:
    slot = _operation_slot.get()
    if slot is not None:
        slot.received = True
        slot.row = slot.row.model_copy(
            update={
                "provider": provider,
                "model": model,
                "request_id": request_id,
                "usage": usage,
                "rerank_metering": metering,
            }
        )


async def observed_call[T](stage: str, model: str, call: Awaitable[T]) -> T:
    with operation(stage, model):
        result = await call
        slot = _operation_slot.get()
        if (
            slot is not None
            and not slot.received
            and isinstance(result, (ModelResult, EmbeddingResult))
        ):
            observe_response(
                provider=result.provider,
                model=result.model,
                request_id=result.request_id,
                usage=result.usage if result.usage.model_fields_set else None,
            )
        return result
