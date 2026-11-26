"""One retry for transient provider failures, bounded by the time its stage has left."""

from __future__ import annotations

import asyncio
import email.utils
import time

import pytest
from pydantic import BaseModel

from models.domain import (
    CompletenessEvaluation,
    CompletenessStatus,
    EmbeddingResult,
    EncodedQuery,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    GenerationDraft,
    HybridHits,
    Identity,
    ModelResult,
    OperationStatus,
    PackedContext,
    QueryUnderstanding,
    RankedChunk,
    RequirementEvaluation,
    RequirementStatus,
    RetrievalPlan,
)
from services.encoding import EmbeddingCache
from services.evaluation import GenerationService, OnlineEvaluator
from services.retrieval import HybridRetriever
from tests.fakes import (
    FakeEmbeddings,
    FakeQueryEncoder,
    FakeRepository,
    FakeTextModel,
    FakeVectorSearch,
    make_chunk,
)
from utils.errors import (
    OperationTimeoutError,
    ProviderError,
    ProviderResponseError,
    TransientProviderError,
)
from utils.request_context import (
    collect_operations,
    current_deadline,
    deadline_scope,
    observed_call,
    stage_timeout,
)
from utils.retry import retry_after_seconds, retry_transient
from utils.settings import RetrievalConfig, RetryConfig

_RETRY = RetryConfig(max_retries=1, backoff_seconds=0, min_retry_seconds=0)


class _Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class _Calls:
    """Fail with the queued errors in order, then return the ordinal of the successful try."""

    def __init__(self, *errors: Exception) -> None:
        self._errors = list(errors)
        self.ordinals: list[int] = []

    async def __call__(self, retry_ordinal: int) -> int:
        self.ordinals.append(retry_ordinal)
        if self._errors:
            raise self._errors.pop(0)
        return retry_ordinal


async def test_a_transient_failure_is_retried_once_after_the_backoff() -> None:
    calls = _Calls(TransientProviderError("busy"))
    sleeps = _Sleeps()
    policy = RetryConfig(max_retries=1, backoff_seconds=0.25, min_retry_seconds=0)

    result = await retry_transient("generation", calls, policy=policy, sleep=sleeps)

    assert result == 1
    assert calls.ordinals == [0, 1]
    assert sleeps.waits == [0.25]


async def test_a_second_transient_failure_is_raised() -> None:
    calls = _Calls(TransientProviderError("busy"), TransientProviderError("still busy"))

    with pytest.raises(TransientProviderError, match="still busy"):
        await retry_transient("generation", calls, policy=_RETRY, sleep=_Sleeps())

    assert calls.ordinals == [0, 1]


async def test_retries_turned_off_make_a_single_try() -> None:
    calls = _Calls(TransientProviderError("busy"))

    with pytest.raises(TransientProviderError):
        await retry_transient(
            "generation", calls, policy=RetryConfig(max_retries=0), sleep=_Sleeps()
        )

    assert calls.ordinals == [0]


@pytest.mark.parametrize(
    "error",
    [
        OperationTimeoutError("slow"),
        TimeoutError(),
        ProviderResponseError("invalid output"),
        ProviderError("unauthorized"),
    ],
    ids=["adapter-timeout", "stage-timeout", "invalid-output", "permanent"],
)
async def test_timeouts_invalid_output_and_permanent_failures_are_not_retried(
    error: Exception,
) -> None:
    calls = _Calls(error)
    sleeps = _Sleeps()

    with pytest.raises(type(error)):
        await retry_transient("generation", calls, policy=_RETRY, sleep=sleeps)

    assert calls.ordinals == [0]
    assert sleeps.waits == []


async def test_the_wait_follows_the_providers_retry_after() -> None:
    calls = _Calls(TransientProviderError("rate limited", retry_after_seconds=0.75))
    sleeps = _Sleeps()

    await retry_transient("rerank", calls, policy=_RETRY, sleep=sleeps)

    assert sleeps.waits == [0.75]


@pytest.mark.parametrize(
    ("retry_after", "time_left"),
    [(None, 0.5), (5.0, 2.0)],
    ids=["too-little-time", "retry-after-past-the-deadline"],
)
async def test_no_retry_starts_when_the_stage_could_not_finish_it(
    retry_after: float | None, time_left: float
) -> None:
    calls = _Calls(TransientProviderError("busy", retry_after_seconds=retry_after))
    sleeps = _Sleeps()
    policy = RetryConfig(max_retries=1, backoff_seconds=0.1, min_retry_seconds=1)

    with (
        deadline_scope(asyncio.get_running_loop().time() + time_left),
        pytest.raises(TransientProviderError),
    ):
        await retry_transient("rerank", calls, policy=policy, sleep=sleeps)

    assert calls.ordinals == [0]
    assert sleeps.waits == []


async def test_stage_timers_publish_the_earliest_deadline_and_still_time_out() -> None:
    loop = asyncio.get_running_loop()
    assert current_deadline() is None

    with deadline_scope(loop.time() + 0.05):
        attempt_end = current_deadline()
        async with stage_timeout(30):
            # The enclosing attempt deadline is earlier than the stage's own.
            assert current_deadline() == attempt_end
    assert current_deadline() is None

    async with stage_timeout(30):
        stage_end = current_deadline()
        assert stage_end is not None
        assert stage_end - loop.time() <= 30
    assert current_deadline() is None

    with pytest.raises(TimeoutError):
        async with stage_timeout(0.01):
            await asyncio.sleep(1)
    assert current_deadline() is None


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "2"}, 2.0),
        ({"retry-after": "0.5"}, 0.5),
        ({"retry-after-ms": "soon", "retry-after": "3"}, 3.0),
        ({"retry-after": "-1"}, None),
        ({"retry-after": "inf"}, None),
        ({"retry-after": "later"}, None),
        ({}, None),
    ],
)
def test_retry_after_headers_are_read_as_seconds(
    headers: dict[str, str], expected: float | None
) -> None:
    assert retry_after_seconds(headers) == expected


def test_a_retry_after_date_counts_down_and_a_past_date_waits_nothing() -> None:
    future = email.utils.formatdate(time.time() + 30, usegmt=True)
    past = email.utils.formatdate(time.time() - 30, usegmt=True)

    wait = retry_after_seconds({"retry-after": future})

    assert wait is not None
    assert 25 <= wait <= 30
    assert retry_after_seconds({"retry-after": past}) == 0


async def test_each_try_of_a_retried_call_is_its_own_operation_row() -> None:
    calls = _Calls(TransientProviderError("busy"))

    with collect_operations() as collector:
        await retry_transient(
            "generation",
            lambda retry: observed_call("generation", "model-a", calls(retry), retry_ordinal=retry),
            policy=_RETRY,
            sleep=_Sleeps(),
        )

    rows = collector.snapshot()
    assert [(row.retry_ordinal, row.status) for row in rows] == [
        (0, OperationStatus.FAILED),
        (1, OperationStatus.COMPLETED),
    ]


class _FailsOnce(FakeTextModel):
    """Raise one transient failure for the given output type, then answer from the queue."""

    def __init__(self, failing: type[BaseModel]) -> None:
        super().__init__()
        self._failing: type[BaseModel] | None = failing

    async def complete[SchemaT: BaseModel](
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]:
        if output_type is self._failing:
            self._failing = None
            self.calls.append(output_type)
            raise TransientProviderError("busy")
        return await super().complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=output_type,
            max_output_tokens=max_output_tokens,
        )


def _context() -> PackedContext:
    chunk = make_chunk("00000000-0000-0000-0000-000000000001")
    return PackedContext(
        chunks=(RankedChunk(chunk=chunk, fusion_score=1, fusion_rank=1),),
        token_count=chunk.token_count,
    )


async def test_a_judge_retry_does_not_cancel_the_other_judge() -> None:
    model = _FailsOnce(FaithfulnessEvaluation)
    model.queue(
        FaithfulnessEvaluation,
        FaithfulnessEvaluation(
            passed=True,
            failure_type=FaithfulnessStatus.SUPPORTED,
            verified_sources=("E1",),
            confidence=0.9,
        ),
    )
    model.queue(
        CompletenessEvaluation,
        CompletenessEvaluation(
            passed=True,
            failure_type=CompletenessStatus.COMPLETE,
            requirements=(
                RequirementEvaluation(
                    requirement="the answer",
                    status=RequirementStatus.ANSWERED,
                    available_in_context=True,
                    supporting_chunks=("E1",),
                ),
            ),
            confidence=0.9,
        ),
    )
    evaluator = OnlineEvaluator(
        model=model,
        model_name="judge",
        max_output_tokens=256,
        timeout_seconds=1,
        retry=_RETRY,
    )

    with collect_operations() as collector:
        result = await evaluator.evaluate(
            query=QueryUnderstanding(
                literal_question="What is supported?",
                canonical_question="What is supported?",
                rewritten=False,
            ),
            answer="Supported evidence [E1].",
            context=_context(),
        )

    assert result.faithfulness.passed and result.completeness.passed
    # The completeness judge ran once; the faithfulness retry didn't cancel and restart it.
    assert model.calls.count(CompletenessEvaluation) == 1
    assert model.calls.count(FaithfulnessEvaluation) == 2
    rows = [(row.operation, row.retry_ordinal, row.status) for row in collector.snapshot()]
    assert ("faithfulness_evaluation", 0, OperationStatus.FAILED) in rows
    assert ("faithfulness_evaluation", 1, OperationStatus.COMPLETED) in rows
    assert ("completeness_evaluation", 0, OperationStatus.COMPLETED) in rows


class _EmbeddingsFail(FakeEmbeddings):
    """Fail the first ``failures`` queries with a transient error once released."""

    def __init__(self, failures: int = 1) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self._failures = failures

    async def embed_query(self, text: str) -> EmbeddingResult:
        await self.release.wait()
        if self.query_calls < self._failures:
            self.query_calls += 1
            raise TransientProviderError("busy")
        return await super().embed_query(text)


async def test_embedding_waiters_share_the_leaders_retried_result() -> None:
    provider = _EmbeddingsFail()
    cache = EmbeddingCache(10, retry=_RETRY)

    lookups = [asyncio.create_task(cache.get_or_create("query", provider)) for _ in range(3)]
    await asyncio.sleep(0)
    provider.release.set()
    results = await asyncio.gather(*lookups)

    # One failed try and one retry, both made by the leader; the waiters shared its result.
    assert provider.query_calls == 2
    assert {tuple(result.vector) for result in results} == {(5.0, 0.0, 1.0)}
    assert sum(not result.cached for result in results) == 1


async def test_a_leader_out_of_time_for_its_retry_hands_the_call_to_a_waiter() -> None:
    provider = _EmbeddingsFail()
    cache = EmbeddingCache(
        10, retry=RetryConfig(max_retries=1, backoff_seconds=0, min_retry_seconds=1)
    )
    loop = asyncio.get_running_loop()

    async def leader() -> EmbeddingResult:
        # Too little of this request's time is left for a retry.
        with deadline_scope(loop.time() + 0.5):
            return await cache.get_or_create("query", provider)

    first = asyncio.create_task(leader())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(cache.get_or_create("query", provider))
    await asyncio.sleep(0)
    provider.release.set()

    with pytest.raises(TransientProviderError):
        await first
    result = await waiter

    # The waiter had time left, so it made its own call instead of inheriting the failure.
    assert provider.query_calls == 2
    assert not result.cached


async def test_a_leader_that_used_its_retry_shares_the_failure() -> None:
    provider = _EmbeddingsFail(failures=2)
    cache = EmbeddingCache(10, retry=_RETRY)

    lookups = [asyncio.create_task(cache.get_or_create("query", provider)) for _ in range(2)]
    await asyncio.sleep(0)
    provider.release.set()
    results = await asyncio.gather(*lookups, return_exceptions=True)

    assert provider.query_calls == 2
    assert all(isinstance(result, TransientProviderError) for result in results)


class _SearchFailsOnce(FakeVectorSearch):
    async def search(
        self,
        query: EncodedQuery,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        *,
        dense_k: int,
        sparse_k: int,
    ) -> HybridHits:
        if self.search_calls == 0:
            self.search_calls += 1
            raise TransientProviderError("rate limited")
        return await super().search(query, identity, filters, dense_k=dense_k, sparse_k=sparse_k)


async def test_vector_search_is_retried_inside_the_retrieval_timer(identity: Identity) -> None:
    vector_store = _SearchFailsOnce()
    retriever = HybridRetriever(
        encoder=FakeQueryEncoder(),
        vector_store=vector_store,
        repository=FakeRepository(),
        config=RetrievalConfig(),
        retrieval_timeout_seconds=1,
        retry=_RETRY,
    )

    outcome = await retriever.retrieve(
        query="question",
        identity=identity,
        filters={},
        plan=RetrievalPlan(dense_k=4, sparse_k=4, candidate_k=4),
    )

    assert vector_store.search_calls == 2
    assert outcome.ranked_chunks == ()


def _generation_service(
    model: FakeTextModel, *, timeout: float, policy: RetryConfig
) -> GenerationService:
    return GenerationService(
        model=model,
        model_name="generator",
        max_output_tokens=256,
        timeout_seconds=timeout,
        retry=policy,
    )


async def _generate(service: GenerationService) -> object:
    return await service.generate(
        query=QueryUnderstanding(
            literal_question="What is supported?",
            canonical_question="What is supported?",
            rewritten=False,
        ),
        context=_context(),
        conversation_facts=(),
    )


async def test_a_stage_timer_too_short_for_the_retry_stops_it() -> None:
    model = _FailsOnce(GenerationDraft)
    service = _generation_service(
        model,
        timeout=0.5,
        policy=RetryConfig(max_retries=1, backoff_seconds=0, min_retry_seconds=1),
    )

    with pytest.raises(TransientProviderError):
        await _generate(service)

    assert model.calls == [GenerationDraft]


class _FailsThenHangs(FakeTextModel):
    async def complete[SchemaT: BaseModel](
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[SchemaT],
        max_output_tokens: int,
    ) -> ModelResult[SchemaT]:
        self.calls.append(output_type)
        if len(self.calls) == 1:
            raise TransientProviderError("busy")
        await asyncio.sleep(60)
        raise AssertionError("the stage timer should have ended the retry")


async def test_a_stage_timer_that_fires_during_the_retry_ends_it_as_a_timeout() -> None:
    model = _FailsThenHangs()
    service = _generation_service(model, timeout=0.2, policy=_RETRY)

    with collect_operations() as collector, pytest.raises(TimeoutError):
        await _generate(service)

    assert model.calls == [GenerationDraft, GenerationDraft]
    assert [(row.retry_ordinal, row.status) for row in collector.snapshot()] == [
        (0, OperationStatus.FAILED),
        (1, OperationStatus.CANCELLED),
    ]
