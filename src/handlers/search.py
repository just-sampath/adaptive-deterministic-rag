"""Deterministic RAG orchestration used by the search controller."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from handlers.policy import DecisionPolicy, TerminalResult
from handlers.search_strategies.contracts import StrategyPlan
from handlers.search_strategies.registry import SearchStrategyRegistry
from models.domain import (
    AttemptOutcome,
    AttemptTrace,
    CandidateTrace,
    Chunk,
    CompletenessEvaluation,
    CompletenessStatus,
    ContextualizationTrace,
    ConversationMessage,
    EvidenceRef,
    FaithfulnessEvaluation,
    FaithfulnessStatus,
    Identity,
    ModelCallTrace,
    ModelUsage,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    RagState,
    RequestTrace,
    RequirementEvaluation,
    RequirementStatus,
    ResultStatus,
    RetrievalStrategy,
    RetryDecision,
    Source,
    VerifiedAnswer,
)
from services.citations import (
    CitationRenderer,
    RenderedAnswer,
    citation_occurrences,
    has_material_text,
    sources_from_chunks,
)
from services.context import ContextBuilder, ContextOutcome
from services.contracts import ConversationStore, DocumentReader
from services.evaluation import (
    EvaluationResult,
    GenerationResult,
    GenerationService,
    OnlineEvaluator,
)
from services.history import authorized_history, evidence_withdrawal_reason
from services.retrieval import RetrievalOutcome, StrategyRetriever
from services.telemetry import TelemetryService
from services.understanding import QueryUnderstandingService
from utils.errors import InfrastructureError, RequestDeadlineExceededError
from utils.prompts import PROMPT_VERSIONS, STANDARD_ABSTENTION, VERIFICATION_PROMPT_VERSIONS
from utils.request_context import current_request_id
from utils.settings import ContextualizationConfig, PolicyConfig, RetrievalConfig, TimeoutConfig

_DEADLINE_CODE = "REQUEST_DEADLINE_EXCEEDED"


@dataclass(slots=True)
class _RequestRecord:
    """Facts gathered while one request runs; the trace is built from them in every outcome."""

    request_id: str
    identity: Identity
    literal_observation: dict[str, str | int]
    canonical_observation: dict[str, str | int]
    contextualization: ContextualizationTrace | None = None
    contextualization_usage: ModelUsage = field(default_factory=ModelUsage)
    contextualization_calls: tuple[ModelCallTrace, ...] = ()
    history_refs: tuple[EvidenceRef, ...] = ()
    state: RagState | None = None
    terminal: TerminalResult | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class _Evidence:
    retrieval: RetrievalOutcome
    context: ContextOutcome
    refs: dict[str, EvidenceRef]

    @property
    def available(self) -> bool:
        return bool(self.context.packed_context.chunks)


@dataclass(frozen=True, slots=True)
class _Outcome:
    terminal: TerminalResult
    sources: tuple[Source, ...]
    presented: RenderedAnswer
    history_refs: tuple[EvidenceRef, ...] | None


class SearchHandler:
    """Execute the finite strategy registry with one global generation repair."""

    def __init__(
        self,
        *,
        query_understanding: QueryUnderstandingService,
        retriever: StrategyRetriever,
        context_builder: ContextBuilder,
        generator: GenerationService,
        evaluator: OnlineEvaluator,
        decision: DecisionPolicy,
        strategies: SearchStrategyRegistry,
        documents: DocumentReader,
        conversations: ConversationStore,
        telemetry: TelemetryService,
        citations: CitationRenderer,
        policy: PolicyConfig,
        retrieval: RetrievalConfig,
        contextualization: ContextualizationConfig,
        history_ref_limit: int,
        timeouts: TimeoutConfig,
        runtime_versions: dict[str, str],
        configuration_fingerprint: str,
    ) -> None:
        self._query_understanding = query_understanding
        self._retriever = retriever
        self._context_builder = context_builder
        self._generator = generator
        self._evaluator = evaluator
        self._decision = decision
        self._strategies = strategies
        self._documents = documents
        self._conversations = conversations
        self._telemetry = telemetry
        self._citations = citations
        self._policy = policy
        self._retrieval_config = retrieval
        self._contextualization = contextualization
        self._history_ref_limit = history_ref_limit
        self._timeouts = timeouts
        self._runtime_versions = runtime_versions
        self._configuration_fingerprint = configuration_fingerprint

    async def query(self, identity: Identity, request: QueryRequest) -> QueryResponse:
        started = perf_counter()
        start = asyncio.get_running_loop().time()
        observation = self._telemetry.observe_text(request.message)
        record = _RequestRecord(
            request_id=current_request_id() or str(uuid4()),
            identity=identity,
            literal_observation=observation,
            canonical_observation=observation,
        )
        self._emit(record.request_id, identity, "rag.request.started")
        outcome: _Outcome | None = None
        try:
            outcome = await self._run(record, request, start)
        except Exception as exc:
            record.error_code = _error_code(exc)
            raise
        finally:
            await self._save_trace(record, started)
        assert outcome is not None
        return self._response(record, outcome, started)

    async def _run(self, record: _RequestRecord, request: QueryRequest, start: float) -> _Outcome:
        deadline = asyncio.timeout_at(start + self._timeouts.request_seconds)
        try:
            async with deadline:
                terminal = await self._answer(record, request, start)
                record.terminal = terminal
                outcome = await self._authorize_answer(record, terminal, request)
                await self._append_turn(record.identity, request, outcome)
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise RequestDeadlineExceededError(
                "The RAG request exceeded its global deadline"
            ) from exc
        return outcome

    async def _authorize_answer(
        self, record: _RequestRecord, terminal: TerminalResult, request: QueryRequest
    ) -> _Outcome:
        refs = terminal.evidence_refs
        if terminal.status is ResultStatus.INSUFFICIENT_EVIDENCE:
            return _Outcome(terminal, (), self._present(terminal, {}), ())

        required = tuple(dict.fromkeys((*refs.values(), *record.history_refs)))
        chunk_ids = list(dict.fromkeys(ref.chunk_id for ref in required))
        current = await self._documents.hydrate_authorized_chunks(chunk_ids, record.identity)
        for ref in required:
            reason = evidence_withdrawal_reason(ref, current.get(ref.chunk_id), request.filters)
            if reason is not None:
                return self._withdraw(record, terminal, request, reason)

        evidence = {label: current[ref.chunk_id] for label, ref in refs.items()}
        sources = sources_from_chunks([current[key] for key in terminal.verified_chunk_ids])
        saved_refs = required if 0 < len(required) <= self._history_ref_limit else None
        return _Outcome(terminal, sources, self._present(terminal, evidence), saved_refs)

    def _withdraw(
        self, record: _RequestRecord, terminal: TerminalResult, request: QueryRequest, reason: str
    ) -> _Outcome:
        # The trace keeps the selected verdict; this event records the public downgrade.
        self._emit(
            record.request_id,
            record.identity,
            "rag.evidence.withdrawn",
            attributes={
                "reason": reason,
                "selected_status": terminal.status.value,
                "strategy": terminal.strategy.value if terminal.strategy else None,
                "packed_passages": len(terminal.evidence_refs),
            },
        )
        abstention = _abstain(request.message, terminal.strategy, terminal.generation_repaired)
        return _Outcome(abstention, (), self._present(abstention, {}), ())

    async def _answer(
        self,
        record: _RequestRecord,
        request: QueryRequest,
        start: float,
    ) -> TerminalResult:
        """Run understanding and strategies inside the attempt budget, salvaging on failure."""

        attempts = asyncio.timeout_at(
            start + self._timeouts.request_seconds - self._timeouts.finalization_reserve_seconds
        )
        try:
            async with attempts:
                return await self._attempt(record, request)
        except (TimeoutError, InfrastructureError) as exc:
            if attempts.expired():
                record.error_code = _DEADLINE_CODE
                salvaged = self._partial_result(record.state, degraded=True)
                if salvaged is None:
                    raise RequestDeadlineExceededError(
                        "The RAG request exceeded its global deadline"
                    ) from exc
                return salvaged
            record.error_code = type(exc).__name__
            salvaged = self._partial_result(record.state, degraded=True)
            if salvaged is None:
                raise
            return salvaged

    async def _attempt(self, record: _RequestRecord, request: QueryRequest) -> TerminalResult:
        history, record.history_refs = await self._history(record.identity, request)
        understood = await self._query_understanding.understand(request.message, history)
        record.contextualization = understood.trace
        record.contextualization_usage = understood.usage
        record.contextualization_calls = understood.model_calls
        record.canonical_observation = self._telemetry.observe_text(
            understood.query.canonical_question
        )
        state = RagState(
            request_id=record.request_id,
            identity=record.identity,
            query=understood.query,
        )
        record.state = state
        return await self._execute(state, request)

    async def _history(
        self,
        identity: Identity,
        request: QueryRequest,
    ) -> tuple[tuple[ConversationMessage, ...], tuple[EvidenceRef, ...]]:
        if not request.conversation_id or not self._contextualization.enabled:
            return (), ()
        messages = await self._conversations.get_messages(
            identity,
            request.conversation_id,
            self._contextualization.history_messages,
        )
        return await authorized_history(
            messages,
            documents=self._documents,
            identity=identity,
            filters=request.filters,
            reference_limit=self._history_ref_limit,
            character_limit=self._contextualization.max_context_characters,
        )

    async def _append_turn(
        self, identity: Identity, request: QueryRequest, outcome: _Outcome
    ) -> None:
        if not request.conversation_id:
            return
        await self._conversations.append_messages(
            identity,
            request.conversation_id,
            (
                ConversationMessage(role="user", content=request.message),
                ConversationMessage(
                    role="assistant",
                    content=outcome.presented.history_text,
                    evidence_refs=outcome.history_refs,
                ),
            ),
        )

    def _present(self, terminal: TerminalResult, evidence: Mapping[str, Chunk]) -> RenderedAnswer:
        """Render citations first; the partial-answer disclosure is appended afterwards."""

        rendered = self._citations.render(
            terminal.answer,
            evidence=evidence,
            verified_chunk_ids=terminal.verified_chunk_ids,
        )
        if terminal.status is not ResultStatus.VERIFIED_PARTIAL:
            return rendered
        disclosure = (
            "I could not verify from the available sources: "
            + "; ".join(terminal.unanswered_requirements)
            + "."
        )
        return rendered.with_disclosure(disclosure)

    async def _execute(self, state: RagState, request: QueryRequest) -> TerminalResult:
        strategy = RetrievalStrategy.HAPPY_PATH
        previous: _Evidence | None = None
        while True:
            plan = self._strategies.get(strategy)
            state.attempted_strategies.append(strategy)
            self._emit(
                state.request_id,
                state.identity,
                "rag.attempt.started",
                attributes={"strategy": strategy.value},
            )
            started = perf_counter()
            if plan.reuse_previous_context and previous is not None:
                evidence, reused = previous, True
            else:
                evidence, reused = await self._retrieve(state, plan, request), False
            previous = evidence
            outcome = await self._run_attempt(state, plan, evidence, reused=reused, started=started)
            if isinstance(outcome, TerminalResult):
                return outcome
            next_strategy = self._decide(state, plan, evidence, outcome).next_strategy
            if next_strategy is None:
                return self._finalize(state)
            strategy = next_strategy

    async def _retrieve(
        self, state: RagState, plan: StrategyPlan, request: QueryRequest
    ) -> _Evidence:
        retrieval = await self._retriever.retrieve(
            plan=plan.retrieval,
            query=state.query,
            identity=state.identity,
            filters=request.filters,
        )
        if retrieval.metadata_mismatch_count:
            # Qdrant matched these hits, but MongoDB metadata did not; the stores have drifted.
            self._emit(
                state.request_id,
                state.identity,
                "rag.retrieval.metadata_mismatch",
                attributes={
                    "strategy": plan.kind.value,
                    "count": retrieval.metadata_mismatch_count,
                },
            )
        context = await self._context_builder.build(
            query=state.query.canonical_question,
            candidates=retrieval.ranked_chunks,
            identity=state.identity,
            filters=request.filters,
        )
        refs = {
            label: EvidenceRef.from_chunk(chunk)
            for label, chunk in context.packed_context.evidence_by_label().items()
        }
        return _Evidence(retrieval=retrieval, context=context, refs=refs)

    async def _run_attempt(
        self,
        state: RagState,
        plan: StrategyPlan,
        evidence: _Evidence,
        *,
        reused: bool,
        started: float,
    ) -> TerminalResult | EvaluationResult | AttemptOutcome:
        generation, evaluation = await self._generate_and_evaluate(state, plan, evidence)
        self._record_attempt(
            state=state,
            strategy=plan.kind,
            evidence=evidence,
            generation=generation,
            evaluation=evaluation,
            repaired=False,
            retrieval_reused=reused,
            latency_ms=_elapsed_ms(started),
        )
        if generation is None or isinstance(evaluation, AttemptOutcome):
            return evaluation
        if self._decision.passed(evaluation):
            return _complete(
                plan.kind,
                generation,
                evaluation,
                repaired=False,
                refs=evidence.refs,
            )
        self._keep_best(
            state, plan.kind, generation.draft.answer, evaluation, evidence, repaired=False
        )
        if not self._decision.can_repair(state, evaluation):
            return evaluation
        return await self._repair(state, plan, evidence, generation, evaluation)

    async def _repair(
        self,
        state: RagState,
        plan: StrategyPlan,
        evidence: _Evidence,
        generation: GenerationResult,
        evaluation: EvaluationResult,
    ) -> TerminalResult | EvaluationResult | AttemptOutcome:
        started = perf_counter()
        state.generation_repairs_used += 1
        packed = evidence.context.packed_context
        repaired = await self._generator.repair(
            query=state.query,
            previous_answer=generation.draft.answer,
            context=packed,
            faithfulness=evaluation.faithfulness,
            completeness=evaluation.completeness,
        )
        repaired_evaluation = await self._evaluate_draft(state, evidence, repaired)
        self._record_attempt(
            state=state,
            strategy=plan.kind,
            evidence=evidence,
            generation=repaired,
            evaluation=repaired_evaluation,
            repaired=True,
            retrieval_reused=True,
            latency_ms=_elapsed_ms(started),
        )
        passed = isinstance(repaired_evaluation, EvaluationResult) and self._decision.passed(
            repaired_evaluation
        )
        self._emit(
            state.request_id,
            state.identity,
            "rag.repair.completed",
            attributes={"successful": passed, "strategy": plan.kind.value},
        )
        if isinstance(repaired_evaluation, AttemptOutcome):
            return repaired_evaluation
        self._keep_best(
            state, plan.kind, repaired.draft.answer, repaired_evaluation, evidence, repaired=True
        )
        if passed:
            return _complete(
                plan.kind,
                repaired,
                repaired_evaluation,
                repaired=True,
                refs=evidence.refs,
            )
        return repaired_evaluation

    async def _generate_and_evaluate(
        self,
        state: RagState,
        plan: StrategyPlan,
        evidence: _Evidence,
    ) -> tuple[GenerationResult | None, EvaluationResult | AttemptOutcome]:
        if not evidence.available:
            if self._policy.verify_answers:
                return None, AttemptOutcome.EMPTY_EVIDENCE
            return None, self._empty_evaluation(state.query.literal_question)
        packed = evidence.context.packed_context
        generation = await self._generator.generate(
            query=state.query,
            context=packed,
            conversation_facts=_conversation_facts(state, plan),
        )
        return generation, await self._evaluate_draft(state, evidence, generation)

    async def _evaluate_draft(
        self, state: RagState, evidence: _Evidence, generation: GenerationResult
    ) -> EvaluationResult | AttemptOutcome:
        occurrences = None
        if self._policy.verify_answers:
            if not has_material_text(generation.draft.answer):
                return AttemptOutcome.EMPTY_DRAFT
            occurrences = citation_occurrences(generation.draft.answer)
            if occurrences is None:
                return AttemptOutcome.CITATION_LIMIT
        return await self._evaluator.evaluate(
            query=state.query,
            answer=generation.draft.answer,
            context=evidence.context.packed_context,
            occurrences=occurrences,
        )

    def _record_attempt(
        self,
        *,
        state: RagState,
        strategy: RetrievalStrategy,
        evidence: _Evidence,
        generation: GenerationResult | None,
        evaluation: EvaluationResult | AttemptOutcome,
        repaired: bool,
        retrieval_reused: bool,
        latency_ms: float,
    ) -> None:
        retrieval = evidence.retrieval
        context = evidence.context
        reranked = {item.chunk.id: item for item in context.reranked_chunks}
        packed_ids = {item.chunk.id for item in context.packed_context.chunks}
        candidates = tuple(
            CandidateTrace(
                chunk_id=item.chunk.id,
                document_id=item.chunk.document_id,
                dense_score=item.dense_score,
                dense_rank=item.dense_rank,
                sparse_score=item.sparse_score,
                sparse_rank=item.sparse_rank,
                fusion_score=item.fusion_score,
                fusion_rank=item.fusion_rank,
                reranker_score=reranked.get(item.chunk.id, item).reranker_score,
                reranker_rank=reranked.get(item.chunk.id, item).reranker_rank,
                used=item.chunk.id in packed_ids,
            )
            for item in retrieval.ranked_chunks
        )
        verdict = evaluation if isinstance(evaluation, EvaluationResult) else None
        usage = verdict.usage if verdict else ModelUsage()
        model_calls = verdict.model_calls if verdict else ()
        if not retrieval_reused:
            usage += retrieval.embedding_usage + retrieval.query_recovery_usage
            model_calls = retrieval.model_calls + model_calls
        if generation:
            usage += generation.usage
            model_calls = (generation.model_call, *model_calls)
        state.attempts.append(
            AttemptTrace(
                attempt_number=len(state.attempts) + 1,
                strategy=strategy,
                retrieval_queries=retrieval.queries,
                candidate_k=retrieval.candidate_k,
                generation_k=self._retrieval_config.generation_k,
                dense_result_count=retrieval.dense_result_count,
                sparse_result_count=retrieval.sparse_result_count,
                fused_result_count=retrieval.fused_result_count,
                candidates=candidates,
                packed_chunk_ids=tuple(item.chunk.id for item in context.packed_context.chunks),
                outcome=evaluation
                if isinstance(evaluation, AttemptOutcome)
                else (AttemptOutcome.EVALUATED if self._policy.verify_answers else None),
                faithfulness=verdict.faithfulness if verdict else None,
                completeness=verdict.completeness if verdict else None,
                failure_types=(),
                generation_repair_attempted=repaired,
                latency_ms=latency_ms,
                token_usage=usage,
                model_calls=model_calls,
                answer_observation=self._telemetry.observe_text(
                    generation.draft.answer if generation else ""
                ),
            )
        )
        self._emit(
            state.request_id,
            state.identity,
            "rag.attempt.completed",
            attributes={
                "strategy": strategy.value,
                "repaired": repaired,
                "latency_ms": round(latency_ms, 3),
                "faithful": verdict.faithfulness.passed if verdict else None,
                "complete": verdict.completeness.passed if verdict else None,
            },
        )

    def _decide(
        self,
        state: RagState,
        plan: StrategyPlan,
        evidence: _Evidence,
        evaluation: EvaluationResult | AttemptOutcome,
    ) -> RetryDecision:
        decision = self._decision.decide_next(
            state,
            evaluation,
            conversation_facts_used=bool(_conversation_facts(state, plan)),
            evidence_available=evidence.available,
        )
        failures = self._decision.failures(evaluation, decision=decision, passed=False)
        state.attempts[-1] = state.attempts[-1].model_copy(
            update={"decision": decision, "failure_types": failures}
        )
        next_strategy = decision.next_strategy
        self._emit(
            state.request_id,
            state.identity,
            "rag.retry.decided",
            attributes={
                "hypothesis": decision.hypothesis.value,
                "next_strategy": next_strategy.value if next_strategy is not None else None,
            },
        )
        return decision

    def _keep_best(
        self,
        state: RagState,
        strategy: RetrievalStrategy,
        answer: str,
        evaluation: EvaluationResult,
        evidence: _Evidence,
        *,
        repaired: bool,
    ) -> None:
        self._decision.preserve_best(
            state,
            answer=answer,
            evaluation=evaluation,
            strategy=strategy,
            repaired=repaired,
            evidence_refs=evidence.refs,
        )

    def _partial_result(self, state: RagState | None, *, degraded: bool) -> TerminalResult | None:
        best = state.best_verified_answer if state else None
        if best is None or not self._policy.allow_verified_partial:
            return None
        return _partial(best, degraded=degraded)

    def _finalize(self, state: RagState) -> TerminalResult:
        partial = self._partial_result(state, degraded=False)
        if partial is not None:
            return partial
        return _abstain(
            state.query.literal_question,
            state.attempted_strategies[-1] if state.attempted_strategies else None,
            bool(state.generation_repairs_used),
        )

    async def _save_trace(self, record: _RequestRecord, started: float) -> None:
        latency_ms = _elapsed_ms(started)
        state = record.state
        attempts = tuple(state.attempts) if state else ()
        terminal = record.terminal
        token_usage = record.contextualization_usage + sum(
            (attempt.token_usage for attempt in attempts), ModelUsage()
        )
        trace = RequestTrace(
            request_id=record.request_id,
            tenant_id=record.identity.tenant_id,
            user_id=record.identity.user_id,
            literal_question_observation=record.literal_observation,
            canonical_question_observation=record.canonical_observation,
            attempted_strategies=tuple(state.attempted_strategies) if state else (),
            generation_repair_used=bool(state and state.generation_repairs_used),
            attempts=attempts,
            final_status=terminal.status if terminal else None,
            final_source_ids=terminal.verified_chunk_ids if terminal else (),
            latency_ms=latency_ms,
            token_usage=token_usage,
            model_calls=record.contextualization_calls
            + tuple(call for attempt in attempts for call in attempt.model_calls),
            prompt_versions=VERIFICATION_PROMPT_VERSIONS
            if self._policy.verify_answers
            else PROMPT_VERSIONS,
            runtime_versions=self._runtime_versions,
            configuration_fingerprint=self._configuration_fingerprint,
            error_code=record.error_code,
            contextualization=record.contextualization,
            degraded=terminal.degraded if terminal else False,
        )
        await self._telemetry.save_trace(trace)
        self._emit(
            record.request_id,
            record.identity,
            "rag.request.completed",
            attributes={
                "status": terminal.status.value if terminal else "SYSTEM_FAILURE",
                "attempts": len(attempts),
                "latency_ms": round(latency_ms, 3),
                "error_code": record.error_code,
                "degraded": terminal.degraded if terminal else False,
            },
        )

    def _response(self, record: _RequestRecord, outcome: _Outcome, started: float) -> QueryResponse:
        terminal = outcome.terminal
        return QueryResponse(
            request_id=record.request_id,
            status=terminal.status,
            answer=outcome.presented.text,
            sources=outcome.sources,
            citations=outcome.presented.citations,
            unanswered_requirements=terminal.unanswered_requirements,
            metadata=QueryMetadata(
                strategy=terminal.strategy,
                generation_repaired=terminal.generation_repaired,
                attempts=len(record.state.attempts) if record.state else 0,
                latency_ms=_elapsed_ms(started),
                degraded=terminal.degraded,
            ),
        )

    def _emit(
        self,
        request_id: str,
        identity: Identity,
        event_type: str,
        *,
        attributes: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self._telemetry.emit(
            event_type,
            request_id=request_id,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
            attributes=attributes,
        )

    @staticmethod
    def _empty_evaluation(question: str) -> EvaluationResult:
        return EvaluationResult(
            faithfulness=FaithfulnessEvaluation(
                passed=False,
                failure_type=FaithfulnessStatus.NO_SUPPORTED_MATERIAL_CLAIMS,
                confidence=1,
            ),
            completeness=CompletenessEvaluation(
                passed=False,
                failure_type=CompletenessStatus.MISSING_RETRIEVAL_EVIDENCE,
                requirements=(
                    RequirementEvaluation(
                        requirement=question,
                        status=RequirementStatus.MISSING,
                        available_in_context=False,
                    ),
                ),
                missing_information_available=False,
                confidence=1,
            ),
            usage=ModelUsage(),
        )


def _error_code(error: BaseException) -> str:
    if isinstance(error, RequestDeadlineExceededError):
        return _DEADLINE_CODE
    return type(error).__name__


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


def _conversation_facts(state: RagState, plan: StrategyPlan) -> tuple[str, ...]:
    return () if plan.isolate_conversation else state.query.relevant_conversation_context


def _abstain(question: str, strategy: RetrievalStrategy | None, repaired: bool) -> TerminalResult:
    return TerminalResult(
        status=ResultStatus.INSUFFICIENT_EVIDENCE,
        answer=STANDARD_ABSTENTION,
        verified_chunk_ids=(),
        unanswered_requirements=(question,),
        strategy=strategy,
        generation_repaired=repaired,
    )


def _complete(
    strategy: RetrievalStrategy,
    generation: GenerationResult,
    evaluation: EvaluationResult,
    *,
    repaired: bool,
    refs: dict[str, EvidenceRef],
) -> TerminalResult:
    return TerminalResult(
        status=ResultStatus.COMPLETE,
        answer=generation.draft.answer,
        verified_chunk_ids=evaluation.faithfulness.verified_sources,
        unanswered_requirements=(),
        strategy=strategy,
        generation_repaired=repaired,
        evidence_refs=refs,
    )


def _partial(best: VerifiedAnswer, *, degraded: bool) -> TerminalResult:
    return TerminalResult(
        status=ResultStatus.VERIFIED_PARTIAL,
        answer=best.answer,
        verified_chunk_ids=best.verified_chunk_ids,
        unanswered_requirements=best.unanswered_requirements,
        strategy=best.strategy,
        generation_repaired=best.generation_repaired,
        evidence_refs=best.evidence_refs,
        degraded=degraded,
    )
