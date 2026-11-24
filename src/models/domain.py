"""Domain contracts shared by services and provider adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Role = Annotated[str, Field(min_length=1, max_length=128)]
FILTER_KEY_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,127}$"
FilterKey = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=FILTER_KEY_PATTERN),
]
FilterString = Annotated[str, Field(max_length=512)]
FilterInteger = Annotated[int, Field(ge=-(2**63), le=2**63 - 1)]
FilterFloat = Annotated[float, Field(allow_inf_nan=False)]
FilterValue = FilterString | FilterInteger | FilterFloat | bool


class RetrievalStrategy(StrEnum):
    HAPPY_PATH = "HAPPY_PATH"
    RECALL_EXPANSION = "RECALL_EXPANSION"
    QUERY_RECOVERY = "QUERY_RECOVERY"
    CONTEXT_ISOLATION = "CONTEXT_ISOLATION"


class ResultStatus(StrEnum):
    COMPLETE = "COMPLETE"
    VERIFIED_PARTIAL = "VERIFIED_PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FaithfulnessStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_CLAIMS = "UNSUPPORTED_CLAIMS"
    CONTRADICTED_CLAIMS = "CONTRADICTED_CLAIMS"
    NO_SUPPORTED_MATERIAL_CLAIMS = "NO_SUPPORTED_MATERIAL_CLAIMS"


class CompletenessStatus(StrEnum):
    COMPLETE = "COMPLETE"
    MISSING_ANSWER_CONTENT = "MISSING_ANSWER_CONTENT"
    MISSING_RETRIEVAL_EVIDENCE = "MISSING_RETRIEVAL_EVIDENCE"
    QUESTION_NOT_ANSWERABLE = "QUESTION_NOT_ANSWERABLE"


class RequirementStatus(StrEnum):
    ANSWERED = "answered"
    MISSING = "missing"


class RagFailure(StrEnum):
    LOW_RECALL = "LOW_RECALL"
    QUERY_DRIFT = "QUERY_DRIFT"
    GENERATION_OMISSION = "GENERATION_OMISSION"
    GENERATION_HALLUCINATION = "GENERATION_HALLUCINATION"
    CONTEXT_CONTAMINATION = "CONTEXT_CONTAMINATION"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_CORPUS = "INSUFFICIENT_CORPUS"
    ACCESS_LIMITED = "ACCESS_LIMITED"
    RERANKER_FAILURE = "RERANKER_FAILURE"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"


class Identity(FrozenModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    roles: frozenset[Role] = Field(min_length=1, max_length=64)


class ConversationMessage(FrozenModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class QueryRequest(FrozenModel):
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=16_000)
    filters: dict[FilterKey, FilterValue] = Field(default_factory=dict, max_length=32)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must contain non-whitespace content")
        return value


class Source(FrozenModel):
    document_id: str
    title: str
    sections: tuple[str, ...] = ()
    version: int


class Citation(FrozenModel):
    number: int = Field(ge=1)
    document_id: str
    title: str
    section: str | None = None
    version: int


class QueryMetadata(FrozenModel):
    strategy: RetrievalStrategy | None
    generation_repaired: bool
    attempts: int
    latency_ms: float
    degraded: bool = False


class QueryResponse(FrozenModel):
    request_id: str
    status: ResultStatus
    answer: str
    sources: tuple[Source, ...]
    citations: tuple[Citation, ...] = ()
    unanswered_requirements: tuple[str, ...] = ()
    metadata: QueryMetadata


def _metadata_scalar(value: object) -> object:
    # BSON 64-bit integers are int subclasses. BSON code is a str subclass with a distinct type.
    if isinstance(value, int) or type(value) in (str, float):
        return value
    raise ValueError("Chunk metadata values must be strings, numbers or booleans")


MetadataValue = Annotated[str | int | float | bool, BeforeValidator(_metadata_scalar)]


class Chunk(FrozenModel):
    id: str
    tenant_id: str
    document_id: str
    document_version_id: str
    document_version: int
    document_title: str
    section_id: str
    sequence: int
    heading_path: tuple[str, ...]
    text: str
    token_count: int = Field(strict=True, gt=0)
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    allowed_roles: frozenset[str]
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("token_count", mode="before")
    @classmethod
    def normalize_whole_double(cls, value: object) -> object:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def passage_fields(self) -> dict[str, object]:
        """The label-independent passage supplied to generation and evaluation."""
        return {
            "document": self.document_title,
            "document_version": self.document_version,
            "section": list(self.heading_path),
            "content": self.text,
        }

    def matches_filters(self, filters: Mapping[str, str | int | float | bool]) -> bool:
        for key, expected in filters.items():
            value = self.metadata.get(key)
            # A dotted vector path cannot be established from a flat scalar record.
            if "." in key or value is None:
                return False
            if isinstance(expected, float):
                # The vector manager uses a numeric range with equal endpoints.
                if isinstance(value, bool) or value != expected:
                    return False
            elif type(value) is not type(expected) or value != expected:
                return False
        return True


class EvidenceRef(FrozenModel):
    chunk_id: str
    document_id: str
    document_version_id: str
    passage_digest: str

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> EvidenceRef:
        passage = json.dumps(chunk.passage_fields(), sort_keys=True, separators=(",", ":"))
        return cls(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            passage_digest=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        )


class SearchHit(FrozenModel):
    chunk_id: str
    score: float
    rank: int


class RetrievalPlan(FrozenModel):
    """Retrieval breadth for one strategy; zero values mean "reuse previous evidence"."""

    dense_k: int = Field(ge=0)
    sparse_k: int = Field(ge=0)
    candidate_k: int = Field(ge=0)
    recover_queries: bool = False


class RankedChunk(FrozenModel):
    chunk: Chunk
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fusion_score: float
    fusion_rank: int
    reranker_score: float | None = None
    reranker_rank: int | None = None
    is_neighbor: bool = False
    anchor_chunk_id: str | None = None


def evidence_label(position: int) -> str:
    """Return the prompt-facing label for the 1-based evidence position."""

    return f"E{position}"


class PackedContext(FrozenModel):
    chunks: tuple[RankedChunk, ...]
    token_count: int

    def evidence_by_label(self) -> dict[str, Chunk]:
        return {
            evidence_label(position): item.chunk
            for position, item in enumerate(self.chunks, start=1)
        }


class QueryUnderstanding(FrozenModel):
    literal_question: str
    canonical_question: str
    rewritten: bool
    relevant_conversation_context: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()


class ContextualizationOutput(FrozenModel):
    standalone_question: str
    relevant_conversation_context: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()


class MessageRelevance(FrozenModel):
    index: int = Field(ge=0)
    probability: float = Field(ge=0, le=1)


class FollowupClassification(FrozenModel):
    """Structured output requested from a follow-up decision model."""

    is_followup_probability: float = Field(ge=0, le=1)
    message_relevance: tuple[MessageRelevance, ...] = ()


class QueryRecoveryOutput(FrozenModel):
    keyword_query: str | None = None
    decomposed_queries: tuple[str, ...] = ()


class GenerationDraft(FrozenModel):
    answer: str


class UnsupportedClaim(FrozenModel):
    claim: str


class ContradictedClaim(FrozenModel):
    claim: str
    evidence_chunk_ids: tuple[str, ...] = ()


class FaithfulnessEvaluation(FrozenModel):
    passed: bool
    failure_type: FaithfulnessStatus
    verified_sources: tuple[str, ...] = ()
    unsupported_claims: tuple[UnsupportedClaim, ...] = ()
    contradicted_claims: tuple[ContradictedClaim, ...] = ()
    correcting_evidence_available: bool = False
    correcting_chunks: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def passed_follows_status(cls, data: Any) -> Any:
        return _passed_from_status(data, FaithfulnessStatus.SUPPORTED)


class RequirementEvaluation(FrozenModel):
    requirement: str
    status: RequirementStatus
    available_in_context: bool = False
    supporting_chunks: tuple[str, ...] = ()


class CompletenessEvaluation(FrozenModel):
    passed: bool
    failure_type: CompletenessStatus
    requirements: tuple[RequirementEvaluation, ...] = ()
    missing_information_available: bool = False
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def passed_follows_status(cls, data: Any) -> Any:
        return _passed_from_status(data, CompletenessStatus.COMPLETE)


def _passed_from_status(data: Any, passing: StrEnum) -> Any:
    """Derive ``passed`` from ``failure_type`` so a judge's flag can never disagree with it.

    The strict output schema cannot tie the two fields together, so the status wins.
    """

    if not isinstance(data, dict) or "failure_type" not in data:
        return data
    return {**data, "passed": data["failure_type"] == passing.value}


class ModelUsage(FrozenModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelResult[SchemaT: BaseModel]:
    value: SchemaT
    model: str
    provider: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vector: list[float]
    model: str
    provider: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    request_id: str | None = None
    cached: bool = False


class ModelCallTrace(FrozenModel):
    operation: str
    provider: str
    model: str
    request_id: str | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)


class SparseEmbedding(FrozenModel):
    """Provider-neutral sparse vector with strictly increasing uint32 indices."""

    indices: tuple[int, ...] = ()
    values: tuple[float, ...] = ()

    @model_validator(mode="after")
    def aligned_sorted_indices(self) -> SparseEmbedding:
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse indices and values must have equal length")
        if any(index < 0 or index > 4_294_967_295 for index in self.indices):
            raise ValueError("Sparse indices must fit in uint32")
        if any(right <= left for left, right in pairwise(self.indices)):
            raise ValueError("Sparse indices must be strictly increasing")
        return self


class FollowupDecision(FrozenModel):
    detector: str
    is_followup_probability: float = Field(ge=0, le=1)
    message_scores: tuple[float, ...] = ()  # aligned with supplied history, oldest first
    model_call: ModelCallTrace | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)


@dataclass(frozen=True, slots=True)
class EncodedQuery:
    text: str
    dense: tuple[float, ...]
    sparse: SparseEmbedding
    usage: ModelUsage = field(default_factory=ModelUsage)
    model_calls: tuple[ModelCallTrace, ...] = ()


@dataclass(frozen=True, slots=True)
class HybridHits:
    dense: tuple[SearchHit, ...] = ()
    sparse: tuple[SearchHit, ...] = ()


class CandidateTrace(FrozenModel):
    chunk_id: str
    document_id: str
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fusion_score: float
    fusion_rank: int
    reranker_score: float | None = None
    reranker_rank: int | None = None
    used: bool


class RetryHypothesis(StrEnum):
    CONTEXT_CONTAMINATION = "CONTEXT_CONTAMINATION"
    QUERY_DRIFT = "QUERY_DRIFT"
    NOT_ANSWERABLE = "NOT_ANSWERABLE"
    LOW_RECALL = "LOW_RECALL"
    GENERATION_ONLY = "GENERATION_ONLY"
    UNCLASSIFIED = "UNCLASSIFIED"


class RetryDecision(FrozenModel):
    hypothesis: RetryHypothesis
    next_strategy: RetrievalStrategy | None
    drift_score: float = Field(ge=0, le=1)
    reason: str


class AttemptTrace(FrozenModel):
    attempt_number: int
    strategy: RetrievalStrategy
    retrieval_queries: tuple[str, ...]
    candidate_k: int
    generation_k: int
    dense_result_count: int
    sparse_result_count: int
    fused_result_count: int
    candidates: tuple[CandidateTrace, ...]
    packed_chunk_ids: tuple[str, ...]
    faithfulness: FaithfulnessEvaluation
    completeness: CompletenessEvaluation
    failure_types: tuple[RagFailure, ...]
    generation_repair_attempted: bool
    latency_ms: float
    token_usage: ModelUsage
    model_calls: tuple[ModelCallTrace, ...] = ()
    answer_observation: dict[str, Any]
    decision: RetryDecision | None = None


class ContextualizationTrace(FrozenModel):
    enabled: bool
    detector: str | None = None
    followup_probability: float | None = None
    fail_open: bool = False
    history_messages: int = Field(default=0, ge=0)
    selected_messages: int = Field(default=0, ge=0)
    rewritten: bool = False


class RequestTrace(FrozenModel):
    request_id: str
    tenant_id: str
    user_id: str
    literal_question_observation: dict[str, Any]
    canonical_question_observation: dict[str, Any]
    attempted_strategies: tuple[RetrievalStrategy, ...]
    generation_repair_used: bool
    attempts: tuple[AttemptTrace, ...]
    final_status: ResultStatus | None
    final_source_ids: tuple[str, ...]
    latency_ms: float
    token_usage: ModelUsage
    model_calls: tuple[ModelCallTrace, ...] = ()
    prompt_versions: dict[str, str]
    runtime_versions: dict[str, str]
    configuration_fingerprint: str
    error_code: str | None = None
    contextualization: ContextualizationTrace | None = None
    degraded: bool = False
    created_at: datetime = Field(default_factory=utc_now)


_MAX_TELEMETRY_TEXT = 256


class TelemetryEvent(FrozenModel):
    """Central low-cardinality event persisted by the telemetry pipeline."""

    event_type: str = Field(min_length=1, max_length=128)
    request_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("attributes")
    @classmethod
    def truncate_long_text(
        cls,
        value: dict[str, str | int | float | bool | None],
    ) -> dict[str, str | int | float | bool | None]:
        return {
            key: item[:_MAX_TELEMETRY_TEXT] if isinstance(item, str) else item
            for key, item in value.items()
        }


@dataclass(slots=True)
class VerifiedAnswer:
    answer: str
    verified_chunk_ids: tuple[str, ...]
    unanswered_requirements: tuple[str, ...]
    strategy: RetrievalStrategy
    generation_repaired: bool
    answered_requirements: int = 0
    evidence_refs: dict[str, EvidenceRef] = field(default_factory=dict)


@dataclass(slots=True)
class RagState:
    request_id: str
    identity: Identity
    query: QueryUnderstanding
    attempted_strategies: list[RetrievalStrategy] = field(default_factory=list)
    generation_repairs_used: int = 0
    attempts: list[AttemptTrace] = field(default_factory=list)
    best_verified_answer: VerifiedAnswer | None = None


class GoldenQuestion(FrozenModel):
    id: str
    question: str
    conversation_history: tuple[ConversationMessage, ...] = ()
    expected_answer_facts: tuple[str, ...] = ()
    acceptable_document_ids: frozenset[str] = frozenset()
    required_document_ids: frozenset[str] = frozenset()
    answerable: bool
    categories: frozenset[str]
    tenant_id: str
    roles: frozenset[str]
