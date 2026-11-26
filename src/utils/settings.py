"""Load stable YAML configuration and environment-owned external values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote_plus, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymongo.errors import PyMongoError
from pymongo.uri_parser_shared import split_options

from utils.errors import ConfigurationError


class Environment(StrEnum):
    """Supported deployment environments."""

    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class FusionStrategy(StrEnum):
    RRF = "rrf"
    WEIGHTED = "weighted"


class ContentMode(StrEnum):
    FULL = "full"
    REDACTED = "redacted"
    METADATA_ONLY = "metadata_only"


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _default_reasoning_effort() -> dict[str, ReasoningEffort]:
    return {"gpt-5.6-luna": "low"}


class ConfigModel(BaseModel):
    """Reject misspelled or obsolete keys in committed YAML configuration."""

    model_config = ConfigDict(extra="forbid")


class AppConfig(ConfigModel):
    host: str = "0.0.0.0"  # noqa: S104
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class AuthConfig(ConfigModel):
    mode: str = "trusted_headers"
    tenant_header: str = "X-Tenant-ID"
    user_header: str = "X-User-ID"
    roles_header: str = "X-Roles"
    token_header: str = "X-Auth-Token"  # noqa: S105
    debug_token_header: str = "X-Debug-Token"  # noqa: S105
    admin_role: str = "administrator"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value != "trusted_headers":
            raise ValueError("Only trusted_headers authentication is implemented")
        return value


class MongoConfig(ConfigModel):
    provider: str = "mongodb"
    database: str = "adaptive_rag"
    trace_collection: str = "rag_request_traces"
    telemetry_collection: str = "telemetry_events"
    trace_retention_days: int = Field(default=30, ge=1)
    telemetry_retention_days: int = Field(default=30, ge=1)
    server_selection_timeout_ms: int = Field(default=3000, ge=100)
    operation_timeout_ms: int = Field(default=5000, ge=100)

    @model_validator(mode="after")
    def keep_observability_collections_separate(self) -> MongoConfig:
        application_collections = {
            "documents",
            "document_versions",
            "chunks",
            "conversations",
            "system_state",
        }
        configured = {self.trace_collection, self.telemetry_collection}
        if len(configured) != 2 or configured & application_collections:
            raise ValueError(
                "Trace and telemetry collection names must be distinct from each other "
                "and from application collections"
            )
        return self


class AIConfig(ConfigModel):
    provider: str = "openai"
    generation_model: str = "gpt-5.6"
    contextualization_model: str = "gpt-5.6-luna"
    query_recovery_model: str = "gpt-5.6-luna"
    evaluation_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=1)
    max_output_tokens: int = Field(default=4096, ge=128)
    # Reasoning effort per model name; models not listed use the provider default.
    reasoning_effort: dict[str, ReasoningEffort] = Field(default_factory=_default_reasoning_effort)


class SparseEncoderName(StrEnum):
    BM25 = "bm25"


class SparseConfig(ConfigModel):
    encoder: SparseEncoderName = SparseEncoderName.BM25
    k1: float = Field(default=1.2, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    average_document_tokens: float = Field(default=256, gt=0)
    hash_space: int = Field(default=2_147_483_647, ge=1_000_003, le=4_294_967_295)
    language: Literal["english"] = "english"


class VectorStoreConfig(ConfigModel):
    provider: str = "qdrant"
    collection_alias: str = Field(default="adaptive_rag_chunks", pattern=r"^[A-Za-z0-9_-]{1,100}$")
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    sparse: SparseConfig = Field(default_factory=SparseConfig)


class RerankerConfig(ConfigModel):
    provider: str = "cohere"
    model: str = "rerank-v4.0-pro"


class RetrievalConfig(ConfigModel):
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    dense_weight: float = Field(default=0.6, ge=0)
    sparse_weight: float = Field(default=0.4, ge=0)
    rrf_k: int = Field(default=60, ge=1)
    happy_dense_k: int = Field(default=40, ge=1)
    happy_sparse_k: int = Field(default=40, ge=1)
    happy_candidate_k: int = Field(default=30, ge=1)
    recall_dense_k: int = Field(default=100, ge=1)
    recall_sparse_k: int = Field(default=100, ge=1)
    recall_candidate_k: int = Field(default=80, ge=1)
    generation_k: int = Field(default=8, ge=1)
    max_query_variants: int = Field(default=4, ge=1, le=10)
    max_decomposed_queries: int = Field(default=3, ge=0, le=10)
    query_variant_concurrency: int = Field(default=4, ge=1, le=20)
    query_drift_threshold: float = Field(default=0.5, ge=0, le=1)
    embedding_cache_entries: int = Field(default=1000, ge=0)

    @model_validator(mode="after")
    def validate_weights_and_limits(self) -> RetrievalConfig:
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("At least one fusion weight must be positive")
        if self.generation_k > self.recall_candidate_k:
            raise ValueError("generation_k cannot exceed recall_candidate_k")
        return self


class FollowupDetectorName(StrEnum):
    MODEL = "model"
    HEURISTIC = "heuristic"
    ALWAYS = "always"


class ContextualizationConfig(ConfigModel):
    enabled: bool = True
    write_history_provenance: bool = Field(default=True, strict=True)
    detector: FollowupDetectorName = FollowupDetectorName.MODEL
    detector_model: str = "gpt-5.6-luna"  # stand-in until the jev adapter exists
    history_messages: int = Field(default=30, ge=1, le=100)
    followup_threshold: float = Field(default=0.2, ge=0, le=1)
    message_threshold: float = Field(default=0.3, ge=0, le=1)
    min_recent_messages: int = Field(default=2, ge=0, le=20)
    max_context_characters: int = Field(default=12_000, ge=500)


class TokenizerEncoding(StrEnum):
    UTF8_BYTES = "utf8_bytes"
    O200K_BASE = "o200k_base"


class ContextConfig(ConfigModel):
    tokenizer_encoding: TokenizerEncoding = TokenizerEncoding.UTF8_BYTES
    tokenizer_verified_for: str | None = None
    max_tokens: int = Field(default=8000, ge=256)
    max_seed_chunks: int = Field(default=8, ge=1)
    max_document_share: float = Field(default=0.4, gt=0, le=1)
    max_neighbors_per_seed: int = Field(default=1, ge=0, le=4)
    short_chunk_tokens: int = Field(default=40, ge=0)


class PolicyConfig(ConfigModel):
    verify_answers: bool = Field(default=True, strict=True)
    max_generation_repairs: int = Field(default=1, ge=0, le=1)
    max_strategies: int = Field(default=4, ge=1, le=4)
    allow_verified_partial: bool = True
    allow_abstention: bool = True

    @model_validator(mode="after")
    def preserve_safe_terminal(self) -> PolicyConfig:
        if not self.allow_abstention:
            raise ValueError(
                "Abstention cannot be disabled because it is the final safety boundary"
            )
        return self


class TimeoutConfig(ConfigModel):
    request_seconds: float = Field(default=60, gt=0)
    finalization_reserve_seconds: float = Field(default=5, gt=0)
    followup_detection_seconds: float = Field(default=2, gt=0)
    contextualization_seconds: float = Field(default=10, gt=0)
    query_recovery_seconds: float = Field(default=10, gt=0)
    embedding_seconds: float = Field(default=2, gt=0)
    retrieval_seconds: float = Field(default=3, gt=0)
    reranking_seconds: float = Field(default=3, gt=0)
    generation_seconds: float = Field(default=20, gt=0)
    evaluation_seconds: float = Field(default=10, gt=0)

    @model_validator(mode="after")
    def reserve_finalization_time(self) -> TimeoutConfig:
        if self.finalization_reserve_seconds >= self.request_seconds:
            raise ValueError("finalization_reserve_seconds must be below request_seconds")
        return self


class RetryConfig(ConfigModel):
    """One retry of a transient provider failure, made only when the time left can hold it."""

    max_retries: int = Field(default=1, ge=0, le=1)
    backoff_seconds: float = Field(default=0.25, ge=0, allow_inf_nan=False)
    min_retry_seconds: float = Field(default=1, ge=0, allow_inf_nan=False)


class TelemetryConfig(ConfigModel):
    enabled: bool = True
    write_query_intent_fields: bool = Field(default=False, strict=True)
    write_accounting_fields: bool = Field(default=False, strict=True)
    content_mode: ContentMode = ContentMode.REDACTED
    queue_capacity: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=100, ge=1, le=1000)
    shutdown_timeout_seconds: float = Field(default=5, gt=0)
    direct_write_timeout_seconds: float = Field(default=1, gt=0)


class AnswersConfig(ConfigModel):
    inline_citations: bool = True


class ExternalSettings(BaseSettings):
    """Values that vary by deployment or must not be committed."""

    # Rejected values here can be credentials or credentialed URLs, so errors never echo input.
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore", hide_input_in_errors=True
    )

    app_env: Environment = Environment.PRODUCTION
    config_dir: Path = Path("config")
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    cohere_api_key: SecretStr | None = None
    cohere_base_url: str = "https://api.cohere.com"
    mongodb_uri: SecretStr = SecretStr("mongodb://localhost:27017")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    auth_shared_token: SecretStr = SecretStr("replace-this-auth-token")
    debug_token: SecretStr = SecretStr("replace-this-debug-token")
    telemetry_hash_key: SecretStr = SecretStr("replace-this-telemetry-key")

    @field_validator(
        "openai_api_key",
        "openai_base_url",
        "cohere_api_key",
        "qdrant_api_key",
        mode="before",
    )
    @classmethod
    def empty_optional_values_are_unset(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("openai_base_url", "cohere_base_url", "qdrant_url")
    @classmethod
    def validate_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Provider endpoints must be absolute HTTP(S) URLs")
        return value.rstrip("/")

    @field_validator("mongodb_uri")
    @classmethod
    def validate_mongodb_uri(cls, value: SecretStr) -> SecretStr:
        uri = value.get_secret_value()
        if not uri.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("MONGODB_URI must use mongodb:// or mongodb+srv://")
        mongodb_transport(uri)
        return value


class Settings(ConfigModel):
    """Validated application settings resolved before the service graph."""

    # Model-level errors would otherwise print the whole input, including the external values.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    app: AppConfig = Field(default_factory=AppConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    mongodb: MongoConfig = Field(default_factory=MongoConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    contextualization: ContextualizationConfig = Field(default_factory=ContextualizationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    answers: AnswersConfig = Field(default_factory=AnswersConfig)
    external: ExternalSettings = Field(default_factory=ExternalSettings)

    @property
    def history_ref_limit(self) -> int:
        """Maximum distinct dependencies in one history window or saved turn."""
        return (
            self.contextualization.history_messages
            * self.context.max_seed_chunks
            * (1 + self.context.max_neighbors_per_seed)
        )

    @model_validator(mode="after")
    def validate_tokenizer_for_generation_model(self) -> Settings:
        verified_for = self.context.tokenizer_verified_for
        if self.context.tokenizer_encoding is TokenizerEncoding.UTF8_BYTES:
            if verified_for is not None:
                raise ValueError("context.tokenizer_verified_for must be null in utf8_bytes mode")
        elif verified_for != self.ai.generation_model or not self.ai.generation_model.strip():
            raise ValueError(
                "context.tokenizer_verified_for must exactly match ai.generation_model "
                "and must be nonblank in tokenizer mode"
            )
        return self

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> Settings:
        if self.ai.provider == "openai" and not self.external.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        if self.reranker.provider == "cohere" and not self.external.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required for the Cohere provider")
        if self.external.app_env is Environment.PRODUCTION:
            _validate_production_secrets(self.external)
            _validate_production_transport(self.external)
        return self


_PRODUCTION_PLACEHOLDERS = frozenset(
    {
        "replace-this-auth-token",
        "replace-this-debug-token",
        "replace-this-telemetry-key",
    }
)
_MIN_PRODUCTION_SECRET_LENGTH = 32


def _validate_production_secrets(external: ExternalSettings) -> None:
    """Reject placeholder, short, or reused secrets; each one guards a separate access path."""

    secrets = [
        external.auth_shared_token.get_secret_value(),
        external.debug_token.get_secret_value(),
        external.telemetry_hash_key.get_secret_value(),
    ]
    if any(secret in _PRODUCTION_PLACEHOLDERS for secret in secrets):
        raise ValueError("Production authentication, debug, and telemetry keys must be changed")
    if any(len(secret) < _MIN_PRODUCTION_SECRET_LENGTH for secret in secrets):
        raise ValueError(
            "Production authentication, debug, and telemetry keys must be at least "
            f"{_MIN_PRODUCTION_SECRET_LENGTH} characters"
        )
    if len(set(secrets)) != len(secrets):
        raise ValueError("Production authentication, debug, and telemetry keys must be distinct")


def _validate_production_transport(external: ExternalSettings) -> None:
    """An endpoint that receives a credential must be reached over an encrypted transport."""

    credentialed_endpoints = (
        (external.openai_base_url, external.openai_api_key),
        (external.cohere_base_url, external.cohere_api_key),
        (external.qdrant_url, external.qdrant_api_key),
    )
    if any(
        url is not None and urlsplit(url).scheme != "https"
        for url, credential in credentialed_endpoints
        if credential is not None
    ):
        raise ValueError("Provider endpoints that receive API keys must use HTTPS in production")
    transport = mongodb_transport(external.mongodb_uri.get_secret_value())
    if transport.credentialed and not transport.encrypted:
        raise ValueError("MongoDB URIs that carry credentials must use TLS in production")


@dataclass(frozen=True, slots=True)
class MongoTransport:
    """Whether a MongoDB URI sends credentials and whether it explicitly requests TLS."""

    credentialed: bool
    encrypted: bool


# Options whose repetition would let one spelling silently override another. The driver keeps the
# last value and only warns, so a duplicate is rejected before its options are collapsed.
_MONGODB_SECURITY_OPTIONS = frozenset(
    {
        "tls",
        "ssl",
        "tlsinsecure",
        "tlsallowinvalidcertificates",
        "tlsallowinvalidhostnames",
        "tlscafile",
        "tlscrlfile",
        "tlscertificatekeyfile",
        "tlscertificatekeyfilepassword",
        "tlsdisablecertificaterevocationcheck",
        "tlsdisableocspendpointcheck",
        "authmechanism",
        "authmechanismproperties",
        "authsource",
    }
)
_INVALID_MONGODB_OPTIONS = "MONGODB_URI has invalid, repeated or conflicting connection options"


def mongodb_transport(uri: str) -> MongoTransport:
    """Resolve whether a MongoDB URI carries credentials and requests TLS.

    This follows PyMongo's own parsing: option names are case-insensitive, ``ssl`` is an alias
    of ``tls``, conflicting ``tls``/``ssl`` values are an error, and a ``mongodb+srv`` URI
    defaults to TLS unless it sets ``tls`` or ``ssl`` itself (DNS TXT records cannot set them).
    Only ``tls``, ``ssl`` and the SRV default count as requesting TLS, although the driver also
    enables it for options such as ``tlsCAFile``. No hostname is resolved and no client is
    built; the driver's option validators only check that a named TLS file can be opened.
    Errors never include the URI.
    """

    srv = uri.startswith("mongodb+srv://")
    scheme_free = uri.split("://", 1)[1] if "://" in uri else uri
    location, _, query = scheme_free.partition("?")
    hosts = location.partition("/")[0]
    options: Any = {}
    if query:
        _reject_repeated_security_options(query)
        try:
            options = split_options(query)
        except (PyMongoError, ValueError, TypeError, OSError):
            # The driver's messages can quote option values, and a missing TLS file raises an
            # OSError naming its path, so the cause is not chained.
            raise ValueError(_INVALID_MONGODB_OPTIONS) from None
    credentialed = "@" in hosts or "authmechanism" in options
    tls = options.get("tls")
    return MongoTransport(credentialed=credentialed, encrypted=srv if tls is None else tls is True)


def _reject_repeated_security_options(query: str) -> None:
    seen: set[str] = set()
    for item in query.replace(";", "&").split("&"):
        name = unquote_plus(item.partition("=")[0]).lower()
        if name not in _MONGODB_SECURITY_OPTIONS:
            continue
        if name in seen:
            raise ValueError(_INVALID_MONGODB_OPTIONS)
        seen.add(name)


def load_settings(external: ExternalSettings | None = None) -> Settings:
    """Load ``default.yaml`` and the optional local overlay, then validate once."""

    resolved_external = external or ExternalSettings()
    default_path = resolved_external.config_dir / "default.yaml"
    values = _read_yaml(default_path)
    if resolved_external.app_env is Environment.LOCAL:
        local_path = resolved_external.config_dir / "local.yaml"
        if local_path.exists():
            values = _deep_merge(values, _read_yaml(local_path))
    values["external"] = resolved_external
    return Settings.model_validate(values)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML configuration: {path}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in overlay.items():
        current = merged.get(key)
        merged[key] = (
            _deep_merge(current, value)
            if isinstance(current, dict) and isinstance(value, dict)
            else value
        )
    return merged
