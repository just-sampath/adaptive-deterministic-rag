from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from providers.mongodb import build_mongo_client
from utils.settings import (
    AppConfig,
    Environment,
    ExternalSettings,
    FollowupDetectorName,
    MongoConfig,
    Settings,
    TimeoutConfig,
    VectorStoreConfig,
    load_settings,
    mongodb_transport,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_CONFIG = _PROJECT_ROOT / "config"


def test_local_yaml_overlays_stable_defaults(tmp_path: Path) -> None:
    (tmp_path / "default.yaml").write_text(
        "ai:\n  provider: fake\nreranker:\n  provider: fake\napp:\n  log_level: INFO\n"
    )
    (tmp_path / "local.yaml").write_text("app:\n  log_level: DEBUG\n")
    external = ExternalSettings(
        _env_file=None,
        app_env=Environment.LOCAL,
        config_dir=tmp_path,
    )

    settings = load_settings(external)

    assert settings.app.log_level == "DEBUG"
    assert settings.retrieval.happy_dense_k == 40


def test_committed_config_enables_history_and_verification_everywhere() -> None:
    def committed(environment: Environment) -> Settings:
        return load_settings(
            ExternalSettings(
                _env_file=None,
                app_env=environment,
                config_dir=_PROJECT_CONFIG,
                openai_api_key="test-key",
                cohere_api_key="test-key",
            )
        )

    for environment in (Environment.LOCAL, Environment.TEST):
        assert committed(environment).contextualization.write_history_provenance is True
        assert committed(environment).policy.verify_answers is True
    assert committed(Environment.LOCAL).telemetry.write_accounting_fields is True
    assert committed(Environment.TEST).telemetry.write_accounting_fields is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_trace_accounting_flag_is_strict(
    make_settings: Callable[..., Settings], value: object
) -> None:
    assert make_settings(
        telemetry={"write_accounting_fields": True}
    ).telemetry.write_accounting_fields
    with pytest.raises(ValidationError, match="bool_type"):
        make_settings(telemetry={"write_accounting_fields": value})


@pytest.mark.parametrize("value", ["true", 1, None])
def test_answer_verification_flag_is_strict(
    make_settings: Callable[..., Settings], value: object
) -> None:
    with pytest.raises(ValidationError, match="bool_type"):
        make_settings(policy={"verify_answers": value})


def test_committed_default_yaml_matches_code_defaults() -> None:
    values = yaml.safe_load((_PROJECT_CONFIG / "default.yaml").read_text(encoding="utf-8"))

    for section, configured in values.items():
        default = Settings.model_fields[section].get_default(call_default_factory=True)
        assert default.model_dump(mode="json") == configured, section


def test_environment_loader_does_not_use_legacy_rag_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_OPENAI_API_KEY", "legacy-key")

    external = ExternalSettings(_env_file=None)

    assert external.openai_api_key is None


def test_blank_optional_secrets_are_treated_as_unset() -> None:
    external = ExternalSettings(
        _env_file=None,
        openai_api_key="",
        cohere_api_key="",
        qdrant_api_key="",
    )

    assert external.openai_api_key is None
    assert external.cohere_api_key is None
    assert external.qdrant_api_key is None


def test_external_provider_endpoint_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="absolute HTTP"):
        ExternalSettings(_env_file=None, qdrant_url="qdrant:6333")


def test_external_settings_fail_closed_to_production() -> None:
    assert ExternalSettings(_env_file=None).app_env is Environment.PRODUCTION


def test_settings_reject_disabled_abstention(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ValidationError, match="Abstention cannot be disabled"):
        make_settings(policy={"allow_abstention": False})


def test_settings_reject_zero_fusion_weights(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ValidationError, match="fusion weight"):
        make_settings(retrieval={"dense_weight": 0, "sparse_weight": 0})


def test_settings_reject_unknown_stable_configuration_key(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        make_settings(retrieval={"candidate_typo": 50})


def test_history_provenance_is_on_by_default_and_can_be_disabled(
    make_settings: Callable[..., Settings],
) -> None:
    assert make_settings().contextualization.write_history_provenance is True
    assert make_settings().policy.verify_answers is True
    settings = make_settings(
        contextualization={"write_history_provenance": False, "history_messages": 3},
        context={"max_seed_chunks": 2, "max_neighbors_per_seed": 2},
    )
    assert settings.contextualization.write_history_provenance is False
    assert settings.history_ref_limit == 18
    assert make_settings(policy={"verify_answers": False}).policy.verify_answers is False


@pytest.mark.parametrize("value", ["true", "false", 1, None])
def test_history_provenance_write_flag_rejects_non_booleans(
    make_settings: Callable[..., Settings], value: object
) -> None:
    with pytest.raises(ValidationError, match="bool_type"):
        make_settings(contextualization={"write_history_provenance": value})


def test_query_intent_trace_writes_require_opt_in(make_settings: Callable[..., Settings]) -> None:
    assert make_settings().telemetry.write_query_intent_fields is False
    settings = make_settings(telemetry={"write_query_intent_fields": True})
    assert settings.telemetry.write_query_intent_fields is True


@pytest.mark.parametrize("value", ["true", "false", 1, None])
def test_query_intent_trace_write_flag_rejects_non_booleans(
    make_settings: Callable[..., Settings], value: object
) -> None:
    with pytest.raises(ValidationError, match="bool_type"):
        make_settings(telemetry={"write_query_intent_fields": value})


def test_unverified_deployment_defaults_to_byte_counting(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_settings().context

    assert context.tokenizer_encoding == "utf8_bytes"
    assert context.tokenizer_verified_for is None


def test_tokenizer_accepts_explicit_verification_for_exact_generation_model(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(
        ai={"provider": "fake", "generation_model": "deployment-v2"},
        context={"tokenizer_encoding": "o200k_base", "tokenizer_verified_for": "deployment-v2"},
    )

    assert settings.context.tokenizer_encoding == "o200k_base"
    assert settings.context.tokenizer_verified_for == "deployment-v2"


@pytest.mark.parametrize(
    ("generation_model", "verified_for"),
    [("deployment-v2", None), ("   ", "   "), ("deployment-v2", "deployment"), ("v2", "v2 ")],
)
def test_tokenizer_requires_nonblank_verification_matching_the_generation_model(
    make_settings: Callable[..., Settings], generation_model: str, verified_for: str | None
) -> None:
    with pytest.raises(ValidationError, match=r"must exactly match ai\.generation_model"):
        make_settings(
            ai={"provider": "fake", "generation_model": generation_model},
            context={"tokenizer_encoding": "o200k_base", "tokenizer_verified_for": verified_for},
        )


def test_byte_counting_rejects_leftover_tokenizer_verification(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ValidationError, match="tokenizer_verified_for must be null"):
        make_settings(context={"tokenizer_verified_for": "deployment-v2"})


def test_settings_reject_unknown_tokenizer_encoding(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_settings(context={"tokenizer_encoding": "unknown"})

    assert exc_info.value.errors()[0]["type"] == "enum"


def test_settings_reject_collection_name_collisions(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ValidationError, match="must be distinct"):
        make_settings(mongodb={"trace_collection": "chunks"})


def test_production_rejects_placeholder_telemetry_key() -> None:
    with pytest.raises(ValidationError, match="telemetry keys must be changed"):
        Settings(
            ai={"provider": "fake"},
            reranker={"provider": "fake"},
            external=ExternalSettings(
                _env_file=None,
                app_env=Environment.PRODUCTION,
                auth_shared_token="production-auth-token",
                debug_token="production-debug-token",
            ),
        )


_AUTH_TOKEN = "auth-token-9f1c4e7b2d8a6f03c5e9b1d7a4f2e8c6"
_DEBUG_TOKEN = "debug-token-3a7e9c1f5b2d8e4a6c0f9b3d7e1a5c28"
_TELEMETRY_KEY = "telemetry-key-6d2b8f4a1e9c7b3d5f0a8e2c6b4d9f17"


def _production(
    *,
    auth_shared_token: str = _AUTH_TOKEN,
    debug_token: str = _DEBUG_TOKEN,
    telemetry_hash_key: str = _TELEMETRY_KEY,
    **endpoints: Any,
) -> Settings:
    return Settings(
        ai={"provider": "fake"},
        reranker={"provider": "fake"},
        external=ExternalSettings(
            _env_file=None,
            app_env=Environment.PRODUCTION,
            auth_shared_token=auth_shared_token,
            debug_token=debug_token,
            telemetry_hash_key=telemetry_hash_key,
            **endpoints,
        ),
    )


def test_production_accepts_rotated_telemetry_key() -> None:
    settings = _production()

    assert settings.external.telemetry_hash_key.get_secret_value() == _TELEMETRY_KEY


@pytest.mark.parametrize(
    "endpoints",
    [
        {"openai_api_key": "sk-test", "openai_base_url": "http://llm-gateway.internal/v1"},
        {"cohere_api_key": "co-test", "cohere_base_url": "http://reranker.internal:8080"},
        {"qdrant_api_key": "qdrant-test", "qdrant_url": "http://qdrant.internal:6333"},
    ],
)
def test_production_rejects_plain_http_endpoints_that_receive_api_keys(
    endpoints: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="must use HTTPS"):
        _production(**endpoints)


@pytest.mark.parametrize(
    "uri",
    [
        "mongodb://appuser:secret@db.internal:27017/adaptive_rag",
        "mongodb://appuser:secret@db.internal:27017/adaptive_rag?tls=false",
        "mongodb://appuser@db.internal:27017/adaptive_rag",
        # An explicit opt-out wins over the SRV default, in either spelling and any case.
        "mongodb+srv://appuser:secret@cluster.example.com/adaptive_rag?tls=false",
        "mongodb+srv://appuser:secret@cluster.example.com/adaptive_rag?SSL=false",
        # A mechanism that sends a credential needs TLS even without user information.
        "mongodb://db.internal:27017/?authMechanism=MONGODB-OIDC",
    ],
)
def test_production_rejects_mongodb_credentials_without_tls(uri: str) -> None:
    with pytest.raises(ValidationError, match="MongoDB URIs that carry credentials must use TLS"):
        _production(mongodb_uri=uri)


@pytest.mark.parametrize(
    "uri",
    [
        "mongodb+srv://appuser:secret@cluster.example.com/adaptive_rag",
        "mongodb+srv://appuser:secret@cluster.example.com/adaptive_rag?tls=true&appName=rag",
        "mongodb://appuser:secret@db.example.com:27017/adaptive_rag?tls=true",
        "mongodb://appuser:secret@db.example.com:27017/adaptive_rag?SSL=true",
        "mongodb://appuser:secret@db.example.com:27017/adaptive_rag?tls=true&ssl=true",
        "mongodb://appuser:secret@db1.example.com,db2.example.com/?tls=true;replicaSet=rs0",
    ],
)
def test_production_accepts_mongodb_credentials_only_over_tls(uri: str) -> None:
    settings = _production(mongodb_uri=uri)

    assert settings.external.mongodb_uri.get_secret_value() == uri


@pytest.mark.parametrize(
    "query",
    [
        "tls=true&TLS=false",  # a repeated option the driver would silently resolve
        "tls=true&tls=true",
        "tls=true&ssl=false",  # the driver rejects conflicting aliases
        "tls=maybe",  # the driver would warn and drop it, falling back to its default
        "TLS=TRUE",  # option names are case-insensitive, but boolean values are not
        "tls",
        "unknownOption=1",
        "tls=true&appName=a;replicaSet=b",
        "tls=true&tlsCRLFile=a.pem&tlsCRLFile=b.pem",
    ],
)
def test_mongodb_uri_rejects_ambiguous_or_invalid_options_without_echoing_them(
    query: str,
) -> None:
    uri = f"mongodb://sentinel-user:sentinel-pass@db.example.com/?{query}"

    with pytest.raises(ValidationError) as caught:
        ExternalSettings(_env_file=None, mongodb_uri=uri)

    rendered = str(caught.value) + repr(caught.value)
    assert "invalid, repeated or conflicting connection options" in rendered
    assert "sentinel" not in rendered
    assert query not in rendered


def test_unreadable_mongodb_tls_file_is_a_safe_settings_error(tmp_path: Path) -> None:
    # The driver's validator opens the file and raises an OSError that names the path.
    missing = tmp_path / "sentinel-ca.pem"
    uri = f"mongodb://sentinel-user:sentinel-pass@db.example.com/?tls=true&tlsCAFile={missing}"

    with pytest.raises(ValidationError) as caught:
        ExternalSettings(_env_file=None, mongodb_uri=uri)

    rendered = str(caught.value) + repr(caught.value)
    assert "invalid, repeated or conflicting connection options" in rendered
    assert "sentinel" not in rendered


def test_mongodb_option_check_runs_in_every_environment() -> None:
    with pytest.raises(ValidationError, match="conflicting connection options"):
        ExternalSettings(
            _env_file=None, app_env=Environment.LOCAL, mongodb_uri="mongodb://h/?tls=yes&tls=no"
        )


def test_transport_rejection_does_not_echo_the_credentialed_uri() -> None:
    uri = "mongodb://sentinel-user:sentinel-pass@db.internal:27017/adaptive_rag?tls=false"

    with pytest.raises(ValidationError) as caught:
        _production(mongodb_uri=uri)

    assert "sentinel" not in str(caught.value)


def test_invalid_mongodb_scheme_does_not_echo_the_uri() -> None:
    with pytest.raises(ValidationError) as caught:
        ExternalSettings(_env_file=None, mongodb_uri="http://sentinel-user:sentinel-pass@db/")

    assert "must use mongodb:// or mongodb+srv://" in str(caught.value)
    assert "sentinel" not in str(caught.value)


def test_production_loader_applies_the_transport_rule(tmp_path: Path) -> None:
    (tmp_path / "default.yaml").write_text("ai:\n  provider: fake\nreranker:\n  provider: fake\n")
    external = ExternalSettings(
        _env_file=None,
        app_env=Environment.PRODUCTION,
        config_dir=tmp_path,
        auth_shared_token=_AUTH_TOKEN,
        debug_token=_DEBUG_TOKEN,
        telemetry_hash_key=_TELEMETRY_KEY,
        mongodb_uri="mongodb+srv://appuser:secret@cluster.example.com/?ssl=false",
    )

    with pytest.raises(ValidationError, match="must use TLS in production"):
        load_settings(external)


def test_startup_failure_on_stderr_omits_the_mongodb_credential(tmp_path: Path) -> None:
    """Settings fail before logging is configured, so check the interpreter's own output."""

    environment = {
        **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        "PYTHONPATH": str(_PROJECT_ROOT / "src"),
        "APP_ENV": "production",
        "CONFIG_DIR": str(_PROJECT_CONFIG),
        "OPENAI_API_KEY": "sk-placeholder",
        "COHERE_API_KEY": "co-placeholder",
        "AUTH_SHARED_TOKEN": _AUTH_TOKEN,
        "DEBUG_TOKEN": _DEBUG_TOKEN,
        "TELEMETRY_HASH_KEY": _TELEMETRY_KEY,
    }
    for uri in (
        "mongodb://sentinel-user:sentinel-pass@db.internal/?tls=false",
        "mongodb://sentinel-user:sentinel-pass@db.internal/?tls=sentinel-value",
    ):
        completed = subprocess.run(
            [sys.executable, "-c", "from utils.settings import load_settings; load_settings()"],
            cwd=tmp_path,
            env={**environment, "MONGODB_URI": uri},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert completed.returncode != 0
        assert "ValidationError" in completed.stderr
        assert "sentinel" not in completed.stderr


@pytest.mark.parametrize(("query", "encrypted"), [("", False), ("?tls=true", True)])
async def test_mongo_client_takes_its_transport_from_the_validated_uri(
    query: str, encrypted: bool
) -> None:
    """Validation reads TLS from the URI alone, so the builder must not override it."""

    uri = f"mongodb://localhost:27017/{query}"
    client = build_mongo_client(uri=uri, server_selection_timeout_ms=100, operation_timeout_ms=100)
    try:
        assert mongodb_transport(uri).encrypted is encrypted
        assert (client.options.pool_options._ssl_context is not None) is encrypted
    finally:
        await client.close()


def test_production_allows_an_uncredentialed_mongodb_uri() -> None:
    settings = _production(mongodb_uri="mongodb://db.internal:27017/adaptive_rag")

    assert (
        settings.external.mongodb_uri.get_secret_value()
        == "mongodb://db.internal:27017/adaptive_rag"
    )


def test_production_allows_plain_http_qdrant_when_no_api_key_is_sent() -> None:
    settings = _production(qdrant_url="http://qdrant.internal:6333")

    assert settings.external.qdrant_url == "http://qdrant.internal:6333"


def test_production_accepts_https_endpoints_that_receive_api_keys() -> None:
    settings = _production(
        openai_api_key="sk-test",
        openai_base_url="https://llm-gateway.example.com/v1",
        cohere_api_key="co-test",
        cohere_base_url="https://reranker.example.com",
        qdrant_api_key="qdrant-test",
        qdrant_url="https://qdrant.example.com",
    )

    assert settings.external.openai_base_url == "https://llm-gateway.example.com/v1"


@pytest.mark.parametrize(
    ("secrets", "message"),
    [
        ({"telemetry_hash_key": "replace-this-telemetry-key"}, "must be changed"),
        ({"auth_shared_token": "x"}, "at least 32 characters"),
        ({"debug_token": "short-debug"}, "at least 32 characters"),
        ({"telemetry_hash_key": ""}, "at least 32 characters"),
        ({"auth_shared_token": _DEBUG_TOKEN}, "must be distinct"),
        ({"telemetry_hash_key": _AUTH_TOKEN}, "must be distinct"),
    ],
)
def test_production_rejects_placeholder_short_empty_or_shared_secrets(
    secrets: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _production(**secrets)


def test_timeout_reserve_must_be_below_request_budget() -> None:
    with pytest.raises(ValidationError, match="finalization_reserve_seconds must be below"):
        TimeoutConfig(request_seconds=5, finalization_reserve_seconds=5)


@pytest.mark.parametrize("level", ["TRACE", "info", "WARN"])
def test_log_level_accepts_only_declared_levels(level: str) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"log_level": level})


def test_log_level_accepts_declared_level() -> None:
    assert AppConfig(log_level="WARNING").log_level == "WARNING"


@pytest.mark.parametrize("alias", ["", "has space", "dotted.name", "slash/name", "x" * 101])
def test_collection_alias_rejects_unsafe_names(alias: str) -> None:
    with pytest.raises(ValidationError):
        VectorStoreConfig(collection_alias=alias)


def test_collection_alias_accepts_letters_digits_hyphen_and_underscore() -> None:
    alias = "adaptive_rag-chunks_1"

    assert VectorStoreConfig(collection_alias=alias).collection_alias == alias


def test_mongo_operation_timeout_has_a_floor() -> None:
    with pytest.raises(ValidationError):
        MongoConfig(operation_timeout_ms=50)


async def test_mongo_client_applies_operation_timeout_separately_from_server_selection() -> None:
    client = build_mongo_client(
        uri="mongodb://localhost:27017",
        server_selection_timeout_ms=3000,
        operation_timeout_ms=1500,
    )
    try:
        assert client.options.timeout == 1.5
        assert client.options.server_selection_timeout == 3.0
        assert client.codec_options.tz_aware
        assert client.codec_options.tzinfo is UTC
    finally:
        await client.close()


def test_new_configuration_sections_parse(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(
        contextualization={"detector": "heuristic", "history_messages": 10},
        answers={"inline_citations": False},
        context={"short_chunk_tokens": 12},
        mongodb={"operation_timeout_ms": 1500},
        telemetry={"shutdown_timeout_seconds": 2, "direct_write_timeout_seconds": 0.5},
        vector_store={"sparse": {"k1": 1.5, "b": 0.5}},
        timeouts={"request_seconds": 30, "finalization_reserve_seconds": 3},
    )

    assert settings.contextualization.detector is FollowupDetectorName.HEURISTIC
    assert settings.contextualization.history_messages == 10
    assert settings.answers.inline_citations is False
    assert settings.context.short_chunk_tokens == 12
    assert settings.mongodb.operation_timeout_ms == 1500
    assert settings.telemetry.shutdown_timeout_seconds == 2
    assert settings.telemetry.direct_write_timeout_seconds == 0.5
    assert settings.vector_store.sparse.k1 == 1.5
    assert settings.timeouts.request_seconds == 30


@pytest.mark.parametrize(
    "retries",
    [
        {"max_retries": 2},
        {"max_retries": -1},
        {"backoff_seconds": -0.1},
        {"backoff_seconds": float("inf")},
        {"min_retry_seconds": -1},
        {"min_retry_seconds": float("nan")},
    ],
    ids=["two-retries", "negative", "negative-wait", "endless-wait", "negative-margin", "nan"],
)
def test_retry_settings_stay_bounded(
    make_settings: Callable[..., Settings], retries: dict[str, float]
) -> None:
    with pytest.raises(ValidationError):
        make_settings(retries=retries)


def test_retries_can_be_turned_off_and_the_sdk_retry_setting_is_gone(
    make_settings: Callable[..., Settings],
) -> None:
    assert make_settings(retries={"max_retries": 0}).retries.max_retries == 0
    with pytest.raises(ValidationError, match="extra_forbidden"):
        make_settings(ai={"provider": "fake", "max_retries": 2})
