"""Make one real call per provider operation to confirm the configured models and schemas work."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel

from models.domain import (
    Chunk,
    CompletenessEvaluation,
    ContextualizationOutput,
    FaithfulnessEvaluation,
    FollowupClassification,
    GenerationDraft,
    PackedContext,
    QueryRecoveryOutput,
    QueryUnderstanding,
    RankedChunk,
)
from providers.cohere import CohereReranker
from providers.openai import OpenAIEmbeddingProvider, OpenAITextModel, build_openai_client
from utils.errors import ConfigurationError
from utils.prompts import (
    COMPLETENESS_SYSTEM,
    CONTEXTUALIZER_SYSTEM,
    FAITHFULNESS_SYSTEM,
    FOLLOWUP_DETECTOR_SYSTEM,
    GENERATOR_SYSTEM,
    QUERY_RECOVERY_SYSTEM,
    contextualization_user_prompt,
    evaluation_user_prompt,
    followup_detection_user_prompt,
    generation_user_prompt,
    query_recovery_user_prompt,
)
from utils.settings import Settings, load_settings

_QUESTION = "How long are request traces kept?"
_ANSWER = "Request traces are kept for 30 days [E1]."
_HISTORY = ("user: Where are request traces stored?", "assistant: In MongoDB.")


@dataclass(frozen=True, slots=True)
class _Check:
    name: str
    run: Callable[[], Awaitable[str]]


def _context() -> PackedContext:
    chunk = Chunk(
        id="00000000-0000-4000-8000-000000000001",
        tenant_id="smoke",
        document_id="smoke-document",
        document_version_id="smoke-version",
        document_version=1,
        document_title="Smoke test handbook",
        section_id="retention",
        sequence=0,
        heading_path=("Retention",),
        text="Request traces are stored in MongoDB and kept for 30 days.",
        token_count=14,
        allowed_roles=frozenset({"reader"}),
    )
    return PackedContext(
        chunks=(RankedChunk(chunk=chunk, fusion_score=1, fusion_rank=1),),
        token_count=chunk.token_count,
    )


def _checks(settings: Settings, text_model: OpenAITextModel) -> list[_Check]:
    ai = settings.ai
    context = _context()
    understanding = QueryUnderstanding(
        literal_question=_QUESTION, canonical_question=_QUESTION, rewritten=False
    )

    def structured(
        name: str, model: str, system: str, user: str, output_type: type[BaseModel]
    ) -> _Check:
        async def run() -> str:
            result = await text_model.complete(
                model=model,
                system_prompt=system,
                user_prompt=user,
                output_type=output_type,
                max_output_tokens=ai.max_output_tokens,
            )
            usage = result.usage
            return (
                f"{result.model}, {usage.output_tokens} output tokens "
                f"({usage.reasoning_tokens} reasoning)"
            )

        return _Check(f"{name} ({model})", run)

    return [
        structured(
            "follow-up detection",
            settings.contextualization.detector_model,
            FOLLOWUP_DETECTOR_SYSTEM,
            followup_detection_user_prompt("and how long?", _HISTORY),
            FollowupClassification,
        ),
        structured(
            "contextualization",
            ai.contextualization_model,
            CONTEXTUALIZER_SYSTEM,
            contextualization_user_prompt("and how long?", _HISTORY),
            ContextualizationOutput,
        ),
        structured(
            "query recovery",
            ai.query_recovery_model,
            QUERY_RECOVERY_SYSTEM,
            query_recovery_user_prompt(understanding),
            QueryRecoveryOutput,
        ),
        structured(
            "generation",
            ai.generation_model,
            GENERATOR_SYSTEM,
            generation_user_prompt(_QUESTION, context, ()),
            GenerationDraft,
        ),
        structured(
            "faithfulness evaluation",
            ai.evaluation_model,
            FAITHFULNESS_SYSTEM,
            evaluation_user_prompt(_QUESTION, _ANSWER, context),
            FaithfulnessEvaluation,
        ),
        structured(
            "completeness evaluation",
            ai.evaluation_model,
            COMPLETENESS_SYSTEM,
            evaluation_user_prompt(_QUESTION, _ANSWER, context),
            CompletenessEvaluation,
        ),
    ]


async def smoke_test(settings: Settings) -> bool:
    """Run every check, print one line per check, and return whether all passed."""

    if settings.external.openai_api_key is None or settings.external.cohere_api_key is None:
        raise ConfigurationError("OPENAI_API_KEY and COHERE_API_KEY are required")
    client = build_openai_client(
        api_key=settings.external.openai_api_key.get_secret_value(),
        base_url=settings.external.openai_base_url,
        max_retries=0,
        timeout_seconds=settings.timeouts.generation_seconds,
    )
    reranker = CohereReranker(
        api_key=settings.external.cohere_api_key.get_secret_value(),
        model=settings.reranker.model,
        base_url=settings.external.cohere_base_url,
        timeout_seconds=settings.timeouts.reranking_seconds,
    )
    embeddings = OpenAIEmbeddingProvider(
        client, model=settings.ai.embedding_model, dimension=settings.ai.embedding_dimension
    )
    text_model = OpenAITextModel(client, reasoning_effort=settings.ai.reasoning_effort)

    async def embed() -> str:
        result = await embeddings.embed_query(_QUESTION)
        return f"{result.model}, {len(result.vector)} dimensions"

    async def rerank() -> str:
        ranked = await reranker.rerank(_QUESTION, _context().chunks, 1)
        return f"{settings.reranker.model}, top score {ranked[0].reranker_score}"

    checks = [
        _Check(f"embedding ({settings.ai.embedding_model})", embed),
        _Check(f"rerank ({settings.reranker.model})", rerank),
        *_checks(settings, text_model),
    ]
    passed = True
    try:
        for check in checks:
            try:
                detail = await check.run()
            except Exception as exc:  # noqa: BLE001 - every failure is reported, then the run fails
                passed = False
                cause = exc.__cause__ or exc
                print(f"FAIL {check.name}: {type(exc).__name__}: {exc} ({cause})")  # noqa: T201
            else:
                print(f"ok   {check.name}: {detail}")  # noqa: T201
    finally:
        await reranker.close()
        await client.close()
    return passed


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not asyncio.run(smoke_test(load_settings())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
