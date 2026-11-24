from __future__ import annotations

import asyncio

import pytest

from models.domain import (
    EncodedQuery,
    HybridHits,
    Identity,
    QueryRecoveryOutput,
    QueryUnderstanding,
    RetrievalPlan,
    SearchHit,
)
from providers.sparse import Bm25SparseEncoder
from services.encoding import EmbeddingCache, HybridQueryEncoder
from services.retrieval import (
    HybridRetriever,
    QueryRecoveryService,
    StrategyRetriever,
    fuse_results,
)
from tests.fakes import (
    FakeEmbeddings,
    FakeRepository,
    FakeTextModel,
    FakeVectorSearch,
    StalledTextModel,
    make_chunk,
)
from utils.settings import FusionStrategy, RetrievalConfig, SparseConfig

_CHUNK_ID = "00000000-0000-0000-0000-000000000001"
_OTHER_ID = "00000000-0000-0000-0000-000000000002"


def _query() -> QueryUnderstanding:
    return QueryUnderstanding(
        literal_question="literal",
        canonical_question="canonical",
        rewritten=False,
    )


def _retriever(
    vector_store: FakeVectorSearch,
    repository: FakeRepository,
    *,
    retrieval_timeout_seconds: float = 1,
) -> HybridRetriever:
    encoder = HybridQueryEncoder(
        dense=FakeEmbeddings(),
        sparse=Bm25SparseEncoder(SparseConfig()),
        cache=EmbeddingCache(max_entries=0),
        embedding_timeout_seconds=1,
    )
    return HybridRetriever(
        encoder=encoder,
        vector_store=vector_store,
        repository=repository,
        config=RetrievalConfig(),
        retrieval_timeout_seconds=retrieval_timeout_seconds,
    )


class _StalledSearch(FakeVectorSearch):
    """A vector store that never returns, so only the stage budget can end the search."""

    async def search(
        self,
        query: EncodedQuery,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        *,
        dense_k: int,
        sparse_k: int,
    ) -> HybridHits:
        del query, identity, filters, dense_k, sparse_k
        await asyncio.sleep(60)
        return HybridHits()


class _OverlapProbe(FakeVectorSearch):
    """Records overlapping searches; each search waits until ``rendezvous`` searches overlap."""

    def __init__(self, *, rendezvous: int) -> None:
        super().__init__()
        self.in_flight = 0
        self.max_in_flight = 0
        self._rendezvous = rendezvous
        self._reached = asyncio.Event()

    async def search(
        self,
        query: EncodedQuery,
        identity: Identity,
        filters: dict[str, str | int | float | bool],
        *,
        dense_k: int,
        sparse_k: int,
    ) -> HybridHits:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            if self.in_flight >= self._rendezvous:
                self._reached.set()
            await self._reached.wait()
            return await super().search(
                query, identity, filters, dense_k=dense_k, sparse_k=sparse_k
            )
        finally:
            self.in_flight -= 1


def _recovery(
    model: FakeTextModel,
    *,
    max_variants: int = 4,
    max_decomposed: int = 3,
) -> QueryRecoveryService:
    return QueryRecoveryService(
        model=model,
        model_name="fake",
        max_output_tokens=100,
        timeout_seconds=1,
        max_variants=max_variants,
        max_decomposed_queries=max_decomposed,
    )


def test_rrf_deduplicates_and_rewards_results_present_in_both_lists() -> None:
    fused = fuse_results(
        [
            SearchHit(chunk_id="both", score=0.8, rank=1),
            SearchHit(chunk_id="dense", score=0.7, rank=2),
        ],
        [
            SearchHit(chunk_id="sparse", score=12, rank=1),
            SearchHit(chunk_id="both", score=10, rank=2),
        ],
        strategy=FusionStrategy.RRF,
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
        limit=3,
    )

    assert [item.chunk_id for item in fused] == ["both", "sparse", "dense"]
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 2


def test_weighted_fusion_normalizes_incomparable_score_ranges() -> None:
    fused = fuse_results(
        [
            SearchHit(chunk_id="dense", score=0.9, rank=1),
            SearchHit(chunk_id="sparse", score=0.1, rank=2),
        ],
        [
            SearchHit(chunk_id="sparse", score=100, rank=1),
            SearchHit(chunk_id="dense", score=1, rank=2),
        ],
        strategy=FusionStrategy.WEIGHTED,
        dense_weight=0.75,
        sparse_weight=0.25,
        rrf_k=60,
        limit=2,
    )

    assert fused[0].chunk_id == "dense"
    assert fused[0].fusion_score == pytest.approx(0.75)


def test_fusion_without_a_limit_keeps_every_distinct_hit() -> None:
    fused = fuse_results(
        [SearchHit(chunk_id=f"dense-{rank}", score=1, rank=rank) for rank in range(1, 6)],
        [SearchHit(chunk_id="dense-1", score=1, rank=1)],
        strategy=FusionStrategy.RRF,
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
        limit=None,
    )

    assert len(fused) == 5
    assert [item.fusion_rank for item in fused] == [1, 2, 3, 4, 5]


async def test_hybrid_retrieval_drops_cross_tenant_and_role_hits(identity: Identity) -> None:
    allowed = make_chunk(_CHUNK_ID)
    other_tenant = make_chunk(_OTHER_ID, tenant_id="tenant-b")
    other_role = make_chunk(
        "00000000-0000-0000-0000-000000000003",
        roles=frozenset({"administrator"}),
    )
    repository = FakeRepository((allowed, other_tenant, other_role))
    vector_store = FakeVectorSearch()
    vector_store.dense_hits = [
        SearchHit(chunk_id=other_tenant.id, score=1, rank=1),
        SearchHit(chunk_id=allowed.id, score=0.9, rank=2),
    ]
    vector_store.sparse_hits = [
        SearchHit(chunk_id=other_role.id, score=9, rank=1),
        SearchHit(chunk_id=allowed.id, score=8, rank=2),
    ]

    outcome = await _retriever(vector_store, repository).retrieve(
        query="answer",
        identity=identity,
        filters={},
        plan=RetrievalPlan(dense_k=10, sparse_k=10, candidate_k=10),
    )

    assert [item.chunk.id for item in outcome.ranked_chunks] == [allowed.id]
    assert outcome.embedding_usage.total_tokens == 2
    assert outcome.model_calls[0].request_id == "embedding-1"


async def test_unauthorized_hits_do_not_consume_candidate_slots(identity: Identity) -> None:
    allowed = make_chunk(_CHUNK_ID)
    other_tenant = make_chunk(_OTHER_ID, tenant_id="tenant-b")
    repository = FakeRepository((allowed, other_tenant))
    vector_store = FakeVectorSearch()
    vector_store.dense_hits = [
        SearchHit(chunk_id=other_tenant.id, score=1, rank=1),
        SearchHit(chunk_id=allowed.id, score=0.9, rank=2),
    ]

    outcome = await _retriever(vector_store, repository).retrieve(
        query="answer",
        identity=identity,
        filters={},
        plan=RetrievalPlan(dense_k=10, sparse_k=0, candidate_k=1),
    )

    assert [(item.chunk.id, item.fusion_rank) for item in outcome.ranked_chunks] == [
        (allowed.id, 1)
    ]


async def test_plan_breadth_reaches_the_vector_store(identity: Identity) -> None:
    chunk = make_chunk(_CHUNK_ID)
    vector_store = FakeVectorSearch()
    vector_store.dense_hits = [
        SearchHit(chunk_id=chunk.id, score=1, rank=1),
        SearchHit(chunk_id=_OTHER_ID, score=0.5, rank=2),
    ]

    outcome = await _retriever(vector_store, FakeRepository((chunk,))).retrieve(
        query="answer",
        identity=identity,
        filters={},
        plan=RetrievalPlan(dense_k=1, sparse_k=0, candidate_k=5),
    )

    assert outcome.dense_result_count == 1
    assert vector_store.last_query is not None
    assert vector_store.last_query.text == "answer"


async def test_decomposed_queries_have_their_own_budget() -> None:
    model = FakeTextModel()
    model.queue(
        QueryRecoveryOutput,
        QueryRecoveryOutput(
            keyword_query="keywords",
            decomposed_queries=("first", "second", "third", "fourth"),
        ),
    )

    variants = await _recovery(model, max_variants=2, max_decomposed=2).build(_query())

    assert variants.values == ("canonical", "literal", "first", "second")


@pytest.mark.parametrize(
    ("literal", "canonical", "expected"),
    [
        pytest.param(
            "What about the second one?",
            "What is the audit log retention?",
            ("What is the audit log retention?", "keywords", "first"),
            id="resolved-reference-is-dropped",
        ),
        pytest.param(
            "What about product A pricing?",
            "Explain product B migration limitations",
            (
                "Explain product B migration limitations",
                "What about product A pricing?",
                "keywords",
                "first",
            ),
            id="literal-with-dropped-words-is-kept",
        ),
    ],
)
async def test_rewritten_follow_up_literal_is_searched_only_when_it_adds_words(
    literal: str, canonical: str, expected: tuple[str, ...]
) -> None:
    model = FakeTextModel()
    model.queue(
        QueryRecoveryOutput,
        QueryRecoveryOutput(keyword_query="keywords", decomposed_queries=("first",)),
    )
    query = QueryUnderstanding(
        literal_question=literal, canonical_question=canonical, rewritten=True
    )

    variants = await _recovery(model).build(query)

    assert variants.values == expected


async def test_variants_are_stripped_deduplicated_and_blank_values_dropped() -> None:
    model = FakeTextModel()
    model.queue(
        QueryRecoveryOutput,
        QueryRecoveryOutput(
            keyword_query="   ",
            decomposed_queries=(" canonical ", "first", "first", "", "second"),
        ),
    )

    variants = await _recovery(model).build(_query())

    assert variants.values == ("canonical", "literal", "first", "second")


async def test_plan_without_recovery_searches_only_the_canonical_question(
    identity: Identity,
) -> None:
    chunk = make_chunk(_CHUNK_ID)
    hybrid = _retriever(FakeVectorSearch(), FakeRepository((chunk,)))
    retriever = StrategyRetriever(
        hybrid=hybrid,
        query_recovery=_recovery(FakeTextModel()),
        config=RetrievalConfig(),
    )

    outcome = await retriever.retrieve(
        plan=RetrievalPlan(dense_k=5, sparse_k=5, candidate_k=5),
        query=_query(),
        identity=identity,
        filters={},
    )

    assert outcome.queries == ("canonical",)


async def test_recovery_plan_runs_every_variant_and_merges_results(identity: Identity) -> None:
    model = FakeTextModel()
    model.queue(
        QueryRecoveryOutput,
        QueryRecoveryOutput(keyword_query="keywords", decomposed_queries=("first",)),
    )
    chunk = make_chunk(_CHUNK_ID)
    vector_store = FakeVectorSearch()
    vector_store.dense_hits = [SearchHit(chunk_id=chunk.id, score=1, rank=1)]
    retriever = StrategyRetriever(
        hybrid=_retriever(vector_store, FakeRepository((chunk,))),
        query_recovery=_recovery(model),
        config=RetrievalConfig(),
    )

    outcome = await retriever.retrieve(
        plan=RetrievalPlan(dense_k=5, sparse_k=0, candidate_k=5, recover_queries=True),
        query=_query(),
        identity=identity,
        filters={},
    )

    assert outcome.queries == ("canonical", "literal", "keywords", "first")
    assert [(item.chunk.id, item.fusion_rank) for item in outcome.ranked_chunks] == [(chunk.id, 1)]


async def test_query_variants_never_exceed_the_configured_concurrency(identity: Identity) -> None:
    model = FakeTextModel()
    model.queue(
        QueryRecoveryOutput,
        QueryRecoveryOutput(keyword_query="keywords", decomposed_queries=("first",)),
    )
    vector_store = _OverlapProbe(rendezvous=1)
    retriever = StrategyRetriever(
        hybrid=_retriever(vector_store, FakeRepository()),
        query_recovery=_recovery(model),
        config=RetrievalConfig(query_variant_concurrency=1),
    )

    outcome = await retriever.retrieve(
        plan=RetrievalPlan(dense_k=5, sparse_k=0, candidate_k=5, recover_queries=True),
        query=_query(),
        identity=identity,
        filters={},
    )

    assert len(outcome.queries) == 4
    assert vector_store.search_calls == 4
    assert vector_store.max_in_flight == 1


async def test_concurrency_limit_is_scoped_to_each_retrieve_call(identity: Identity) -> None:
    model = FakeTextModel()
    model.queue(QueryRecoveryOutput, QueryRecoveryOutput(), QueryRecoveryOutput())
    vector_store = _OverlapProbe(rendezvous=2)
    retriever = StrategyRetriever(
        hybrid=_retriever(vector_store, FakeRepository()),
        query_recovery=_recovery(model),
        config=RetrievalConfig(query_variant_concurrency=1),
    )
    plan = RetrievalPlan(dense_k=5, sparse_k=0, candidate_k=5, recover_queries=True)

    async with asyncio.timeout(5):
        await asyncio.gather(
            retriever.retrieve(plan=plan, query=_query(), identity=identity, filters={}),
            retriever.retrieve(plan=plan, query=_query(), identity=identity, filters={}),
        )

    assert vector_store.max_in_flight == 2


async def test_vector_search_is_bounded_by_the_retrieval_stage_budget(identity: Identity) -> None:
    retriever = _retriever(
        _StalledSearch(),
        FakeRepository((make_chunk(_CHUNK_ID),)),
        retrieval_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await retriever.retrieve(
            query="answer",
            identity=identity,
            filters={},
            plan=RetrievalPlan(dense_k=5, sparse_k=5, candidate_k=5),
        )


async def test_query_recovery_is_bounded_by_its_stage_budget() -> None:
    service = QueryRecoveryService(
        model=StalledTextModel(),
        model_name="fake",
        max_output_tokens=100,
        timeout_seconds=0.01,
        max_variants=4,
        max_decomposed_queries=3,
    )

    with pytest.raises(TimeoutError):
        await service.build(_query())


async def test_fused_candidates_are_hydrated_in_one_repository_call(identity: Identity) -> None:
    chunks = [make_chunk(f"00000000-0000-0000-0000-00000000000{index}") for index in range(1, 5)]
    repository = FakeRepository(chunks)
    vector_store = FakeVectorSearch()
    vector_store.dense_hits = [
        SearchHit(chunk_id=chunks[0].id, score=1, rank=1),
        SearchHit(chunk_id=chunks[1].id, score=0.9, rank=2),
        SearchHit(chunk_id=chunks[2].id, score=0.8, rank=3),
    ]
    vector_store.sparse_hits = [
        SearchHit(chunk_id=chunks[3].id, score=9, rank=1),
        SearchHit(chunk_id=chunks[0].id, score=8, rank=2),
    ]

    outcome = await _retriever(vector_store, repository).retrieve(
        query="answer",
        identity=identity,
        filters={},
        plan=RetrievalPlan(dense_k=10, sparse_k=10, candidate_k=10),
    )

    assert len(outcome.ranked_chunks) == 4
    assert len(repository.hydrate_calls) == 1
    assert sorted(repository.hydrate_calls[0]) == sorted(chunk.id for chunk in chunks)


async def test_filter_hydrated_seeds_before_candidate_limit(identity: Identity) -> None:
    chunks = [
        make_chunk(key).model_copy(update={"metadata": {"lang": lang}})
        for key, lang in [("first", "fr"), ("second", "en")]
    ]
    store = FakeVectorSearch()
    store.dense_hits = [
        SearchHit(chunk_id=c.id, score=1 / rank, rank=rank) for rank, c in enumerate(chunks, 1)
    ]
    outcome = await _retriever(store, FakeRepository(chunks)).retrieve(
        query="handbook",
        identity=identity,
        filters={"lang": "en"},
        plan=RetrievalPlan(dense_k=2, sparse_k=0, candidate_k=1),
    )
    assert [r.chunk.id for r in outcome.ranked_chunks] == ["second"]
    assert outcome.metadata_mismatch_count == 1
