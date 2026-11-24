from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

import pytest

from models.domain import Chunk, Identity, RankedChunk
from services.context import ContextBuilder, ContextPacker, StructuralCompleter, utf8_byte_count
from tests.fakes import FakeRepository, FakeReranker, make_chunk
from utils.settings import ContextConfig

HEADER = "Document: Guide\nSection: Test\nVersion: 1\n\n"
OTHER_VERSION = "00000000-0000-0000-0000-000000000099"


class _StalledReranker(FakeReranker):
    """A reranker that never returns, so only the reranking budget can end the stage."""

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RankedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        del query, chunks, top_n
        await asyncio.sleep(60)
        return []


class _Neighbors(FakeRepository):
    """Serves every supplied chunk so tests exercise only the completer's own constraints."""

    def __init__(self, *chunks: Chunk) -> None:
        super().__init__(chunks)
        self.requests: list[tuple[str, ...]] = []

    async def get_neighbors(
        self,
        chunk_ids: Sequence[str],
        identity: Identity,
    ) -> dict[str, Chunk]:
        del identity
        self.requests.append(tuple(chunk_ids))
        return {
            chunk_id: self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks
        }


def _chunk(
    chunk_id: str, body: str = "Plain evidence.", *, tokens: int = 200, **fields: Any
) -> Chunk:
    chunk = make_chunk(chunk_id, text=HEADER + body, **fields)
    return chunk.model_copy(update={"token_count": tokens})


def _ranked(chunk: Chunk, rank: int = 1) -> RankedChunk:
    return RankedChunk(
        chunk=chunk,
        dense_score=1 / rank,
        dense_rank=rank,
        sparse_score=1 / rank,
        sparse_rank=rank,
        fusion_score=1 / rank,
        fusion_rank=rank,
        reranker_score=1 / rank,
        reranker_rank=rank,
    )


def _neighbor(chunk: Chunk, anchor: str) -> RankedChunk:
    return RankedChunk(
        chunk=chunk,
        fusion_score=0.5,
        fusion_rank=2,
        is_neighbor=True,
        anchor_chunk_id=anchor,
    )


def _completer(
    repository: FakeRepository,
    *,
    max_neighbors: int = 1,
    short_chunk_tokens: int = 0,
) -> StructuralCompleter:
    return StructuralCompleter(
        repository=repository,
        max_neighbors_per_seed=max_neighbors,
        short_chunk_tokens=short_chunk_tokens,
    )


def _config(**overrides: Any) -> ContextConfig:
    values: dict[str, Any] = {
        "max_tokens": 1000,
        "max_seed_chunks": 8,
        "max_document_share": 1.0,
        "max_neighbors_per_seed": 1,
        "short_chunk_tokens": 0,
    }
    return ContextConfig(**(values | overrides))


def _ids(items: Sequence[RankedChunk]) -> list[str]:
    return [item.chunk.id for item in items]


async def test_structural_completion_places_previous_neighbour_before_seed(
    identity: Identity,
) -> None:
    previous = make_chunk(
        "00000000-0000-0000-0000-000000000001",
        sequence=0,
        next_id="00000000-0000-0000-0000-000000000002",
    )
    seed = make_chunk(
        "00000000-0000-0000-0000-000000000002",
        sequence=1,
        previous_id=previous.id,
        text="Document: Guide\nSection: Test\nVersion: 1\n\n2. Continue setup.",
    )
    repository = FakeRepository((previous, seed))
    completer = StructuralCompleter(
        repository=repository,
        max_neighbors_per_seed=1,
        short_chunk_tokens=0,
    )

    completed = await completer.complete(
        [
            RankedChunk(
                chunk=seed,
                fusion_score=1,
                fusion_rank=1,
                reranker_score=1,
                reranker_rank=1,
            )
        ],
        identity,
    )

    assert [item.chunk.id for item in completed] == [previous.id, seed.id]
    assert completed[0].is_neighbor


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("2. Continue setup\nthen finish", {"previous"}, id="mid-list"),
        pytest.param("As described above, configure it.", {"previous"}, id="back-reference"),
        pytest.param("This setting controls retries.", {"previous"}, id="leading-pronoun"),
        pytest.param("| name | value |\n\nClosing sentence.", {"previous"}, id="table-first"),
        pytest.param("print(1)\n```\nthe rest of the code", {"previous"}, id="fence-open-prev"),
        pytest.param("Use the following steps", {"next"}, id="forward-reference"),
        pytest.param("Configure the value:", {"next"}, id="trailing-colon"),
        pytest.param("Steps:\n1. Start\n2. Stop", {"next"}, id="numbered-last"),
        pytest.param("Options:\n- first\n- second", {"next"}, id="bullet-last"),
        pytest.param("Options:\n* first\n* second", {"next"}, id="star-bullet-last"),
        pytest.param("Options:\n+ first\n+ second", {"next"}, id="plus-bullet-last"),
        pytest.param("Steps:\n1) Start\n2) Stop", {"next"}, id="paren-numbered-last"),
        pytest.param("Details are as follows.", {"next"}, id="forward-as-follows"),
        pytest.param("Details continued below.", {"next"}, id="forward-continued-below"),
        pytest.param("Summary\n| a | 1 |", {"next"}, id="table-last"),
        pytest.param("```python\nprint(1)", {"next"}, id="fence-open-next"),
        pytest.param("Plain sentence without signals.", set(), id="no-signal"),
    ],
)
async def test_boundary_signals_request_only_their_neighbour(
    body: str,
    expected: set[str],
    identity: Identity,
) -> None:
    seed = _chunk("seed", body, previous_id="prev", next_id="next")
    repository = _Neighbors(_chunk("prev"), _chunk("next"))

    completed = await _completer(repository, max_neighbors=2).complete([_ranked(seed)], identity)

    roles = {
        "previous" if item.chunk.id == "prev" else "next" for item in completed if item.is_neighbor
    }
    assert roles == expected


@pytest.mark.parametrize(
    "word",
    ["this", "these", "it", "its", "they", "their", "that", "those", "such"],
)
async def test_each_unresolved_pronoun_opening_requests_the_previous_neighbour(
    word: str,
    identity: Identity,
) -> None:
    seed = _chunk("seed", f"{word.capitalize()} setting applies.", previous_id="prev")
    repository = _Neighbors(_chunk("prev"))

    completed = await _completer(repository).complete([_ranked(seed)], identity)

    assert _ids(completed) == ["prev", "seed"]


@pytest.mark.parametrize(
    ("offset", "expected_ids"),
    [(250, {"prev"}), (350, set())],
    ids=["inside-first-300", "beyond-first-300"],
)
async def test_back_reference_is_only_detected_in_the_first_300_characters(
    offset: int,
    expected_ids: set[str],
    identity: Identity,
) -> None:
    body = "x" * (offset - 1) + " as described above."
    seed = _chunk("seed", body, previous_id="prev")
    repository = _Neighbors(_chunk("prev"))

    completed = await _completer(repository, max_neighbors=2).complete([_ranked(seed)], identity)

    assert {item.chunk.id for item in completed if item.is_neighbor} == expected_ids


@pytest.mark.parametrize(
    ("start_from_end", "expected_ids"),
    [(500, {"next"}), (501, set())],
    ids=["inside-last-500", "beyond-last-500"],
)
async def test_forward_reference_is_only_detected_in_the_last_500_characters(
    start_from_end: int,
    expected_ids: set[str],
    identity: Identity,
) -> None:
    total = 700
    phrase = "the following"
    start = total - start_from_end
    body = "x" * (start - 1) + " " + phrase + " " + "x" * (total - start - len(phrase) - 1)
    seed = _chunk("seed", body, next_id="next")
    repository = _Neighbors(_chunk("next"))

    completed = await _completer(repository, max_neighbors=2).complete([_ranked(seed)], identity)

    assert {item.chunk.id for item in completed if item.is_neighbor} == expected_ids


async def test_reranking_is_bounded_by_its_stage_budget(identity: Identity) -> None:
    builder = ContextBuilder(
        reranker=_StalledReranker(),
        completer=_completer(_Neighbors()),
        packer=ContextPacker(_config(), utf8_byte_count),
        generation_k=1,
        reranking_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await builder.build(
            query="question",
            candidates=[_ranked(_chunk("seed"))],
            identity=identity,
        )


async def test_short_chunk_requests_both_neighbours_without_signals(identity: Identity) -> None:
    seed = _chunk("seed", tokens=10, previous_id="prev", next_id="next")
    repository = _Neighbors(_chunk("prev"), _chunk("next"))

    completed = await _completer(repository, max_neighbors=2, short_chunk_tokens=40).complete(
        [_ranked(seed)], identity
    )

    assert _ids(completed) == ["prev", "seed", "next"]


async def test_short_chunk_keeps_previous_neighbour_when_cap_allows_one(identity: Identity) -> None:
    seed = _chunk("seed", tokens=10, previous_id="prev", next_id="next")
    repository = _Neighbors(_chunk("prev"), _chunk("next"))

    completed = await _completer(repository, max_neighbors=1, short_chunk_tokens=40).complete(
        [_ranked(seed)], identity
    )

    assert _ids(completed) == ["prev", "seed"]


@pytest.mark.parametrize(
    "override",
    [{"document_id": "doc-b"}, {"version_id": OTHER_VERSION}, {"section_id": "section-b"}],
    ids=["document", "version", "section"],
)
async def test_neighbour_must_share_document_version_and_section(
    override: dict[str, Any],
    identity: Identity,
) -> None:
    seed = _chunk("seed", "This setting applies.", previous_id="prev")
    repository = _Neighbors(_chunk("prev", **override))

    completed = await _completer(repository).complete([_ranked(seed)], identity)

    assert _ids(completed) == ["seed"]


async def test_rejected_neighbour_does_not_consume_the_per_seed_cap(identity: Identity) -> None:
    seed = _chunk(
        "seed",
        "This setting uses the following steps:",
        previous_id="prev",
        next_id="next",
    )
    repository = _Neighbors(_chunk("prev", section_id="section-b"), _chunk("next"))

    completed = await _completer(repository, max_neighbors=1).complete([_ranked(seed)], identity)

    assert _ids(completed) == ["seed", "next"]


async def test_previous_neighbour_is_preferred_when_cap_allows_one(identity: Identity) -> None:
    seed = _chunk(
        "seed",
        "This setting uses the following steps:",
        previous_id="prev",
        next_id="next",
    )
    repository = _Neighbors(_chunk("prev"), _chunk("next"))

    completed = await _completer(repository, max_neighbors=1).complete([_ranked(seed)], identity)

    assert _ids(completed) == ["prev", "seed"]


async def test_zero_neighbour_cap_returns_unique_seeds_without_a_lookup(identity: Identity) -> None:
    seed = _chunk(
        "seed",
        "This setting uses the following steps:",
        previous_id="prev",
        next_id="next",
    )
    repository = _Neighbors(_chunk("prev"), _chunk("next"))

    completed = await _completer(repository, max_neighbors=0).complete(
        [_ranked(seed), _ranked(seed, 2)], identity
    )

    assert _ids(completed) == ["seed"]
    assert repository.requests == []


async def test_unauthorized_neighbour_is_not_returned(identity: Identity) -> None:
    seed = _chunk("seed", "This setting applies.", previous_id="prev")
    repository = FakeRepository((_chunk("prev", roles=frozenset({"admin"})),))

    completed = await _completer(repository).complete([_ranked(seed)], identity)

    assert _ids(completed) == ["seed"]


async def test_neighbour_that_is_a_seed_is_skipped_without_consuming_the_cap(
    identity: Identity,
) -> None:
    first = _chunk(
        "seed-a",
        "This setting uses the following steps:",
        previous_id="seed-b",
        next_id="next",
    )
    second = _chunk("seed-b")
    repository = _Neighbors(second, _chunk("next"))

    completed = await _completer(repository, max_neighbors=1).complete(
        [_ranked(first, 1), _ranked(second, 2)], identity
    )

    assert _ids(completed) == ["seed-a", "next", "seed-b"]
    assert not completed[2].is_neighbor
    assert completed[2].fusion_rank == 2


async def test_shared_neighbour_is_fetched_once_and_offered_to_each_anchor(
    identity: Identity,
) -> None:
    first = _chunk("seed-a", "Use the following steps:", next_id="shared")
    second = _chunk("seed-b", "This setting applies.", previous_id="shared")
    repository = _Neighbors(_chunk("shared"))

    completed = await _completer(repository).complete(
        [_ranked(first, 1), _ranked(second, 2)], identity
    )

    assert _ids(completed) == ["seed-a", "shared", "shared", "seed-b"]
    assert [item.anchor_chunk_id for item in completed if item.is_neighbor] == [
        "seed-a",
        "seed-b",
    ]
    assert repository.requests == [("shared",)]


async def test_shared_neighbour_is_packed_once_under_its_first_packed_anchor(
    identity: Identity,
) -> None:
    first = _chunk("seed-a", "Use the following steps:", next_id="shared", tokens=100)
    second = _chunk("seed-b", "This setting applies.", previous_id="shared", tokens=100)
    completed = await _completer(_Neighbors(_chunk("shared", "Shared.", tokens=50))).complete(
        [_ranked(first, 1), _ranked(second, 2)], identity
    )

    packed = ContextPacker(_config(), utf8_byte_count).pack(completed)

    assert _ids(packed.chunks) == ["seed-a", "shared", "seed-b"]
    assert packed.token_count == 250


async def test_shared_neighbour_packs_with_a_later_anchor_when_the_first_is_rejected(
    identity: Identity,
) -> None:
    first = _chunk("seed-a", "Use the following steps:", next_id="shared", tokens=400)
    second = _chunk("seed-b", "This setting applies.", previous_id="shared", tokens=100)
    completed = await _completer(_Neighbors(_chunk("shared", "Shared.", tokens=50))).complete(
        [_ranked(first, 1), _ranked(second, 2)], identity
    )

    measurements: list[str] = []

    def count_tokens(text: str) -> int:
        measurements.append(text)
        return len(text)

    packer = ContextPacker(_config(max_tokens=300), count_tokens)
    packed = packer.pack(completed)

    assert _ids(packed.chunks) == ["shared", "seed-b"]
    assert packed.token_count == 150
    assert sorted(measurements) == sorted({item.chunk.text for item in completed})
    measurements.clear()
    assert packer.pack(completed) == packed
    assert sorted(measurements) == sorted({item.chunk.text for item in completed})


async def test_chunks_without_signals_trigger_no_neighbour_fetch(identity: Identity) -> None:
    seed = _chunk("seed", "Plain evidence.", previous_id="prev", next_id="next")
    repository = _Neighbors(_chunk("prev"), _chunk("next"))

    completed = await _completer(repository, max_neighbors=2).complete([_ranked(seed)], identity)

    assert _ids(completed) == ["seed"]
    assert repository.requests == []


async def test_neighbours_follow_reading_order_and_inherit_seed_scores(identity: Identity) -> None:
    first = _chunk(
        "seed-a",
        "This setting uses the following steps:",
        previous_id="prev-a",
        next_id="next-a",
    )
    second = _chunk("seed-b", "Use the following steps:", next_id="next-b")
    repository = _Neighbors(_chunk("prev-a"), _chunk("next-a"), _chunk("next-b"))
    seeds = [_ranked(first, 1), _ranked(second, 2)]

    completed = await _completer(repository, max_neighbors=2).complete(seeds, identity)

    assert _ids(completed) == ["prev-a", "seed-a", "next-a", "seed-b", "next-b"]
    previous_item, next_item = completed[0], completed[4]
    assert previous_item.is_neighbor
    assert previous_item.anchor_chunk_id == "seed-a"
    assert previous_item.fusion_rank == 1
    assert previous_item.dense_score == seeds[0].dense_score
    assert next_item.is_neighbor
    assert next_item.anchor_chunk_id == "seed-b"
    assert next_item.fusion_rank == 2
    assert next_item.reranker_rank == 2
    assert not completed[1].is_neighbor
    assert completed[1].anchor_chunk_id is None


async def test_builder_completes_only_generation_seeds(identity: Identity) -> None:
    first = _chunk("seed-a", "Use the following steps:", next_id="next-a")
    second = _chunk("seed-b", "Use the following steps:", next_id="next-b")
    builder = ContextBuilder(
        reranker=FakeReranker(),
        completer=_completer(_Neighbors(_chunk("next-a"), _chunk("next-b"))),
        packer=ContextPacker(_config(), utf8_byte_count),
        generation_k=1,
        reranking_timeout_seconds=1,
    )

    outcome = await builder.build(
        query="question",
        candidates=[_ranked(first, 1), _ranked(second, 2)],
        identity=identity,
    )

    assert _ids(outcome.packed_context.chunks) == ["seed-a", "next-a"]


async def test_builder_counts_the_batch_on_one_worker(identity: Identity) -> None:
    threads: list[int] = []

    def count_tokens(text: str) -> int:
        threads.append(threading.get_ident())
        return len(text)

    builder = ContextBuilder(
        reranker=FakeReranker(),
        completer=_completer(_Neighbors()),
        packer=ContextPacker(_config(), count_tokens),
        generation_k=2,
        reranking_timeout_seconds=1,
    )
    candidates = [_ranked(_chunk("first", "First.")), _ranked(_chunk("second", "Second."), 2)]

    outcome = await builder.build(query="question", candidates=candidates, identity=identity)

    assert _ids(outcome.packed_context.chunks) == ["first", "second"]
    assert len(threads) == 2
    assert threads[0] == threads[1] != threading.get_ident()


def test_context_packer_limits_document_share_when_multiple_documents_exist() -> None:
    first = make_chunk("00000000-0000-0000-0000-000000000001").model_copy(
        update={"token_count": 100}
    )
    same_document = make_chunk("00000000-0000-0000-0000-000000000002", sequence=1).model_copy(
        update={"token_count": 100}
    )
    other_document = make_chunk(
        "00000000-0000-0000-0000-000000000003", document_id="doc-b"
    ).model_copy(update={"token_count": 100})
    ranked = [
        RankedChunk(chunk=chunk, fusion_score=1 / rank, fusion_rank=rank)
        for rank, chunk in enumerate((first, same_document, other_document), start=1)
    ]
    packer = ContextPacker(
        ContextConfig(
            max_tokens=256,
            max_seed_chunks=3,
            max_document_share=0.5,
            max_neighbors_per_seed=0,
        ),
        utf8_byte_count,
    )

    context = packer.pack(ranked)

    assert [item.chunk.id for item in context.chunks] == [first.id, other_document.id]
    assert context.token_count == 200


@pytest.mark.parametrize(
    ("other_text", "other_count"),
    [("short", 257), ("é" * 129, 20)],
    ids=["oversized-count", "oversized-utf8"],
)
def test_only_document_with_a_packable_seed_can_use_the_full_budget(
    other_text: str, other_count: int
) -> None:
    seed = make_chunk("seed", document_id="guide", text="é" * 128)
    oversized = make_chunk("appendix", document_id="manual", text=other_text).model_copy(
        update={"token_count": other_count}
    )
    neighbor = make_chunk("note", document_id="manual", text="A short note.")
    candidates = [_ranked(seed), _ranked(oversized, 2), _neighbor(neighbor, oversized.id)]

    packed = ContextPacker(_config(max_tokens=256, max_document_share=0.5), utf8_byte_count).pack(
        candidates
    )

    assert _ids(packed.chunks) == [seed.id]
    assert packed.token_count == 256


def test_packed_neighbours_keep_reading_order_around_their_seed() -> None:
    candidates = [
        _neighbor(_chunk("prev"), "seed"),
        _ranked(_chunk("seed")),
        _neighbor(_chunk("next"), "seed"),
    ]

    packed = ContextPacker(_config(), utf8_byte_count).pack(candidates)

    assert _ids(packed.chunks) == ["prev", "seed", "next"]
    assert packed.token_count == 600


def test_packer_caps_seeds_and_neighbours_independently() -> None:
    candidates = [
        _neighbor(_chunk("prev-1"), "seed-1"),
        _ranked(_chunk("seed-1"), 1),
        _neighbor(_chunk("next-1"), "seed-1"),
        _neighbor(_chunk("prev-2"), "seed-2"),
        _ranked(_chunk("seed-2"), 2),
        _neighbor(_chunk("next-2"), "seed-2"),
        _ranked(_chunk("seed-3"), 3),
    ]

    packed = ContextPacker(
        _config(max_seed_chunks=2, max_neighbors_per_seed=1), utf8_byte_count
    ).pack(candidates)

    assert _ids(packed.chunks) == ["prev-1", "seed-1", "next-1", "seed-2"]


def test_neighbours_of_an_unpacked_seed_are_dropped_even_when_they_fit() -> None:
    candidates = [
        _ranked(_chunk("seed-1", tokens=200), 1),
        _neighbor(_chunk("prev-2", tokens=10), "seed-2"),
        _ranked(_chunk("seed-2", tokens=200), 2),
        _neighbor(_chunk("next-2", tokens=10), "seed-2"),
        _ranked(_chunk("seed-3", "Third.", tokens=50), 3),
    ]

    packed = ContextPacker(_config(max_tokens=300), utf8_byte_count).pack(candidates)

    assert _ids(packed.chunks) == ["seed-1", "seed-3"]
    assert packed.token_count == 250


def test_neighbour_cannot_displace_its_seed_under_the_token_budget() -> None:
    candidates = [
        _neighbor(_chunk("prev", tokens=100), "seed"),
        _ranked(_chunk("seed", tokens=200)),
    ]

    packed = ContextPacker(_config(max_tokens=256), utf8_byte_count).pack(candidates)

    assert _ids(packed.chunks) == ["seed"]


def test_neighbour_without_a_candidate_anchor_is_never_packed() -> None:
    orphan = _neighbor(_chunk("orphan"), "missing")
    unanchored = RankedChunk(
        chunk=_chunk("loose"),
        fusion_score=0.5,
        fusion_rank=2,
        is_neighbor=True,
    )

    packed = ContextPacker(_config(), utf8_byte_count).pack(
        [orphan, unanchored, _ranked(_chunk("seed"))]
    )

    assert _ids(packed.chunks) == ["seed"]


def test_zero_neighbour_budget_packs_seeds_only() -> None:
    candidates = [
        _neighbor(_chunk("prev"), "seed"),
        _ranked(_chunk("seed")),
        _neighbor(_chunk("next"), "seed"),
    ]

    packed = ContextPacker(_config(max_neighbors_per_seed=0), utf8_byte_count).pack(candidates)

    assert _ids(packed.chunks) == ["seed"]


def test_duplicate_candidates_are_packed_once() -> None:
    seed = _ranked(_chunk("seed"))

    packed = ContextPacker(_config(), utf8_byte_count).pack([seed, seed])

    assert _ids(packed.chunks) == ["seed"]
    assert packed.token_count == 200


@pytest.mark.parametrize(("language", "ids"), [("en", ["seed", "next"]), ("fr", ["seed"])])
async def test_completion_respects_language_filter(
    identity: Identity, language: str, ids: list[str]
) -> None:
    seed = _chunk("seed", "The following:", next_id="next").model_copy(
        update={"metadata": {"lang": "en"}}
    )
    neighbor = _chunk("next").model_copy(update={"metadata": {"lang": language}})
    result = await _completer(_Neighbors(neighbor)).complete(
        [_ranked(seed)], identity, filters={"lang": "en"}
    )
    assert _ids(result) == ids


@pytest.mark.parametrize(
    ("text", "declared", "expected"),
    [
        ("é" * 512, 1, 0),
        ("é" * 512 + "a", 1, 0),
        ("a" * 1000, 256, 0),
        ("Short.", 256, 256),
        ("é" * 128, 1, 256),
        ("é" * 128 + "a", 1, 0),
    ],
    ids=["unicode", "unicode-plus-ascii", "ascii", "declared-floor", "exact-fit", "one-over"],
)
def test_byte_packing_charges_the_greater_of_bytes_and_declared_count(
    text: str, declared: int, expected: int
) -> None:
    chunk = make_chunk("page", text=text).model_copy(update={"token_count": declared})
    result = ContextPacker(_config(max_tokens=256), utf8_byte_count).pack([_ranked(chunk)])
    assert result.token_count == expected
    assert bool(result.chunks) == bool(expected)


@pytest.mark.parametrize(
    ("text", "measured", "declared", "budget", "expected"),
    [
        (" Authentication is required for every transaction." * 50, 350, 350, 512, 350),
        ("é" * 512, 512, 1, 256, 0),
        ("é" * 256, 256, 1, 256, 256),
        ("Short.", 2, 256, 256, 256),
    ],
    ids=["accurate-prose", "underdeclared-unicode", "exact-fit", "declared-floor"],
)
def test_tokenizer_packing_uses_the_greater_of_local_tokens_and_declared_count(
    text: str, measured: int, declared: int, budget: int, expected: int
) -> None:
    chunk = make_chunk("page", text=text).model_copy(update={"token_count": declared})
    counter = {text: measured}.__getitem__

    result = ContextPacker(_config(max_tokens=budget), counter).pack([_ranked(chunk)])

    assert result.token_count == expected
    assert bool(result.chunks) == bool(expected)
