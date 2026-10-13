"""Reranking, structural completion, and bounded context packing."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from models.domain import Chunk, Identity, PackedContext, RankedChunk
from services.contracts import DocumentReader, Reranker
from utils.settings import ContextConfig

_MID_LIST = re.compile(r"^\s*(?:[2-9]|\d{2,})[.)]\s")
_BACK_REFERENCE = re.compile(
    r"\b(?:as described above|the previous|the former|the latter|this process|these steps)\b",
    re.IGNORECASE,
)
_FORWARD_REFERENCE = re.compile(
    r"\b(?:the following|as follows|continued below)\b",
    re.IGNORECASE,
)
_UNRESOLVED_START = re.compile(
    r"^\s*(?:this|these|it|its|they|their|that|those|such)\b",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^(?:\d+[.)]|[-*+])(?:\s|$)")
_FENCE = "```"
_Requested = tuple[str | None, str | None]


@dataclass(frozen=True, slots=True)
class ContextOutcome:
    reranked_chunks: tuple[RankedChunk, ...]
    packed_context: PackedContext


class StructuralCompleter:
    """Fetch only neighbours indicated by boundary or reference signals."""

    def __init__(
        self,
        *,
        repository: DocumentReader,
        max_neighbors_per_seed: int,
        short_chunk_tokens: int,
    ) -> None:
        self._repository = repository
        self._max_neighbors_per_seed = max_neighbors_per_seed
        self._short_chunk_tokens = short_chunk_tokens

    async def complete(
        self,
        seeds: Sequence[RankedChunk],
        identity: Identity,
    ) -> tuple[RankedChunk, ...]:
        unique = _first_by_id(seeds)
        requested = {seed.chunk.id: self._requested(seed.chunk) for seed in unique}
        neighbors = await self._fetch(requested, identity)
        # A neighbour may be offered to several anchors; the packer places it once.
        seed_ids = {seed.chunk.id for seed in unique}
        completed: list[RankedChunk] = []
        for seed in unique:
            previous, following = self._admit_neighbors(
                seed.chunk,
                requested[seed.chunk.id],
                neighbors,
                seed_ids,
            )
            if previous is not None:
                completed.append(_as_neighbor(seed, previous))
            completed.append(seed)
            if following is not None:
                completed.append(_as_neighbor(seed, following))
        return tuple(completed)

    def _requested(self, chunk: Chunk) -> _Requested:
        body = _body(chunk.text)
        short = chunk.token_count < self._short_chunk_tokens
        previous = chunk.previous_chunk_id if short or _wants_previous(body) else None
        following = chunk.next_chunk_id if short or _wants_next(body) else None
        return previous, following

    async def _fetch(
        self,
        requested: Mapping[str, _Requested],
        identity: Identity,
    ) -> dict[str, Chunk]:
        ids = dict.fromkeys(
            neighbor_id
            for pair in requested.values()
            for neighbor_id in pair
            if neighbor_id is not None
        )
        if not ids:
            return {}
        return await self._repository.get_neighbors(list(ids), identity)

    def _admit_neighbors(
        self,
        seed: Chunk,
        requested: _Requested,
        neighbors: Mapping[str, Chunk],
        seed_ids: set[str],
    ) -> tuple[Chunk | None, Chunk | None]:
        previous_id, following_id = requested
        previous = _admit_one(seed, previous_id, neighbors, seed_ids, self._max_neighbors_per_seed)
        remaining = self._max_neighbors_per_seed - (previous is not None)
        following = _admit_one(seed, following_id, neighbors, seed_ids, remaining)
        return previous, following


class ContextPacker:
    """Greedily pack ranked evidence within token, count, and document-share limits."""

    def __init__(self, config: ContextConfig) -> None:
        self._config = config

    def pack(self, candidates: Sequence[RankedChunk]) -> PackedContext:
        unique = _without_duplicate_seeds(candidates)
        packing = _Packing(self._config, unique)
        for position, item in enumerate(unique):
            packing.offer(position, item)
        chunks = tuple(item for position, item in enumerate(unique) if position in packing.chosen)
        return PackedContext(chunks=chunks, token_count=packing.tokens)


class _Packing:
    """One greedy pass; a neighbour is decided only after its anchor seed is decided.

    Items are tracked by candidate position. A neighbour offered by several anchors is admitted
    at most once, under the first anchor whose seed was packed.
    """

    def __init__(self, config: ContextConfig, candidates: Sequence[RankedChunk]) -> None:
        self._config = config
        self._max_neighbors = config.max_seed_chunks * config.max_neighbors_per_seed
        self._document_cap = _document_cap(config, candidates)
        self._seed_decisions: dict[str, bool] = {}
        self._waiting: dict[str, list[tuple[int, RankedChunk]]] = {}
        self._chosen_ids: set[str] = set()
        self._document_tokens: dict[str, int] = {}
        self._seeds = 0
        self._neighbors = 0
        self.chosen: set[int] = set()
        self.tokens = 0

    def offer(self, position: int, item: RankedChunk) -> None:
        if item.is_neighbor:
            self._offer_neighbor(position, item)
        else:
            self._offer_seed(position, item)

    def _offer_seed(self, position: int, item: RankedChunk) -> None:
        chunk = item.chunk
        packed = self._seeds < self._config.max_seed_chunks and self._fits(chunk)
        self._seed_decisions[chunk.id] = packed
        if packed:
            self._seeds += 1
            self._take(position, chunk)
        for waiting_position, neighbor in self._waiting.pop(chunk.id, []):
            self._offer_neighbor(waiting_position, neighbor)

    def _offer_neighbor(self, position: int, item: RankedChunk) -> None:
        anchor = item.anchor_chunk_id
        if anchor is None:
            return
        if anchor not in self._seed_decisions:
            self._waiting.setdefault(anchor, []).append((position, item))
        elif self._admits_neighbor(anchor, item.chunk):
            self._neighbors += 1
            self._take(position, item.chunk)

    def _admits_neighbor(self, anchor: str, chunk: Chunk) -> bool:
        return (
            self._seed_decisions[anchor]
            and chunk.id not in self._chosen_ids
            and self._neighbors < self._max_neighbors
            and self._fits(chunk)
        )

    def _fits(self, chunk: Chunk) -> bool:
        document_total = self._document_tokens.get(chunk.document_id, 0) + chunk.token_count
        return (
            self.tokens + chunk.token_count <= self._config.max_tokens
            and document_total <= self._document_cap
        )

    def _take(self, position: int, chunk: Chunk) -> None:
        self.chosen.add(position)
        self._chosen_ids.add(chunk.id)
        self.tokens += chunk.token_count
        self._document_tokens[chunk.document_id] = (
            self._document_tokens.get(chunk.document_id, 0) + chunk.token_count
        )


class ContextBuilder:
    """Convert broad retrieval candidates into narrow generation evidence."""

    def __init__(
        self,
        *,
        reranker: Reranker,
        completer: StructuralCompleter,
        packer: ContextPacker,
        generation_k: int,
        reranking_timeout_seconds: float,
    ) -> None:
        self._reranker = reranker
        self._completer = completer
        self._packer = packer
        self._generation_k = generation_k
        self._reranking_timeout_seconds = reranking_timeout_seconds

    async def build(
        self,
        *,
        query: str,
        candidates: Sequence[RankedChunk],
        identity: Identity,
    ) -> ContextOutcome:
        async with asyncio.timeout(self._reranking_timeout_seconds):
            reranked = await self._reranker.rerank(query, candidates, len(candidates))
        seeds = reranked[: self._generation_k]
        completed = await self._completer.complete(seeds, identity)
        return ContextOutcome(
            reranked_chunks=tuple(reranked),
            packed_context=self._packer.pack(completed),
        )


def _body(text: str) -> str:
    parts = text.split("\n\n", maxsplit=1)
    return parts[1] if len(parts) == 2 else text


def _edge_lines(body: str) -> tuple[str, str]:
    lines = [stripped for raw in body.splitlines() if (stripped := raw.strip())]
    return (lines[0], lines[-1]) if lines else ("", "")


def _wants_previous(body: str) -> bool:
    first, _ = _edge_lines(body)
    return bool(
        _MID_LIST.search(body)
        or _BACK_REFERENCE.search(body[:300])
        or _UNRESOLVED_START.search(body)
        or first.startswith("|")
        or (_has_open_fence(body) and not _starts_with_fence(body))
    )


def _wants_next(body: str) -> bool:
    _, last = _edge_lines(body)
    return bool(
        _FORWARD_REFERENCE.search(body[-500:])
        or body.rstrip().endswith(":")
        or _LIST_ITEM.match(last)
        or last.startswith("|")
        or (_has_open_fence(body) and _starts_with_fence(body))
    )


def _has_open_fence(body: str) -> bool:
    return body.count(_FENCE) % 2 == 1


def _starts_with_fence(body: str) -> bool:
    return body.lstrip().startswith(_FENCE)


def _admit_one(
    seed: Chunk,
    neighbor_id: str | None,
    neighbors: Mapping[str, Chunk],
    seed_ids: set[str],
    slots: int,
) -> Chunk | None:
    if neighbor_id is None or slots <= 0:
        return None
    neighbor = neighbors.get(neighbor_id)
    if neighbor is None or neighbor.id in seed_ids or not _same_section(seed, neighbor):
        return None
    return neighbor


def _same_section(seed: Chunk, neighbor: Chunk) -> bool:
    return (
        neighbor.document_id == seed.document_id
        and neighbor.document_version_id == seed.document_version_id
        and neighbor.section_id == seed.section_id
    )


def _as_neighbor(seed: RankedChunk, chunk: Chunk) -> RankedChunk:
    return seed.model_copy(
        update={"chunk": chunk, "is_neighbor": True, "anchor_chunk_id": seed.chunk.id}
    )


def _first_by_id(items: Iterable[RankedChunk]) -> tuple[RankedChunk, ...]:
    first: dict[str, RankedChunk] = {}
    for item in items:
        first.setdefault(item.chunk.id, item)
    return tuple(first.values())


def _without_duplicate_seeds(items: Sequence[RankedChunk]) -> tuple[RankedChunk, ...]:
    """Keep the first copy of each seed and drop neighbours that are seeds.

    Repeated neighbours are kept: each anchor offers its own copy, and the packer admits one.
    """

    seed_ids = {item.chunk.id for item in items if not item.is_neighbor}
    kept: list[RankedChunk] = []
    seen: set[str] = set()
    for item in items:
        if item.is_neighbor:
            if item.chunk.id not in seed_ids:
                kept.append(item)
        elif item.chunk.id not in seen:
            seen.add(item.chunk.id)
            kept.append(item)
    return tuple(kept)


def _document_cap(config: ContextConfig, candidates: Sequence[RankedChunk]) -> int:
    documents = {item.chunk.document_id for item in candidates}
    if len(documents) > 1:
        return int(config.max_tokens * config.max_document_share)
    return config.max_tokens
