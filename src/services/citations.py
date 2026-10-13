"""Render evidence labels in generated answers as verified, numbered citations."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from models.domain import Chunk, Citation

# Only evidence labels are markers; bracketed numbers and code indexing such as items[0] are text.
_MARKER = re.compile(r"([ \t]*)\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")
_LABEL = re.compile(r"E\d+")
_CLOSING_PUNCTUATION = frozenset(".,;:!?)")


@dataclass(frozen=True, slots=True)
class RenderedAnswer:
    text: str
    citations: tuple[Citation, ...]


class CitationRenderer:
    """Number cited evidence by first appearance and remove markers that cite nothing usable."""

    def __init__(self, *, inline: bool) -> None:
        self._inline = inline

    def render(
        self,
        answer: str,
        *,
        evidence: Mapping[str, Chunk],
        verified_chunk_ids: Collection[str],
        allowed_document_ids: Collection[str],
    ) -> RenderedAnswer:
        usable: dict[str, Chunk] = {}
        if self._inline:
            usable = _usable_evidence(evidence, verified_chunk_ids, allowed_document_ids)
        numbers = _number_chunks(answer, usable)
        return RenderedAnswer(
            text=_substitute(answer, usable, numbers),
            citations=_citations(numbers, usable),
        )

    def without_markers(self, answer: str) -> str:
        """Return the answer without any markers, the form kept in conversation history."""

        return _substitute(answer, {}, {})


def _substitute(answer: str, usable: Mapping[str, Chunk], numbers: Mapping[str, int]) -> str:
    site = _unused_character(answer)
    marked = _MARKER.sub(lambda match: _replace_marker(match, usable, numbers, site), answer)
    return re.sub(rf"{re.escape(site)}+([ \t]*)", _site_separator, marked)


def _usable_evidence(
    evidence: Mapping[str, Chunk],
    verified_chunk_ids: Collection[str],
    allowed_document_ids: Collection[str],
) -> dict[str, Chunk]:
    return {
        label: chunk
        for label, chunk in evidence.items()
        if chunk.id in verified_chunk_ids and chunk.document_id in allowed_document_ids
    }


def _number_chunks(answer: str, usable: Mapping[str, Chunk]) -> dict[str, int]:
    """Assign citation numbers to chunk IDs in the order their labels first appear."""

    numbers: dict[str, int] = {}
    for match in _MARKER.finditer(answer):
        for label in _labels(match):
            chunk = usable.get(label)
            if chunk is not None and chunk.id not in numbers:
                numbers[chunk.id] = len(numbers) + 1
    return numbers


def _cited_numbers(
    match: re.Match[str],
    usable: Mapping[str, Chunk],
    numbers: Mapping[str, int],
) -> tuple[int, ...]:
    cited = {numbers[usable[label].id] for label in _labels(match) if label in usable}
    return tuple(sorted(cited))


def _citations(numbers: Mapping[str, int], usable: Mapping[str, Chunk]) -> tuple[Citation, ...]:
    chunks = {chunk.id: chunk for chunk in usable.values()}
    return tuple(
        Citation(
            number=number,
            document_id=chunks[chunk_id].document_id,
            title=chunks[chunk_id].document_title,
            section=" > ".join(chunks[chunk_id].heading_path) or None,
            version=chunks[chunk_id].document_version,
        )
        for chunk_id, number in sorted(numbers.items(), key=lambda item: item[1])
    )


def _replace_marker(
    match: re.Match[str],
    usable: Mapping[str, Chunk],
    numbers: Mapping[str, int],
    site: str,
) -> str:
    """Render a marker that cites something; mark a removed one for _site_separator."""

    cited = _cited_numbers(match, usable, numbers)
    if cited:
        rendered = ", ".join(str(number) for number in cited)
        return f"{match.group(1)}[{rendered}]"
    # A removed marker takes the spacing before it, but keeps indentation at the start of a line.
    indentation = match.group(1) if _starts_line(match.string, match.start()) else ""
    return indentation + site


def _site_separator(match: re.Match[str]) -> str:
    """Choose the separator at a removal site from the text on either side of it.

    Runs of adjacent removed markers form one site, so the decision sees the real neighbours.
    """

    text = match.string
    following = text[match.end()] if match.end() < len(text) else ""
    if (
        not following
        or following in _CLOSING_PUNCTUATION
        or following in ("\r", "\n")
        or _starts_line(text, match.start())
    ):
        return ""
    return " " if match.group(1) or following == "[" else ""


def _starts_line(text: str, position: int) -> bool:
    """Return whether only horizontal whitespace precedes the position on its line."""

    line_start = text.rfind("\n", 0, position) + 1
    return not text[line_start:position].strip(" \t")


def _unused_character(answer: str) -> str:
    """Return a private-use character that the answer does not contain, to mark removal sites."""

    return next(chr(code) for code in range(0xE000, 0xF900) if chr(code) not in answer)


def _labels(match: re.Match[str]) -> tuple[str, ...]:
    return tuple(_LABEL.findall(match.group(2)))
