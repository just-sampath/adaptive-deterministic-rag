from __future__ import annotations

import pytest

from models.domain import Chunk, Citation
from services.citations import CitationRenderer
from tests.fakes import make_chunk


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-a",
    title: str = "Guide",
    heading: tuple[str, ...] = ("Test",),
    version: int = 1,
) -> Chunk:
    return make_chunk(chunk_id, document_id=document_id).model_copy(
        update={
            "document_title": title,
            "heading_path": heading,
            "document_version": version,
        }
    )


def test_citations_are_numbered_by_first_appearance_and_sorted_within_a_marker() -> None:
    evidence = {"E1": _chunk("chunk-1"), "E2": _chunk("chunk-2", title="Manual")}

    rendered = CitationRenderer(inline=True).render(
        "Second fact [E2]. First fact [E1, E2].",
        evidence=evidence,
        verified_chunk_ids={"chunk-1", "chunk-2"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Second fact [1]. First fact [1, 2]."
    assert rendered.citations == (
        Citation(number=1, document_id="doc-a", title="Manual", section="Test", version=1),
        Citation(number=2, document_id="doc-a", title="Guide", section="Test", version=1),
    )


def test_unknown_unverified_and_disallowed_labels_are_removed_with_their_leading_space() -> None:
    evidence = {
        "E1": _chunk("chunk-1"),
        "E2": _chunk("chunk-2"),
        "E3": _chunk("chunk-3", document_id="doc-b"),
    }

    rendered = CitationRenderer(inline=True).render(
        "Kept [E1]. Unknown [E9] here. Unverified [E2]. Blocked [E3] .",
        evidence=evidence,
        verified_chunk_ids={"chunk-1", "chunk-3"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Kept [1]. Unknown here. Unverified. Blocked."
    assert [item.document_id for item in rendered.citations] == ["doc-a"]


def test_mixed_marker_keeps_only_its_usable_labels() -> None:
    evidence = {"E1": _chunk("chunk-1")}

    rendered = CitationRenderer(inline=True).render(
        "Fact [E1, E9].",
        evidence=evidence,
        verified_chunk_ids={"chunk-1"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Fact [1]."


def test_removed_marker_leaves_no_space_before_following_punctuation_or_double_gap() -> None:
    rendered = CitationRenderer(inline=True).render(
        "Fact [E9] . Next [E9], then  kept spacing  here. Gap [E9]  here.",
        evidence={},
        verified_chunk_ids=set(),
        allowed_document_ids=set(),
    )

    assert rendered.text == "Fact. Next, then  kept spacing  here. Gap here."
    assert rendered.citations == ()


def test_removed_marker_at_the_start_leaves_no_leading_space() -> None:
    rendered = CitationRenderer(inline=True).render(
        "[E9] Yes, it is. [E1] Next.",
        evidence={"E1": _chunk("chunk-1")},
        verified_chunk_ids={"chunk-1"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Yes, it is. [1] Next."


def test_inline_disabled_removes_every_marker_and_returns_no_citations() -> None:
    evidence = {"E1": _chunk("chunk-1")}

    rendered = CitationRenderer(inline=False).render(
        "Fact [E1] here [E9].",
        evidence=evidence,
        verified_chunk_ids={"chunk-1"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Fact here."
    assert rendered.citations == ()


def test_citation_section_joins_heading_path_and_is_none_without_headings() -> None:
    evidence = {
        "E1": _chunk("chunk-1", heading=("Limits", "Quotas"), version=3, title="Policy"),
        "E2": _chunk("chunk-2", heading=()),
    }

    rendered = CitationRenderer(inline=True).render(
        "Quota [E1] and other [E2].",
        evidence=evidence,
        verified_chunk_ids={"chunk-1", "chunk-2"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.citations == (
        Citation(
            number=1, document_id="doc-a", title="Policy", section="Limits > Quotas", version=3
        ),
        Citation(number=2, document_id="doc-a", title="Guide", section=None, version=1),
    )
    assert rendered.text == "Quota [1] and other [2]."


def test_text_outside_markers_is_left_unchanged() -> None:
    evidence = {"E1": _chunk("chunk-1")}

    rendered = CitationRenderer(inline=True).render(
        "Indented  text [E1]\n    code ,  kept",
        evidence=evidence,
        verified_chunk_ids={"chunk-1"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Indented  text [1]\n    code ,  kept"


def test_adjacent_removed_markers_leave_one_separator_before_punctuation() -> None:
    rendered = CitationRenderer(inline=True).render(
        "The limit is 5 [E9] [E8]. Fact [E9] [E8], next. A [E9] [E8] b.",
        evidence={},
        verified_chunk_ids=set(),
        allowed_document_ids=set(),
    )

    assert rendered.text == "The limit is 5. Fact, next. A b."


def test_valid_marker_followed_by_removed_marker_keeps_one_space() -> None:
    rendered = CitationRenderer(inline=True).render(
        "Kept [E1] [E9] more.",
        evidence={"E1": _chunk("chunk-1")},
        verified_chunk_ids={"chunk-1"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Kept [1] more."


def test_removed_markers_are_settled_when_inline_citations_are_disabled() -> None:
    rendered = CitationRenderer(inline=False).render(
        "Fact [E1] [E2]. Next [E1], then.",
        evidence={"E1": _chunk("chunk-1"), "E2": _chunk("chunk-2")},
        verified_chunk_ids={"chunk-1", "chunk-2"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Fact. Next, then."
    assert rendered.citations == ()


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Revenue [E9]'s growth", "Revenue's growth"),
        ("**Revenue [E9]** rose", "**Revenue** rose"),
        ("Kept [E9][E1].", "Kept [1]."),
        ("Unknown [E9] here", "Unknown here"),
        ("[E9] Yes, it is.", "Yes, it is."),
        ("  [E9] indented", "  indented"),
        ("Fact.[E9] Next", "Fact. Next"),
    ],
)
def test_removed_marker_spacing_follows_the_text_around_it(answer: str, expected: str) -> None:
    rendered = CitationRenderer(inline=True).render(
        answer,
        evidence={"E1": _chunk("chunk-1")},
        verified_chunk_ids={"chunk-1"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == expected


def test_bracketed_numbers_and_code_indexing_are_text_and_never_cited() -> None:
    rendered = CitationRenderer(inline=True).render(
        "Use items[0] and arr[1] here [E1].",
        evidence={"E1": _chunk("chunk-1")},
        verified_chunk_ids={"chunk-1"},
        allowed_document_ids={"doc-a"},
    )

    assert rendered.text == "Use items[0] and arr[1] here [1]."
    assert rendered.citations == (
        Citation(number=1, document_id="doc-a", title="Guide", section="Test", version=1),
    )


def test_without_markers_removes_every_marker_for_conversation_history() -> None:
    stripped = CitationRenderer(inline=True).without_markers(
        "Limitations are cost [E2] and latency [E1]. Plain [E3, E4] text. Code arr[1]."
    )

    assert stripped == "Limitations are cost and latency. Plain text. Code arr[1]."
