"""Authorize complete conversation turns before reusing their text in a prompt."""

from collections.abc import Iterable, Mapping, Sequence

from models.domain import Chunk, ConversationMessage, EvidenceRef, Identity
from services.contracts import DocumentReader
from services.understanding import project_history
from utils.prompts import STANDARD_ABSTENTION


def evidence_withdrawal_reason(
    ref: EvidenceRef, chunk: Chunk | None, filters: Mapping[str, str | int | float | bool]
) -> str | None:
    """Compare a stored passage reference with the authoritative read for this caller."""
    if chunk is None:
        return "unavailable"
    if not chunk.matches_filters(filters):
        return "metadata"
    if EvidenceRef.from_chunk(chunk) != ref:
        return "changed"
    return None


def new_evidence_refs(
    existing: Mapping[str, EvidenceRef], incoming: Iterable[EvidenceRef], limit: int
) -> dict[str, EvidenceRef] | None:
    """Return the additions if the complete union fits and each ID has a single snapshot."""
    if len(existing) > limit:
        return None
    additions: dict[str, EvidenceRef] = {}
    for ref in incoming:
        known = existing.get(ref.chunk_id) or additions.get(ref.chunk_id)
        if known is None:
            if len(existing) + len(additions) >= limit:
                return None
            additions[ref.chunk_id] = ref
        elif known != ref:
            return None
    return additions


async def authorized_history(
    messages: Sequence[ConversationMessage],
    *,
    documents: DocumentReader,
    identity: Identity,
    filters: Mapping[str, str | int | float | bool],
    reference_limit: int,
    character_limit: int,
) -> tuple[tuple[ConversationMessage, ...], tuple[EvidenceRef, ...]]:
    """Return the prompt-visible suffix and all dependencies of its admitted pairs."""
    projected = project_history(messages, character_limit)
    refs: dict[str, EvidenceRef] = {}
    additions_by_pair: list[tuple[EvidenceRef, ...]] = []
    # Walk the original pairs even when the first visible line is an assistant excerpt.
    for offset in range(0, len(projected), 2):
        end = len(messages) - offset
        if end < 2:
            break
        user, assistant = messages[end - 2], messages[end - 1]
        if user.role != "user" or assistant.role != "assistant" or user.evidence_refs is not None:
            break
        pair_refs = assistant.evidence_refs
        if pair_refs is None or (not pair_refs and assistant.content != STANDARD_ABSTENTION):
            break
        additions = new_evidence_refs(refs, pair_refs, reference_limit)
        if additions is None:
            break
        refs.update(additions)
        additions_by_pair.append(tuple(additions.values()))

    current = await documents.hydrate_authorized_chunks(tuple(refs), identity) if refs else {}
    accepted: list[EvidenceRef] = []
    message_count = 0
    for additions_for_pair in additions_by_pair:
        # Each distinct passage belongs to the newest pair that needs it. Reaching an
        # older pair means the references it shares with newer pairs have already passed.
        if any(
            evidence_withdrawal_reason(ref, current.get(ref.chunk_id), filters)
            for ref in additions_for_pair
        ):
            break
        accepted.extend(additions_for_pair)
        message_count += 2
    if not message_count:
        return (), ()
    return projected[-message_count:], tuple(accepted)
