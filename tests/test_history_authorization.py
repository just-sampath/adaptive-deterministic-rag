from collections.abc import Sequence

import pytest

from models.domain import Chunk, ConversationMessage, EvidenceRef, Identity
from services.history import authorized_history
from tests.fakes import FakeRepository, make_chunk
from utils.errors import ProviderResponseError
from utils.prompts import STANDARD_ABSTENTION


def pair(
    label: str, refs: tuple[EvidenceRef, ...] | None, *, answer: str | None = None
) -> tuple[ConversationMessage, ConversationMessage]:
    return (
        ConversationMessage(role="user", content=f"Ask about {label}"),
        ConversationMessage(role="assistant", content=answer or label, evidence_refs=refs),
    )


async def test_full_window_deduplicates_before_read_and_stops_at_overflow(
    identity: Identity,
) -> None:
    chunks = [make_chunk(str(i)) for i in range(4)]
    a, b, c, overflow = map(EvidenceRef.from_chunk, chunks)
    documents = FakeRepository(chunks)
    newest = pair("new", (b, a, b))
    middle = pair("middle", (a, c))
    older = tuple(message for index in range(46) for message in pair(f"older {index}", (b, c)))
    over_budget = pair("overflow", (a, overflow))
    history, refs = await authorized_history(
        (*pair("behind overflow", (a,)), *over_budget, *older, *middle, *newest),
        documents=documents,
        identity=identity,
        filters={},
        reference_limit=3,
        character_limit=10_000,
    )
    assert history == (*older, *middle, *newest)
    assert set(refs) == {a, b, c}
    assert len(refs) == 3
    assert len(documents.hydrate_calls) == 1
    assert set(documents.hydrate_calls[0]) == {"0", "1", "2"}
    assert len(documents.hydrate_calls[0]) == 3


@pytest.mark.parametrize("revocation", ["access", "tenant", "superseded", "digest", "filter"])
async def test_changed_older_pair_keeps_newer_pair_allowed_through_another_role(
    identity: Identity, revocation: str
) -> None:
    older = make_chunk("old", document_id="old-doc").model_copy(update={"metadata": {"lang": "en"}})
    newer = make_chunk("new", roles=frozenset({"other", "reader"})).model_copy(
        update={"metadata": {"lang": "en"}}
    )
    old_ref, new_ref = map(EvidenceRef.from_chunk, (older, newer))
    current_identity = identity.model_copy(update={"roles": frozenset({"other"})})
    # The alternate role still permits the newer message and the baseline older chunk.
    older = older.model_copy(update={"allowed_roles": frozenset({"reader", "other"})})
    changes: dict[str, dict[str, object]] = {
        "access": {"allowed_roles": frozenset({"reader"})},
        "tenant": {"tenant_id": "elsewhere"},
        "superseded": {"document_version_id": "changed"},
        "digest": {"text": "Updated without changing the version."},
        "filter": {"metadata": {"lang": "fr"}},
    }
    documents = FakeRepository([older.model_copy(update=changes[revocation]), newer])
    recent_pair = pair("recent", (new_ref,))
    history, refs = await authorized_history(
        (*pair("further back", (new_ref,)), *pair("old", (new_ref, old_ref)), *recent_pair),
        documents=documents,
        identity=current_identity,
        filters={"lang": "en"},
        reference_limit=3,
        character_limit=500,
    )
    assert history == recent_pair
    assert refs == (new_ref,)


@pytest.mark.parametrize("kind", ["legacy", "fake-abstention", "unknown-abstention", "conflict"])
async def test_unsafe_newest_pair_blocks_every_older_pair_before_hydration(
    identity: Identity, kind: str
) -> None:
    chunk = make_chunk("a")
    ref = EvidenceRef.from_chunk(chunk)
    mismatch = ref.model_copy(update={"passage_digest": "different"})
    candidates = {
        "legacy": pair("unknown", None),
        "fake-abstention": pair("secret", ()),
        "unknown-abstention": pair("unknown", None, answer=STANDARD_ABSTENTION),
        "conflict": pair("conflict", (ref, mismatch)),
    }
    documents = FakeRepository([chunk])
    assert await authorized_history(
        (*pair("eligible but behind unsafe pair", (ref,)), *candidates[kind]),
        documents=documents,
        identity=identity,
        filters={},
        reference_limit=3,
        character_limit=500,
    ) == ((), ())
    assert documents.hydrate_calls == []


async def test_conflicting_older_snapshot_cannot_poison_the_newest_one(identity: Identity) -> None:
    chunk = make_chunk("c")
    ref = EvidenceRef.from_chunk(chunk)
    old_ref = ref.model_copy(update={"document_version_id": "previous"})
    documents = FakeRepository([chunk])
    recent = pair("latest", (ref,))
    history, refs = await authorized_history(
        (*pair("previous version", (old_ref,)), *recent),
        documents=documents,
        identity=identity,
        filters={},
        reference_limit=3,
        character_limit=500,
    )
    assert (history, refs) == (recent, (ref,))


async def test_abstention_after_clipped_pair_needs_no_store_read(identity: Identity) -> None:
    documents = FakeRepository()
    recent = pair("unknown answer", (), answer=STANDARD_ABSTENTION)
    history, refs = await authorized_history(
        (ConversationMessage(role="assistant", content="clipped pair"), *recent),
        documents=documents,
        identity=identity,
        filters={},
        reference_limit=3,
        character_limit=500,
    )
    assert (history, refs) == (recent, ())
    assert documents.hydrate_calls == []


async def test_unpaired_newest_message_blocks_history(identity: Identity) -> None:
    chunk = make_chunk("c")
    ref = EvidenceRef.from_chunk(chunk)
    documents = FakeRepository([chunk])
    assert await authorized_history(
        (*pair("whole pair", (ref,)), ConversationMessage(role="user", content="unanswered")),
        documents=documents,
        identity=identity,
        filters={},
        reference_limit=2,
        character_limit=500,
    ) == ((), ())
    assert documents.hydrate_calls == []


async def test_newest_excerpt_keeps_all_pair_refs_and_skips_older_read(identity: Identity) -> None:
    chunks = [make_chunk(str(i)) for i in range(3)]
    older, a, b = map(EvidenceRef.from_chunk, chunks)
    documents = FakeRepository(chunks)
    recent = pair("newest", (a, b), answer="猫" * 600)
    history, refs = await authorized_history(
        (*pair("old", (older,)), *recent),
        documents=documents,
        identity=identity,
        filters={},
        reference_limit=3,
        character_limit=500,
    )
    assert len(history) == 1
    assert history[0].content == "猫" * (500 - len("assistant: "))
    assert set(refs) == {a, b}
    assert documents.hydrate_calls == [("1", "2")]
    assert recent[1].content == "猫" * 600


async def test_failed_authoritative_read_propagates(identity: Identity) -> None:
    class Unavailable(FakeRepository):
        async def hydrate_authorized_chunks(
            self, chunk_ids: Sequence[str], identity: Identity
        ) -> dict[str, Chunk]:
            raise ProviderResponseError("Document store unavailable")

    with pytest.raises(ProviderResponseError):
        await authorized_history(
            pair("blocked", (EvidenceRef.from_chunk(make_chunk("c")),)),
            documents=Unavailable(),
            identity=identity,
            filters={},
            reference_limit=3,
            character_limit=500,
        )
