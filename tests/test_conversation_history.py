from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, cast

import pytest
from bson import BSON
from pydantic import BaseModel, ConfigDict, ValidationError
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.common import MAX_BSON_SIZE
from pymongo.errors import (
    AutoReconnect,
    DocumentTooLarge,
    OperationFailure,
    WriteConcernError,
    WriteError,
)

from managers.conversations import MongoConversationManager
from models.domain import ConversationMessage, EvidenceRef, Identity
from tests.fakes import make_chunk
from utils.errors import InfrastructureError


class LegacyMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str
    created_at: datetime


class ConversationCollection:
    """Apply the single-document update and reject an oversized result before applying it."""

    def __init__(self, max_size: int = MAX_BSON_SIZE) -> None:
        self.max_size = max_size
        self.docs: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.writes = 0
        self.failures: list[Exception] = []

    @staticmethod
    def key(query: dict[str, Any]) -> tuple[str, str, str]:
        return (query["tenant_id"], query["user_id"], query["id"])

    async def update_one(
        self, query: dict[str, Any], update: dict[str, Any], *, upsert: bool
    ) -> None:
        assert upsert
        await asyncio.sleep(0)
        self.writes += 1
        if self.failures:
            raise self.failures.pop(0)
        key = self.key(query)
        doc = self.docs.get(key, {**update["$setOnInsert"], "messages": []})
        push = update["$push"]["messages"]
        result = {
            **doc,
            **update["$set"],
            "messages": (doc["messages"] + push["$each"])[push["$slice"] :],
        }
        encoded = BSON.encode(result)
        if len(encoded) > self.max_size:
            raise WriteError("Resulting document is larger than the limit", 17419)
        self.docs[key] = BSON(encoded).decode()

    async def find_one(
        self, query: dict[str, Any], projection: dict[str, Any]
    ) -> dict[str, Any] | None:
        doc = self.docs.get(self.key(query))
        if doc is None:
            return None
        return {"messages": doc["messages"][projection["messages"]["$slice"] :]}


def manager(
    collection: ConversationCollection, *, write_provenance: bool = False, max_refs: int = 4
) -> MongoConversationManager:
    database = cast(AsyncDatabase[dict[str, Any]], {"conversations": collection})
    return MongoConversationManager(
        database, write_provenance=write_provenance, max_history_refs=max_refs
    )


def pair(
    name: str, refs: tuple[EvidenceRef, ...] | None
) -> tuple[ConversationMessage, ConversationMessage]:
    return (
        ConversationMessage(role="user", content=f"question {name}"),
        ConversationMessage(role="assistant", content=f"answer {name}", evidence_refs=refs),
    )


async def test_old_and_compatible_readers_consume_actual_bson_at_each_rollout_stage(
    identity: Identity,
) -> None:
    collection = ConversationCollection()
    compatibility = manager(collection)
    enriched = manager(collection, write_provenance=True)
    ref = EvidenceRef.from_chunk(make_chunk("handbook"))
    await compatibility.append_messages(identity, "shared", pair("old", (ref,)))
    stored = collection.docs[("tenant-a", "user-a", "shared")]["messages"]
    for row in stored:
        assert LegacyMessage.model_validate(row).content
        assert set(row) == {"role", "content", "created_at"}

    await enriched.append_messages(identity, "shared", pair("unknown", None))
    await enriched.append_messages(identity, "shared", pair("source-free", ()))
    await enriched.append_messages(identity, "shared", pair("known", (ref,)))
    stored = collection.docs[("tenant-a", "user-a", "shared")]["messages"]
    assert all("evidence_refs" not in row for row in stored[::2])
    assert "evidence_refs" not in stored[3]
    assert stored[5]["evidence_refs"] == []
    assert stored[7]["evidence_refs"] == [ref.model_dump()]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LegacyMessage.model_validate(stored[7])
    stored[3]["evidence_refs"] = None  # Also accept explicit BSON null from a compatible writer.

    for reader in (compatibility, enriched):
        tail = await reader.get_messages(identity, "shared", 7)
        assert [row.content for row in tail] == [row["content"] for row in stored[-7:]]
        assert [row.evidence_refs for row in tail if row.role == "assistant"] == [
            None,
            None,
            (),
            (ref,),
        ]


async def test_reader_bounds_distinct_refs_from_newest_and_retains_complete_lists(
    identity: Identity,
) -> None:
    collection = ConversationCollection()
    store = manager(collection, write_provenance=True, max_refs=2)
    a, b, c = (EvidenceRef.from_chunk(make_chunk(name)) for name in ("a", "b", "c"))
    for name, refs in [("old", (c,)), ("shared", (a, b)), ("new", (a, b))]:
        await store.append_messages(identity, "id", pair(name, refs))
    raw = collection.docs[("tenant-a", "user-a", "id")]["messages"]
    raw[-1]["evidence_refs"].append(a.model_dump())

    rows = await store.get_messages(identity, "id", 100)

    assert [row.evidence_refs for row in rows[1::2]] == [None, (a, b), (a, b)]


async def test_malformed_or_oversized_provenance_does_not_poison_a_newer_pair(
    identity: Identity,
) -> None:
    collection = ConversationCollection()
    store = manager(collection, write_provenance=True, max_refs=1)
    a, b = (EvidenceRef.from_chunk(make_chunk(name)) for name in ("a", "b"))
    for name, refs in [("too-many", (a, b)), ("bad", (a,)), ("valid", (a,))]:
        await store.append_messages(identity, "id", pair(name, refs))
    raw = collection.docs[("tenant-a", "user-a", "id")]["messages"]
    assert "evidence_refs" not in raw[1]
    raw[1]["evidence_refs"] = [a.model_dump(), b.model_dump()]
    raw[2]["evidence_refs"] = {"bad": "user metadata"}
    raw[3]["evidence_refs"] = [a.model_dump(), {"wrong": "shape"}]

    rows = await store.get_messages(identity, "id", 100)

    assert [row.evidence_refs for row in rows[1::2]] == [None, None, (a,)]
    assert rows[2].evidence_refs is not None  # The invalid user row also blocks its pair.


async def test_oversize_utf8_provenance_is_omitted_before_sending_any_write(
    identity: Identity,
) -> None:
    collection = ConversationCollection()
    store = manager(collection, write_provenance=True)
    ref = EvidenceRef.from_chunk(make_chunk("large")).model_copy(
        update={"document_id": "😀" * (MAX_BSON_SIZE // 4)}
    )

    await store.append_messages(identity, "id", pair("large", (ref,)))

    assert collection.writes == 1
    assert (await store.get_messages(identity, "id", 2))[-1].evidence_refs is None


@pytest.mark.parametrize(
    "refusal",
    [
        DocumentTooLarge("command too large"),
        WriteError("oversized update", 17419),
        WriteError("oversized upsert", 17420),
        WriteError("BSONObjectTooLarge", 10334),
    ],
)
async def test_definite_size_refusal_retries_once_without_provenance(
    identity: Identity, refusal: Exception
) -> None:
    collection = ConversationCollection()
    collection.failures.append(refusal)
    store = manager(collection, write_provenance=True)
    await store.append_messages(
        identity, "id", pair("only once", (EvidenceRef.from_chunk(make_chunk("a")),))
    )

    rows = await store.get_messages(identity, "id", 100)
    assert [row.content for row in rows] == ["question only once", "answer only once"]
    assert rows[-1].evidence_refs is None
    assert collection.writes == 2


@pytest.mark.parametrize(
    "failure",
    [
        AutoReconnect("connection lost after sending"),
        WriteConcernError("write might have happened", 10334),
        OperationFailure("command failed", 13),
        WriteError("document validation failed", 121),
    ],
)
async def test_unrelated_and_ambiguous_errors_never_retry(
    identity: Identity, failure: Exception
) -> None:
    collection = ConversationCollection()
    collection.failures.append(failure)
    store = manager(collection, write_provenance=True)
    with pytest.raises(InfrastructureError):
        await store.append_messages(identity, "id", pair("one", ()))
    assert collection.writes == 1


async def test_size_failure_of_fallback_is_propagated(identity: Identity) -> None:
    collection = ConversationCollection(max_size=10)
    store = manager(collection, write_provenance=True)
    with pytest.raises(InfrastructureError):
        await store.append_messages(identity, "id", pair("one", ()))
    assert collection.writes == 2
    assert await store.get_messages(identity, "id", 2) == ()


async def test_concurrent_appends_near_document_limit_preserve_pairs_and_ownership(
    identity: Identity,
) -> None:
    collection = ConversationCollection()
    store = manager(collection, write_provenance=True)
    ref = EvidenceRef.from_chunk(make_chunk("a"))
    for number in range(50):
        await store.append_messages(identity, "id", pair(str(number), None))
    # Leave space for one enriched pair in the retained 100-message document.
    key = ("tenant-a", "user-a", "id")
    doc = collection.docs[key]
    collection.max_size = len(BSON.encode(doc)) + len(BSON.encode(ref.model_dump())) + 45
    other_user = identity.model_copy(update={"user_id": "other"})
    other_tenant = identity.model_copy(update={"tenant_id": "other"})
    await asyncio.gather(
        store.append_messages(identity, "id", pair("A", (ref,))),
        store.append_messages(identity, "id", pair("B", (ref,))),
        store.append_messages(other_user, "id", pair("C", (ref,))),
        store.append_messages(other_tenant, "id", pair("D", (ref,))),
    )

    rows = await store.get_messages(identity, "id", 100)
    assert len(rows) == 100
    assert rows[0].content == "question 2"
    assert {tuple(row.content for row in rows[i : i + 2]) for i in (96, 98)} == {
        ("question A", "answer A"),
        ("question B", "answer B"),
    }
    assert {row.evidence_refs for row in rows[-3::2]} == {(ref,), None}
    assert [row.content for row in await store.get_messages(other_user, "id", 2)] == [
        "question C",
        "answer C",
    ]
    assert [row.content for row in await store.get_messages(other_tenant, "id", 2)] == [
        "question D",
        "answer D",
    ]
