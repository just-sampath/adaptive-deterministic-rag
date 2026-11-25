"""Repeated, interrupted and concurrent local seeds converge on one logical corpus revision."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import AutoReconnect, DuplicateKeyError
from qdrant_client import AsyncQdrantClient, models
from scripts import seed_dummy_data as seed
from scripts.seed_dummy_data import (
    LegacyCounter,
    SeedConflictError,
    SeedFixture,
    SeedReconciliationRequiredError,
    build_fixture,
    seed_fixture,
)

from db.initializer import DatabaseInitializer
from db.vector_index import VectorIndexSpec, build_vector_index_spec
from managers.vector import to_qdrant_sparse
from models.domain import Identity
from providers.sparse import Bm25SparseEncoder
from tests.fakes import FakeEmbeddings
from utils.errors import ConfigurationError, ProviderResponseError
from utils.settings import (
    Environment,
    ExternalSettings,
    MongoConfig,
    Settings,
    SparseConfig,
    VectorStoreConfig,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

_TENANT = "local-tenant"
_UNIQUE_KEYS = {"system_state": ("tenant_id",)}
Hook = Callable[[str, str, dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class _UpdateResult:
    matched_count: int


def _get(document: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    node: Any = document
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _set(document: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = document
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = copy.deepcopy(value)


def _matches(document: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, condition in query.items():
        present, value = _get(document, key)
        if isinstance(condition, dict) and "$exists" in condition:
            if present is not condition["$exists"]:
                return False
        elif isinstance(condition, dict) and "$in" in condition:
            if not present or value not in condition["$in"]:
                return False
        elif not present or value != condition:
            return False
    return True


class FakeCollection:
    """The subset of an async collection the seed uses, with unique keys and fault hooks."""

    def __init__(self, store: FakeMongo, name: str) -> None:
        self._store = store
        self.name = name
        self.rows: list[dict[str, Any]] = []

    async def find_one(
        self, query: dict[str, Any], projection: dict[str, int] | None = None
    ) -> dict[str, Any] | None:
        await self._store.before("find_one", self.name, query)
        for row in self.rows:
            if _matches(row, query):
                return copy.deepcopy(row)
        return None

    def find(self, query: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async def rows() -> AsyncIterator[dict[str, Any]]:
            await self._store.before("find", self.name, query)
            for row in list(self.rows):
                if _matches(row, query):
                    yield copy.deepcopy(row)

        return rows()

    async def update_one(
        self, query: dict[str, Any], update: dict[str, Any], upsert: bool = False
    ) -> _UpdateResult:
        await self._store.before("update_one", self.name, query)
        for row in self.rows:
            if _matches(row, query):
                self._apply(row, update, inserting=False)
                await self._store.after("update_one", self.name, query)
                return _UpdateResult(matched_count=1)
        if upsert:
            row = {
                key: value
                for key, value in query.items()
                if not key.startswith("$") and not isinstance(value, dict)
            }
            self._apply(row, update, inserting=True)
            unique = _UNIQUE_KEYS.get(self.name, ("tenant_id", "id"))
            if any(all(old.get(key) == row.get(key) for key in unique) for old in self.rows):
                raise DuplicateKeyError("E11000 duplicate key")
            self.rows.append(row)
        await self._store.after("update_one", self.name, query)
        return _UpdateResult(matched_count=0)

    @staticmethod
    def _apply(row: dict[str, Any], update: dict[str, Any], *, inserting: bool) -> None:
        for key, value in update.get("$setOnInsert", {}).items() if inserting else ():
            _set(row, key, value)
        for key, value in update.get("$set", {}).items():
            _set(row, key, value)
        for key, value in update.get("$inc", {}).items():
            _, current = _get(row, key)
            _set(row, key, (current or 0) + value)


class FakeMongo:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.before_hooks: list[Hook] = []
        self.after_hooks: list[Hook] = []

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection(self, name))

    async def before(self, operation: str, collection: str, query: dict[str, Any]) -> None:
        for hook in list(self.before_hooks):
            await hook(operation, collection, query)

    async def after(self, operation: str, collection: str, query: dict[str, Any]) -> None:
        for hook in list(self.after_hooks):
            await hook(operation, collection, query)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {name: copy.deepcopy(item.rows) for name, item in self.collections.items()}


class FaultyQdrant:
    """Delegate to the in-memory client, optionally failing after a point write lands."""

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client
        self.fail_before_upsert = False
        self.fail_after_upsert = False
        self.upserts = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def upsert(self, **kwargs: Any) -> Any:
        self.upserts += 1
        if self.fail_before_upsert:
            self.fail_before_upsert = False
            raise AutoReconnect("qdrant went away")
        result = await self._client.upsert(**kwargs)
        if self.fail_after_upsert:
            self.fail_after_upsert = False
            raise AutoReconnect("acknowledgment lost")
        return result


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.document_texts: list[str] = []

    async def embed_documents(self, texts: Any) -> Any:
        self.document_texts.extend(texts)
        return await super().embed_documents(texts)


@dataclass(slots=True)
class Stores:
    mongo: FakeMongo
    qdrant: FaultyQdrant
    index: VectorIndexSpec
    embeddings: CountingEmbeddings
    clock: list[datetime]

    async def seed(
        self,
        *,
        fixture: SeedFixture | None = None,
        legacy: LegacyCounter | None = None,
    ) -> dict[str, str | int]:
        self.clock[0] += timedelta(minutes=1)
        return await seed_fixture(
            cast(AsyncDatabase[dict[str, Any]], self.mongo),
            cast(AsyncQdrantClient, self.qdrant),
            index=self.index,
            embeddings=self.embeddings,
            sparse=Bm25SparseEncoder(SparseConfig()),
            fixture=fixture or build_fixture(_TENANT),
            legacy_counter=legacy,
            clock=lambda: self.clock[0],
        )

    async def point_ids(self) -> set[str]:
        points, _ = await self.qdrant.scroll(
            collection_name=self.index.physical_collection, limit=100
        )
        return {str(point.id) for point in points}

    def state(self, tenant: str = _TENANT) -> dict[str, Any] | None:
        rows = [row for row in self.mongo["system_state"].rows if row["tenant_id"] == tenant]
        return rows[0] if rows else None


@pytest.fixture
async def stores() -> AsyncIterator[Stores]:
    client = AsyncQdrantClient(location=":memory:")
    index = build_vector_index_spec(
        VectorStoreConfig(collection_alias="seed_chunks"),
        encoder_fingerprint="fake-encoder",
        dense_dimension=3,
        sparse_idf_modifier=True,
    )
    mongo_client = cast(AsyncMongoClient[dict[str, Any]], object())
    await DatabaseInitializer(
        mongo_client, client, mongo=MongoConfig(), index=index
    )._ensure_qdrant()
    yield Stores(
        mongo=FakeMongo(),
        qdrant=FaultyQdrant(client),
        index=index,
        embeddings=CountingEmbeddings(),
        clock=[datetime(2026, 11, 25, tzinfo=UTC)],
    )
    await client.close()


def _fixture_ids(fixture: SeedFixture) -> set[str]:
    return {chunk.id for chunk in fixture.chunks}


async def test_first_seed_creates_one_revision_and_a_repeat_changes_nothing(stores: Stores) -> None:
    fixture = build_fixture(_TENANT)

    first = await stores.seed()
    after_first = stores.mongo.snapshot()
    second = await stores.seed()

    assert (
        first
        == second
        == {
            "document_id": fixture.document_id,
            "document_version_id": fixture.version_id,
            "chunks_indexed": 2,
        }
    )
    assert stores.mongo.snapshot() == after_first
    assert len(after_first["documents"]) == 1
    assert len(after_first["document_versions"]) == 1
    assert len(after_first["chunks"]) == 2
    assert await stores.point_ids() == _fixture_ids(fixture)
    state = stores.state()
    assert state is not None
    assert state["corpus_version"] == 1
    assert state["local_seed"]["status"] == "completed"
    assert state["local_seed"]["fingerprint"] == fixture.fingerprint
    # The repeat found every point and embedded nothing.
    assert len(stores.embeddings.document_texts) == 2


async def test_a_repeat_restores_missing_points_without_a_new_revision(stores: Stores) -> None:
    fixture = build_fixture(_TENANT)
    await stores.seed()
    rows = stores.mongo.snapshot()
    await stores.qdrant.delete(
        collection_name=stores.index.physical_collection,
        points_selector=[fixture.chunks[1].id],
    )

    await stores.seed()

    assert await stores.point_ids() == _fixture_ids(fixture)
    assert stores.embeddings.document_texts[2:] == [fixture.chunks[1].text]
    assert stores.mongo.snapshot()["chunks"] == rows["chunks"]
    state = stores.state()
    assert state is not None and state["corpus_version"] == 1


async def test_a_repeat_restores_missing_rows_without_a_new_revision(stores: Stores) -> None:
    fixture = build_fixture(_TENANT)
    await stores.seed()
    kept_chunk = copy.deepcopy(stores.mongo["chunks"].rows[0])
    stores.mongo["document_versions"].rows.clear()
    del stores.mongo["chunks"].rows[1]

    await stores.seed()

    versions = stores.mongo["document_versions"].rows
    assert [row["id"] for row in versions] == [fixture.version_id]
    assert {row["id"] for row in stores.mongo["chunks"].rows} == _fixture_ids(fixture)
    assert stores.mongo["chunks"].rows[0] == kept_chunk
    # The points were all present, so nothing was embedded again.
    assert len(stores.embeddings.document_texts) == 2
    state = stores.state()
    assert state is not None and state["corpus_version"] == 1


@pytest.mark.parametrize("failure", ["after_mongo_rows", "after_points", "before_marker"])
async def test_an_interrupted_seed_resumes_and_counts_once(stores: Stores, failure: str) -> None:
    fixture = build_fixture(_TENANT)

    async def fail_completion(operation: str, collection: str, query: dict[str, Any]) -> None:
        # Only the completion update filters on the marker's status.
        if collection == "system_state" and "local_seed.status" in query:
            raise AutoReconnect("store went away")

    if failure == "after_mongo_rows":
        # Every row is written before the points, so a failed point write leaves only rows.
        stores.qdrant.fail_before_upsert = True
    elif failure == "after_points":
        stores.qdrant.fail_after_upsert = True
    else:
        stores.mongo.before_hooks.append(fail_completion)

    with pytest.raises(AutoReconnect):
        await stores.seed()
    first_rows = stores.mongo.snapshot()
    first_points = await stores.point_ids()
    stores.mongo.before_hooks.clear()
    interrupted = stores.state()
    assert interrupted is not None
    assert interrupted["local_seed"]["status"] == "reserved"
    assert "corpus_version" not in interrupted
    assert len(first_rows["chunks"]) == 2
    assert first_points == (set() if failure == "after_mongo_rows" else _fixture_ids(fixture))

    await stores.seed()

    state = stores.state()
    assert state is not None
    assert state["corpus_version"] == 1
    assert state["local_seed"]["status"] == "completed"
    assert await stores.point_ids() == _fixture_ids(fixture)
    # Rows written by the interrupted run keep their creation times.
    for name in ("documents", "document_versions", "chunks"):
        assert stores.mongo.snapshot()[name] == first_rows[name]


async def test_a_lost_acknowledgment_after_completion_does_not_count_again(stores: Stores) -> None:
    async def lose_ack(operation: str, collection: str, query: dict[str, Any]) -> None:
        state = stores.state()
        if collection == "system_state" and state and state["local_seed"]["status"] == "completed":
            stores.mongo.after_hooks.clear()
            raise AutoReconnect("acknowledgment lost")

    stores.mongo.after_hooks.append(lose_ack)
    with pytest.raises(AutoReconnect):
        await stores.seed()

    await stores.seed()

    state = stores.state()
    assert state is not None and state["corpus_version"] == 1


async def test_concurrent_identical_seeds_count_once(stores: Stores) -> None:
    """Both runs read the empty state together, then both reach completion together."""

    arrivals: dict[str, int] = {"read": 0, "complete": 0}
    gates = {"read": asyncio.Event(), "complete": asyncio.Event()}

    async def meet(stage: str) -> None:
        arrivals[stage] += 1
        if arrivals[stage] == 2:
            gates[stage].set()
        await gates[stage].wait()

    async def barrier(operation: str, collection: str, query: dict[str, Any]) -> None:
        if collection != "system_state":
            return
        if operation == "find_one" and not gates["read"].is_set():
            await meet("read")
        elif operation == "update_one" and "local_seed.status" in query:
            await meet("complete")

    stores.mongo.before_hooks.append(barrier)
    first, second = await asyncio.wait_for(asyncio.gather(stores.seed(), stores.seed()), 5)

    assert first == second
    assert arrivals["complete"] == 2
    state = stores.state()
    assert state is not None and state["corpus_version"] == 1
    assert len(stores.mongo["documents"].rows) == 1


@pytest.mark.parametrize("winner", ["identical", "different"])
async def test_a_lost_first_insert_race_compares_the_row_that_won(
    stores: Stores, winner: str
) -> None:
    """MongoDB can report E11000 when a concurrent upsert inserts the same key first."""

    fixture = build_fixture(_TENANT)
    earlier = stores.clock[0]
    title = fixture.document["title"] if winner == "identical" else "Someone else's handbook"
    won = {**fixture.document, "title": title, "created_at": earlier}

    async def insert_first(operation: str, collection: str, query: dict[str, Any]) -> None:
        if collection == "documents" and operation == "update_one":
            stores.mongo.before_hooks.clear()
            stores.mongo["documents"].rows.append(copy.deepcopy(won))
            raise DuplicateKeyError("E11000 duplicate key error")

    stores.mongo.before_hooks.append(insert_first)

    if winner == "identical":
        await stores.seed()
        state = stores.state()
        assert state is not None and state["corpus_version"] == 1
        assert stores.mongo["documents"].rows == [won]
    else:
        with pytest.raises(SeedConflictError, match="document"):
            await stores.seed()
        state = stores.state()
        assert state is not None and "corpus_version" not in state
        assert state["local_seed"]["status"] == "reserved"


async def test_a_tenant_with_other_data_counts_the_fresh_handbook_once(stores: Stores) -> None:
    fixture = build_fixture(_TENANT)
    other_document = {"tenant_id": "other-tenant", "id": fixture.document_id, "title": "Theirs"}
    stores.mongo["documents"].rows.append(copy.deepcopy(other_document))
    stores.mongo["system_state"].rows.append(
        {"tenant_id": _TENANT, "corpus_version": 7, "unrelated": "kept"}
    )
    stores.mongo["system_state"].rows.append({"tenant_id": "other-tenant", "corpus_version": 3})

    await stores.seed()
    await stores.seed()

    state = stores.state()
    assert state is not None
    assert state["corpus_version"] == 8
    assert state["unrelated"] == "kept"
    assert stores.state("other-tenant") == {"tenant_id": "other-tenant", "corpus_version": 3}
    # Another tenant's row under the same ID is neither read as a conflict nor touched.
    assert [row for row in stores.mongo["documents"].rows if row["tenant_id"] != _TENANT] == [
        other_document
    ]


_LEGACY_TIME = datetime(2026, 1, 5, tzinfo=UTC)


async def _old_script_rows(mongo: FakeMongo, *, prior_counter: int | None, count: bool) -> None:
    """Write the Mongo rows the previous script wrote, the way it wrote them.

    That script upserted the document with ``$set``, replaced the version and chunk rows, and
    finished with an unmarked ``$inc``. ``count`` false stops it before that increment.
    """

    fixture = build_fixture(_TENANT)
    now = _LEGACY_TIME
    if prior_counter is not None:
        mongo["system_state"].rows.append(
            {"tenant_id": _TENANT, "corpus_version": prior_counter, "updated_at": now}
        )
    await mongo["documents"].update_one(
        {"tenant_id": _TENANT, "id": fixture.document_id},
        {
            "$set": {
                "title": "Adaptive RAG Local Handbook",
                "source_type": "dummy",
                "source_uri": None,
                "metadata": {"seed": "local"},
                "current_version_id": fixture.version_id,
                "last_version_number": 1,
                "deleted_at": None,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    mongo["document_versions"].rows.append(
        {
            "id": fixture.version_id,
            "document_id": fixture.document_id,
            "tenant_id": _TENANT,
            "version": 1,
            "checksum": "local-dummy-v1",
            "active": True,
            "created_at": now,
        }
    )
    for chunk in fixture.chunks:
        item: dict[str, Any] = chunk.model_dump(mode="python")
        item["allowed_roles"] = sorted(chunk.allowed_roles)
        item["heading_path"] = list(chunk.heading_path)
        item["created_at"] = now
        mongo["chunks"].rows.append(item)
    if count:
        await mongo["system_state"].update_one(
            {"tenant_id": _TENANT},
            {
                "$setOnInsert": {"tenant_id": _TENANT},
                "$inc": {"corpus_version": 1},
                "$set": {"updated_at": now},
            },
            upsert=True,
        )


async def _old_script_points(stores: Stores) -> None:
    fixture = build_fixture(_TENANT)
    texts = [chunk.text for chunk in fixture.chunks]
    dense = await FakeEmbeddings().embed_documents(texts)
    sparse = Bm25SparseEncoder(SparseConfig()).encode_documents(texts)
    await stores.qdrant.upsert(
        collection_name=stores.index.physical_collection,
        points=[
            models.PointStruct(
                id=chunk.id,
                vector={
                    stores.index.dense_vector_name: result.vector,
                    stores.index.sparse_vector_name: to_qdrant_sparse(vector),
                },
                payload={
                    "tenant_id": chunk.tenant_id,
                    "allowed_roles": sorted(chunk.allowed_roles),
                    "document_id": chunk.document_id,
                    "document_version_id": chunk.document_version_id,
                    "section_id": chunk.section_id,
                    "metadata": chunk.metadata,
                    "active": True,
                },
            )
            for chunk, result, vector in zip(fixture.chunks, dense, sparse, strict=True)
        ],
        wait=True,
    )


async def _old_script_seed(stores: Stores, *, prior_counter: int | None, count: bool) -> None:
    await _old_script_rows(stores.mongo, prior_counter=prior_counter, count=count)
    await _old_script_points(stores)


async def test_two_legacy_histories_leave_the_same_stores() -> None:
    # The old script counted its only seed, or a rerun stopped before counting after an earlier
    # seed had set the counter. Nothing in the stores tells the two apart.
    counted, interrupted = FakeMongo(), FakeMongo()
    await _old_script_rows(counted, prior_counter=None, count=True)
    await _old_script_rows(interrupted, prior_counter=1, count=False)

    assert counted.snapshot() == interrupted.snapshot()


@pytest.mark.parametrize("history", ["counted", "interrupted_after_earlier_seed"])
async def test_legacy_records_with_an_unmarked_counter_need_an_owner_decision(
    stores: Stores, history: str
) -> None:
    if history == "counted":
        await _old_script_seed(stores, prior_counter=None, count=True)
    else:
        await _old_script_seed(stores, prior_counter=1, count=False)
    before = stores.mongo.snapshot()
    points = await stores.point_ids()

    with pytest.raises(SeedReconciliationRequiredError, match="--legacy-counter"):
        await stores.seed()

    assert stores.mongo.snapshot() == before
    assert await stores.point_ids() == points


@pytest.mark.parametrize(
    ("decision", "expected"), [(LegacyCounter.ADOPT, 1), (LegacyCounter.COUNT, 2)]
)
async def test_an_owner_decision_is_recorded_with_completion(
    stores: Stores, decision: LegacyCounter, expected: int
) -> None:
    await _old_script_seed(stores, prior_counter=None, count=True)
    legacy_rows = stores.mongo.snapshot()

    await stores.seed(legacy=decision)
    await stores.seed()

    state = stores.state()
    assert state is not None
    assert state["corpus_version"] == expected
    assert state["local_seed"]["legacy_counter"] == decision.value
    assert state["local_seed"]["status"] == "completed"
    # The old script's rows are accepted as the handbook and left as they were.
    for name in ("documents", "document_versions", "chunks"):
        assert stores.mongo.snapshot()[name] == legacy_rows[name]
    assert stores.embeddings.document_texts == []


async def test_legacy_records_without_any_counter_resume_and_count_once(stores: Stores) -> None:
    await _old_script_seed(stores, prior_counter=None, count=False)

    await stores.seed()

    state = stores.state()
    assert state is not None and state["corpus_version"] == 1


async def test_conflicting_fixed_id_content_is_refused_before_any_write(stores: Stores) -> None:
    await stores.seed()
    chunk_row = stores.mongo["chunks"].rows[0]
    chunk_row["text"] = "Edited by hand"
    before = stores.mongo.snapshot()

    with pytest.raises(SeedConflictError, match="chunk"):
        await stores.seed()

    assert stores.mongo.snapshot() == before


async def test_a_different_fingerprint_is_refused_while_an_identical_writer_is_paused(
    stores: Stores,
) -> None:
    fixture = build_fixture(_TENANT)
    reserved = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_reservation(
        operation: str, collection: str, query: dict[str, Any]
    ) -> None:
        state = stores.state()
        if collection == "documents" and state and "local_seed" in state and not reserved.is_set():
            reserved.set()
            await release.wait()

    stores.mongo.before_hooks.append(pause_after_reservation)
    writer = asyncio.create_task(stores.seed())
    await reserved.wait()
    revised = replace(fixture, fingerprint="0" * 64)

    with pytest.raises(SeedConflictError, match="different handbook"):
        await stores.seed(fixture=revised)

    release.set()
    await writer
    state = stores.state()
    assert state is not None
    assert state["local_seed"]["fingerprint"] == fixture.fingerprint
    assert state["corpus_version"] == 1


def _settings(environment: Environment) -> Settings:
    return Settings(
        ai={"provider": "openai"},
        reranker={"provider": "fake"},
        external=ExternalSettings(
            _env_file=None,
            app_env=environment,
            openai_api_key="sk-test",
            auth_shared_token="a" * 40,
            debug_token="b" * 40,
            telemetry_hash_key="c" * 40,
        ),
    )


async def test_a_nonlocal_environment_is_refused_before_any_client_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_client(**_: Any) -> Any:
        raise AssertionError("no client may be built outside APP_ENV=local")

    monkeypatch.setattr(seed, "build_mongo_client", no_client)
    identity = Identity(tenant_id=_TENANT, user_id="admin", roles=frozenset({"administrator"}))

    with pytest.raises(ConfigurationError, match="APP_ENV=local"):
        await seed.seed_dummy_data(_settings(Environment.PRODUCTION), identity)


async def test_an_index_layout_mismatch_stops_the_seed_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Closeable:
        async def close(self) -> None:
            return None

    async def mismatch(self: DatabaseInitializer) -> None:
        raise ProviderResponseError("Qdrant alias points to another collection")

    async def never(*_: Any, **__: Any) -> Any:
        raise AssertionError("nothing may be written after a failed layout check")

    monkeypatch.setattr(seed, "build_mongo_client", lambda **_: Closeable())
    monkeypatch.setattr(seed, "build_qdrant_client", lambda **_: Closeable())
    monkeypatch.setattr(seed, "build_openai_client", lambda **_: Closeable())
    monkeypatch.setattr(DatabaseInitializer, "ensure_ready", mismatch)
    monkeypatch.setattr(seed, "seed_fixture", never)
    identity = Identity(tenant_id=_TENANT, user_id="admin", roles=frozenset({"administrator"}))

    with pytest.raises(ProviderResponseError, match="another collection"):
        await seed.seed_dummy_data(_settings(Environment.LOCAL), identity)


def test_the_fingerprint_ignores_time_and_follows_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = build_fixture(_TENANT), build_fixture(_TENANT)

    assert first.fingerprint == second.fingerprint
    assert build_fixture("another-tenant").fingerprint != first.fingerprint
    assert len(first.fingerprint) == 64
    heading, text = seed._TEXTS[1]
    monkeypatch.setattr(seed, "_TEXTS", (seed._TEXTS[0], (heading, text + " Edited.")))
    assert build_fixture(_TENANT).fingerprint != first.fingerprint
