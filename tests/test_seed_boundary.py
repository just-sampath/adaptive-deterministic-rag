from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient
from scripts.seed_dummy_data import _chunks

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
_QDRANT_WRITE_METHODS = frozenset(
    {
        "upsert",
        "upload_points",
        "upload_collection",
        "update_vectors",
        "delete_vectors",
        "delete",
        "delete_payload",
        "clear_payload",
        "set_payload",
        "overwrite_payload",
        "batch_update_points",
    }
)
# Client methods that take point selectors, payloads or vectors; any such method is a writer.
_POINT_ARGUMENTS = frozenset({"points", "points_selector", "vectors"})


def _runtime_trees() -> dict[Path, ast.Module]:
    return {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in _SOURCE_ROOT.rglob("*.py")
    }


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _qdrant_write_calls(tree: ast.Module) -> list[str]:
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _QDRANT_WRITE_METHODS
    ]


def test_boundary_scans_the_whole_runtime_package() -> None:
    assert len(_runtime_trees()) >= 30


def test_runtime_source_never_imports_the_seed_script() -> None:
    offenders = {
        path.name
        for path, tree in _runtime_trees().items()
        if any(
            module == "scripts" or module.startswith("scripts.")
            for module in _imported_modules(tree)
        )
    }

    assert offenders == set()


def test_runtime_source_calls_no_qdrant_point_writer() -> None:
    writers = {
        path.name: calls
        for path, tree in _runtime_trees().items()
        if (calls := _qdrant_write_calls(tree))
    }

    assert writers == {}


@pytest.mark.parametrize("method", sorted(_QDRANT_WRITE_METHODS))
def test_guard_flags_each_denied_point_writer_in_source(method: str) -> None:
    tree = ast.parse(f"await self._client.{method}(collection_name='chunks')")

    assert _qdrant_write_calls(tree) == [method]


def test_guard_names_exist_on_the_installed_qdrant_client() -> None:
    missing = {name for name in _QDRANT_WRITE_METHODS if not hasattr(AsyncQdrantClient, name)}

    assert missing == set()


def test_guard_covers_every_point_taking_method_of_the_installed_client() -> None:
    point_takers = {
        name
        for name, function in inspect.getmembers(AsyncQdrantClient, inspect.isfunction)
        if not name.startswith("_")
        and _POINT_ARGUMENTS & set(inspect.signature(function).parameters)
    }

    assert point_takers <= _QDRANT_WRITE_METHODS


def test_dummy_chunks_are_fixed_and_idempotent() -> None:
    first = _chunks("tenant-a", "document", "00000000-0000-0000-0000-000000000010")
    second = _chunks("tenant-a", "document", "00000000-0000-0000-0000-000000000010")

    assert first == second
    assert len(first) == 2
    assert first[0].allowed_roles == frozenset({"reader", "administrator"})


def test_dummy_chunks_are_linked_by_sequence() -> None:
    first, second = _chunks("tenant-a", "document", "00000000-0000-0000-0000-000000000010")

    assert first.previous_chunk_id is None
    assert first.next_chunk_id == second.id
    assert second.previous_chunk_id == first.id
    assert second.next_chunk_id is None
