from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import httpx
import pytest
from scripts import prefetch_tokenizer

from providers import tokenizer

_ROOT = Path(__file__).resolve().parents[1]
_PREFETCH = "uv run --frozen python -m scripts.prefetch_tokenizer"
_OFFLINE = r"""
import sys
from pathlib import Path

def offline(event, args):
    if event.startswith("socket."):
        raise AssertionError(f"Unexpected network access: {event}")

sys.addaudithook(offline)
sys.path.insert(0, sys.argv[1])
from providers import tokenizer
from utils.errors import ConfigurationError

if sys.argv[2] != "installed":
    tokenizer.O200K_ASSET_PATH = Path(sys.argv[2])
try:
    count = tokenizer.load_o200k_counter()
except ConfigurationError as error:
    print(str(error))
    sys.exit(1)

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import tiktoken

texts = ["", "hello world", "<|endoftext|>", "お誕生日おめでとう", "👩🏽‍💻", "1234567"]
with ThreadPoolExecutor(max_workers=3) as pool:
    print(list(pool.map(count, texts)))

# Exercise the pinned SDK's reference constructor on exactly the prefetched bytes.
with patch("tiktoken.load.read_file_cached", return_value=tokenizer.O200K_ASSET_PATH.read_bytes()):
    reference = tiktoken.get_encoding("o200k_base")
for sample in [*texts, "WE'RE ready.\r\nNEXT", "e\u0301\t\u3000", "!!///\n", "foo   bar  "]:
    assert count(sample) == len(reference.encode_ordinary(sample))
"""


def _offline_load(tmp_path: Path, asset: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"TIKTOKEN_CACHE_DIR": str(tmp_path / "empty-cache")}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", _OFFLINE, str(_ROOT / "src"), asset],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("contents", [None, b"truncated ranks"], ids=["missing", "corrupt"])
def test_unusable_vocabulary_fails_offline_with_setup_command(
    tmp_path: Path, contents: bytes | None
) -> None:
    asset = tmp_path / "o200k_base.tiktoken"
    if contents is not None:
        asset.write_bytes(contents)

    result = _offline_load(tmp_path, str(asset))

    assert result.returncode == 1, result.stderr
    assert _PREFETCH in result.stdout, result.stderr
    assert not (tmp_path / "empty-cache").exists()


def test_prefetched_vocabulary_counts_ordinary_text_offline(tmp_path: Path) -> None:
    # Counts independently obtained with tiktoken 0.14.0's official o200k_base.
    result = _offline_load(tmp_path, "installed")

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "[0, 2, 7, 8, 7, 3]"
    assert not (tmp_path / "empty-cache").exists()


def test_concurrent_prefetch_repairs_corrupt_vocabulary_then_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = tokenizer.O200K_ASSET_PATH.read_bytes()
    destination = tmp_path / "o200k_base.tiktoken"
    destination.write_bytes(b"truncated")
    monkeypatch.setattr(tokenizer, "O200K_ASSET_PATH", destination)
    both_fetching = Barrier(2)

    def download(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "openaipublic.blob.core.windows.net"
        both_fetching.wait(timeout=10)
        return httpx.Response(200, content=verified)

    with (
        httpx.Client(transport=httpx.MockTransport(download)) as client,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(prefetch_tokenizer.prefetch, client)
        second = pool.submit(prefetch_tokenizer.prefetch, client)
        first.result()
        second.result()

    prefetch_tokenizer.prefetch(client)  # Already closed: a second download would fail.
    assert destination.read_bytes() == verified
    assert list(tmp_path.iterdir()) == [destination]


def test_prefetch_rejects_a_corrupt_download_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "o200k_base.tiktoken"
    monkeypatch.setattr(tokenizer, "O200K_ASSET_PATH", destination)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"bad response"))

    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(ValueError, match="SHA-256"),
    ):
        prefetch_tokenizer.prefetch(client)

    assert not destination.exists()
