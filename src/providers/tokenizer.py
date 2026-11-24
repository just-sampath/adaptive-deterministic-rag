"""Offline ordinary-text counting with OpenAI's o200k_base encoding."""

import base64
import hashlib
from collections.abc import Callable
from pathlib import Path

from tiktoken import Encoding

from utils.errors import ConfigurationError

O200K_ASSET_PATH = Path(__file__).resolve().parents[2] / ".tokenizer" / "o200k_base.tiktoken"
_PREFETCH = "Run uv run --frozen python -m scripts.prefetch_tokenizer."

# Hash and pre-tokenization pattern from tiktoken 0.14.0, tiktoken_ext/openai_public.py,
# o200k_base(). Its constructor can download ranks, so build the encoding from verified bytes.
_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
# The following pattern is from tiktoken (MIT license, Copyright (c) 2022 OpenAI, Shantanu Jain).
# The license and permission notice ship with the pinned tiktoken distribution.
_PATTERN = "|".join(
    (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    )
)


def vocabulary_is_valid(data: bytes) -> bool:
    """Check the original vocabulary's SHA-256 before installing or parsing it."""
    return hashlib.sha256(data).hexdigest() == _SHA256


def load_o200k_counter() -> Callable[[str], int]:
    """Build a shareable counter from the prefetched vocabulary without consulting an SDK cache."""
    try:
        data = O200K_ASSET_PATH.read_bytes()
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read the local o200k_base vocabulary. {_PREFETCH}"
        ) from exc
    if not vocabulary_is_valid(data):
        raise ConfigurationError(
            f"Local o200k_base vocabulary failed SHA-256 verification. {_PREFETCH}"
        )

    ranks = {}
    for line in data.splitlines():
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    encoding = Encoding(
        "o200k_base_ordinary", pat_str=_PATTERN, mergeable_ranks=ranks, special_tokens={}
    )
    return lambda text: len(encoding.encode_ordinary(text))
