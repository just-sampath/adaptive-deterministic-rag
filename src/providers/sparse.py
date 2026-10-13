"""Deterministic BM25 sparse encoding over hashed lexical terms."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from models.domain import SparseEmbedding
from utils.settings import SparseConfig

_TOKEN = re.compile(r"\w+(?:[.:/-]\w+)*")
_SEPARATOR = re.compile(r"[._:/-]")
_STOPWORDS = frozenset(
    word
    for line in (
        "a all also although am an and any are as at be because been before being below between",
        "both but by can could did do does doing down during each few for from further had has",
        "have having he her here hers him his how i if in into is it its just me my of off on once",
        "or our out over she should so some such than that the their them then there these they",
        "this those through to too under until up very was we were what when where which while who",
        "why will with would you your",
    )
    for word in line.split()
)


class Bm25SparseEncoder:
    """Encode text as BM25 term weights; Qdrant applies IDF at query time."""

    def __init__(self, config: SparseConfig) -> None:
        self._k1 = float(config.k1)
        self._b = float(config.b)
        self._average_tokens = float(config.average_document_tokens)
        self._hash_space = config.hash_space
        self._fingerprint = (
            f"bm25:v1:k1={self._k1}:b={self._b}:avgdl={self._average_tokens}"
            f":space={self._hash_space}:lang={config.language}"
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def requires_idf_modifier(self) -> bool:
        return True

    def encode_query(self, text: str) -> SparseEmbedding:
        return _sparse(dict.fromkeys({self._index(term) for term in _terms(text)}, 1.0))

    def encode_documents(self, texts: Sequence[str]) -> list[SparseEmbedding]:
        return [self._encode_document(text) for text in texts]

    def _encode_document(self, text: str) -> SparseEmbedding:
        terms = _terms(text)
        frequencies = Counter(self._index(term) for term in terms)
        length_norm = self._k1 * (1 - self._b + self._b * len(terms) / self._average_tokens)
        return _sparse(
            {
                index: frequency * (self._k1 + 1) / (frequency + length_norm)
                for index, frequency in frequencies.items()
            }
        )

    def _index(self, term: str) -> int:
        digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._hash_space


def _terms(text: str) -> list[str]:
    """Return whole tokens and the retained parts of compound identifiers, casefolded."""

    terms: list[str] = []
    for match in _TOKEN.finditer(text.casefold()):
        token = match.group()
        if token not in _STOPWORDS:
            terms.append(token)
        if _SEPARATOR.search(token):
            terms.extend(part for part in _SEPARATOR.split(token) if _keep_part(part))
    return terms


def _keep_part(part: str) -> bool:
    return part not in _STOPWORDS and (len(part) >= 2 or part.isdigit())


def _sparse(weights: Mapping[int, float]) -> SparseEmbedding:
    ordered = sorted(weights.items())
    return SparseEmbedding(
        indices=tuple(index for index, _ in ordered),
        values=tuple(value for _, value in ordered),
    )
