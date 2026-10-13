from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.domain import SparseEmbedding
from providers.sparse import Bm25SparseEncoder
from utils.settings import SparseConfig


def test_compound_identifiers_keep_the_whole_token_and_its_parts() -> None:
    encoder = Bm25SparseEncoder(SparseConfig())

    compound = encoder.encode_query("Error-Code")

    assert len(compound.indices) == 3
    assert set(encoder.encode_query("error code").indices) <= set(compound.indices)


def test_casefolding_makes_identifiers_match_across_case() -> None:
    encoder = Bm25SparseEncoder(SparseConfig())

    assert encoder.encode_query("ERR_42") == encoder.encode_query("err_42")


def test_short_parts_are_dropped_but_numeric_parts_are_kept() -> None:
    encoder = Bm25SparseEncoder(SparseConfig())

    assert len(encoder.encode_query("x.y").indices) == 1
    assert len(encoder.encode_query("1.2").indices) == 3


def test_stopwords_are_removed_from_whole_tokens_and_parts() -> None:
    encoder = Bm25SparseEncoder(SparseConfig())

    assert encoder.encode_query("the and of") == SparseEmbedding()
    stopword_part = encoder.encode_query("data.the")
    assert len(stopword_part.indices) == 2
    assert set(encoder.encode_query("data").indices) <= set(stopword_part.indices)


def test_query_weights_are_one_per_distinct_term() -> None:
    encoded = Bm25SparseEncoder(SparseConfig()).encode_query("alpha alpha beta")

    assert len(encoded.indices) == 2
    assert encoded.values == (1.0, 1.0)


def test_document_weights_follow_bm25_with_length_normalization() -> None:
    encoder = Bm25SparseEncoder(SparseConfig(average_document_tokens=3))

    (encoded,) = encoder.encode_documents(["alpha beta alpha"])

    # Document length equals the average, so alpha (tf=2) weighs 4.4 / 3.2 and beta (tf=1) 1.0.
    assert sorted(encoded.values) == pytest.approx([1.0, 1.375])


def test_term_frequency_saturates_below_the_bm25_ceiling() -> None:
    encoder = Bm25SparseEncoder(SparseConfig())
    texts = ["alpha", "alpha alpha", " ".join(["alpha"] * 20)]

    weights = [encoded.values[0] for encoded in encoder.encode_documents(texts)]

    ceiling = SparseConfig().k1 + 1
    assert weights == sorted(weights)
    assert len(set(weights)) == len(weights)
    assert all(weight < ceiling for weight in weights)


def test_encoding_is_deterministic_and_indices_are_sorted() -> None:
    encoder = Bm25SparseEncoder(SparseConfig())
    text = "Retrieval fuses dense and sparse ranks; sparse ranks use BM25 weights."

    first = encoder.encode_documents([text])
    second = encoder.encode_documents([text])

    assert first == second
    assert list(first[0].indices) == sorted(first[0].indices)


def test_empty_text_encodes_to_an_empty_vector() -> None:
    encoder = Bm25SparseEncoder(SparseConfig())

    assert encoder.encode_query("") == SparseEmbedding()
    assert encoder.encode_documents([""]) == [SparseEmbedding()]


def test_indices_stay_inside_the_configured_hash_space() -> None:
    encoder = Bm25SparseEncoder(SparseConfig(hash_space=1_000_003))
    text = " ".join(f"term{number}" for number in range(500))

    (encoded,) = encoder.encode_documents([text])

    assert max(encoded.indices) < 1_000_003


def test_fingerprint_is_identical_for_default_and_yaml_style_numbers() -> None:
    default = Bm25SparseEncoder(SparseConfig())
    explicit = Bm25SparseEncoder(SparseConfig(average_document_tokens=256.0))

    assert default.fingerprint == explicit.fingerprint
    assert default.fingerprint == (
        "bm25:v1:k1=1.2:b=0.75:avgdl=256.0:space=2147483647:lang=english"
    )
    assert default.requires_idf_modifier is True


@pytest.mark.parametrize(
    ("indices", "values"),
    [
        ((1, 2), (0.5,)),
        ((-1,), (1.0,)),
        ((4_294_967_296,), (1.0,)),
        ((3, 2), (1.0, 1.0)),
        ((2, 2), (1.0, 1.0)),
    ],
    ids=["length-mismatch", "below-uint32", "above-uint32", "unsorted", "duplicate"],
)
def test_sparse_embedding_rejects_malformed_vectors(
    indices: tuple[int, ...],
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError):
        SparseEmbedding(indices=indices, values=values)


def test_sparse_embedding_accepts_sorted_uint32_bounds() -> None:
    vector = SparseEmbedding(indices=(0, 7, 4_294_967_295), values=(1.0, 0.5, 2.0))

    assert vector.indices == (0, 7, 4_294_967_295)
    assert vector.values == (1.0, 0.5, 2.0)
