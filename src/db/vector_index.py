"""Physical Qdrant collection identity derived from the encoder and vector layout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from utils.settings import VectorStoreConfig


@dataclass(frozen=True, slots=True)
class VectorIndexSpec:
    alias: str
    physical_collection: str
    dense_vector_name: str
    sparse_vector_name: str
    dense_dimension: int
    sparse_idf_modifier: bool
    encoder_fingerprint: str


def build_vector_index_spec(
    config: VectorStoreConfig,
    *,
    encoder_fingerprint: str,
    dense_dimension: int,
    sparse_idf_modifier: bool,
) -> VectorIndexSpec:
    """Name the physical collection so any encoder or layout change needs a new collection."""

    layout = (
        f"{encoder_fingerprint}|cosine|idf={sparse_idf_modifier}"
        f"|dense={config.dense_vector_name}|sparse={config.sparse_vector_name}"
        f"|dim={dense_dimension}"
    )
    digest = hashlib.sha256(layout.encode()).hexdigest()[:12]
    return VectorIndexSpec(
        alias=config.collection_alias,
        physical_collection=f"{config.collection_alias}__{digest}",
        dense_vector_name=config.dense_vector_name,
        sparse_vector_name=config.sparse_vector_name,
        dense_dimension=dense_dimension,
        sparse_idf_modifier=sparse_idf_modifier,
        encoder_fingerprint=encoder_fingerprint,
    )
