"""Chemical similarity search and reference compound management."""

from src.similarity.chembl_cache import load_or_create_reference_cache, build_approved_reference_cache
from src.similarity.index import (
    ChemicalSimilarityIndex,
    SimilarityMatch,
    SimilaritySearchResult,
    compute_tanimoto_similarity,
)

__all__ = [
    "load_or_create_reference_cache",
    "build_approved_reference_cache",
    "ChemicalSimilarityIndex",
    "SimilarityMatch",
    "SimilaritySearchResult",
    "compute_tanimoto_similarity",
]
