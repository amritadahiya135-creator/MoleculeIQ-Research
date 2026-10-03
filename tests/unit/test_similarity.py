"""Unit and mathematical validation tests for the chemical similarity engine (Module M06)."""

import pytest
from src.similarity.index import (
    ChemicalSimilarityIndex,
    compute_tanimoto_similarity,
    SimilaritySearchResult,
    SimilarityMatch,
)
from src.similarity.chembl_cache import load_or_create_reference_cache


class TestChemicalSimilarity:
    """Tests for Module M06: Chemical Similarity Search."""

    def test_tanimoto_mathematical_properties(self):
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        paracetamol = "CC(=O)Nc1ccc(O)cc1"
        caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

        # Property 1: Identity T(A, A) = 1.0
        assert pytest.approx(compute_tanimoto_similarity(aspirin, aspirin), 0.001) == 1.0
        assert pytest.approx(compute_tanimoto_similarity(caffeine, caffeine), 0.001) == 1.0

        # Property 2: Symmetry T(A, B) == T(B, A)
        sim_ab = compute_tanimoto_similarity(aspirin, paracetamol)
        sim_ba = compute_tanimoto_similarity(paracetamol, aspirin)
        assert pytest.approx(sim_ab, 0.0001) == sim_ba

        # Property 3: Bounds 0.0 <= T(A, B) <= 1.0
        sim_ac = compute_tanimoto_similarity(aspirin, caffeine)
        assert 0.0 <= sim_ab <= 1.0
        assert 0.0 <= sim_ac <= 1.0

    def test_similarity_index_search_aspirin_retrieval(self):
        index = ChemicalSimilarityIndex()
        assert index.total_compounds > 0

        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        result = index.search(aspirin, top_k=5)

        assert isinstance(result, SimilaritySearchResult)
        assert result.status == "SUCCESS"
        assert len(result.matches) > 0
        assert result.best_similarity == 1.0  # Aspirin is in anchor reference set

        # Top hit must be Aspirin
        top_hit = result.matches[0]
        assert "Aspirin" in top_hit.name or top_hit.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"

        # Check descending order of similarities
        sims = [m.similarity for m in result.matches]
        assert sims == sorted(sims, reverse=True)

    def test_invalid_query_safe_handling(self):
        index = ChemicalSimilarityIndex()
        result = index.search("NOT_A_VALID_SMILES", top_k=5)
        assert result.status == "INVALID_INPUT"
        assert result.error_message is not None
        assert len(result.matches) == 0
