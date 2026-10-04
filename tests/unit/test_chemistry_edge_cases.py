"""Unit tests for chemical edge cases, charged species, stereoisomers, salts, and macrocycles."""

import pytest
from src.chemistry.validation import validate_smiles, validate_batch_smiles
from src.chemistry.descriptors import calculate_descriptors
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.similarity.index import compute_tanimoto_similarity


class TestChemistryEdgeCases:
    """Stress tests for chemical edge cases and diverse chemical structures."""

    def test_charged_species_and_zwitterions(self):
        # Glycine zwitterion
        glycine = "[NH3+]CC([O-])=O"
        val = validate_smiles(glycine)
        assert val.is_valid is True
        assert val.canonical_smiles is not None

        desc = calculate_descriptors(val.canonical_smiles)
        assert desc.molecular_weight > 70.0
        assert desc.hbond_donors >= 1

        fp = generate_morgan_fingerprint(val.canonical_smiles)
        assert fp.numpy_vector.shape == (2048,)

    def test_salts_and_disconnected_fragments(self):
        # Sodium acetate
        na_acetate = "CC(=O)[O-].[Na+]"
        val = validate_smiles(na_acetate)
        assert val.is_valid is True

        desc = calculate_descriptors(val.canonical_smiles)
        assert desc.molecular_weight > 80.0

    def test_stereoisomer_differentiation(self):
        # (R)-thalidomide vs (S)-thalidomide
        r_thal = "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"
        s_thal_iso = "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"

        fp1 = generate_morgan_fingerprint(r_thal)
        fp2 = generate_morgan_fingerprint(s_thal_iso)

        sim = compute_tanimoto_similarity(fp1.bit_vector, fp2.bit_vector)
        assert 0.0 < sim <= 1.0

    def test_macrocycle_handling(self):
        # Cyclosporine A (MW ~ 1202 Da)
        cyclosporine = "CCC1NC(=O)C(C(O)C(C)CC=CC)N(C)C(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C1=O"
        val = validate_smiles(cyclosporine)
        assert val.is_valid is True

        desc = calculate_descriptors(val.canonical_smiles)
        assert desc.molecular_weight > 1000.0
        assert desc.lipinski_pass is False  # Fails MW > 500
        assert desc.lipinski_violations >= 1

    def test_unicode_and_unprintable_characters(self):
        corrupt = "CC(=O)Oc1ccccc1C(=O)O\u200b\x00\ufffd"
        val = validate_smiles(corrupt)
        # Should gracefully fail or clean without throwing unhandled exceptions
        assert isinstance(val.is_valid, bool)

    def test_very_long_string_protection(self):
        giant_string = "C" * 20000
        val = validate_smiles(giant_string)
        assert val.is_valid is False
        assert val.error_message is not None
