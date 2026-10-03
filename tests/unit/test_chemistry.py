"""Unit and scientific validation tests for the chemistry engine (Modules M01, M02, M03)."""

import pytest
import numpy as np
from src.chemistry.validation import validate_smiles, validate_batch_smiles, MoleculeValidationResult
from src.chemistry.descriptors import calculate_descriptors, MolecularDescriptors
from src.chemistry.fingerprints import generate_morgan_fingerprint, FingerprintResult


class TestMolecularValidation:
    """Tests for Module M01: Input Validation and Structure Identity."""

    def test_valid_smiles_ethanol(self):
        result = validate_smiles("CCO")
        assert result.is_valid is True
        assert result.canonical_smiles == "CCO"
        assert result.formula == "C2H6O"
        assert result.num_heavy_atoms == 3
        assert result.error_code is None

    def test_valid_smiles_aspirin(self):
        result = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert result.is_valid is True
        assert result.formula == "C9H8O4"
        assert result.num_heavy_atoms == 13
        assert result.error_message is None

    def test_whitespace_handling(self):
        result = validate_smiles("  CCO \n\t")
        assert result.is_valid is True
        assert result.canonical_smiles == "CCO"

    def test_empty_input_rejected(self):
        res1 = validate_smiles("")
        assert res1.is_valid is False
        assert res1.error_code == "E-001_EMPTY_INPUT"

        res2 = validate_smiles("    ")
        assert res2.is_valid is False
        assert res2.error_code == "E-001_EMPTY_INPUT"

    def test_invalid_type_rejected(self):
        result = validate_smiles(None)  # type: ignore
        assert result.is_valid is False
        assert result.error_code == "E-001_INVALID_TYPE"

    def test_malformed_smiles_rejected(self):
        # Unclosed ring
        res1 = validate_smiles("C1CC")
        assert res1.is_valid is False
        assert res1.error_code == "E-001_MALFORMED_SMILES"

        # Invalid chemical syntax
        res2 = validate_smiles("NotAChemicalStructure!!")
        assert res2.is_valid is False
        assert res2.error_code == "E-001_MALFORMED_SMILES"

    def test_batch_validation_fault_isolation(self):
        batch = ["CCO", "MALFORMED_SMILES_STRING", "c1ccccc1", ""]
        results = validate_batch_smiles(batch)
        assert len(results) == 4
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True
        assert results[3].is_valid is False


class TestMolecularDescriptors:
    """Tests for Module M02: Physicochemical Descriptors and Lipinski Rules."""

    def test_ethanol_descriptors(self):
        desc = calculate_descriptors("CCO")
        assert isinstance(desc, MolecularDescriptors)
        assert pytest.approx(desc.molecular_weight, 0.01) == 46.069
        assert desc.hbond_donors == 1
        assert desc.hbond_acceptors == 1
        assert desc.rotatable_bonds == 0
        assert pytest.approx(desc.tpsa, 0.1) == 20.23
        assert desc.lipinski_violations == 0
        assert desc.lipinski_pass is True

    def test_aspirin_descriptors(self):
        desc = calculate_descriptors("CC(=O)Oc1ccccc1C(=O)O")
        assert pytest.approx(desc.molecular_weight, 0.05) == 180.16
        assert desc.hbond_donors == 1
        assert desc.hbond_acceptors == 3
        assert desc.rotatable_bonds == 2
        assert desc.lipinski_violations == 0
        assert desc.lipinski_pass is True

    def test_caffeine_descriptors(self):
        # Caffeine: C8H10N4O2 (RDKit Lipinski counts 2 carbonyl oxygens + 1 basic imidazole N = 3)
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        desc = calculate_descriptors(caffeine_smiles)
        assert pytest.approx(desc.molecular_weight, 0.1) == 194.19
        assert desc.hbond_donors == 0
        assert desc.hbond_acceptors == 3
        assert desc.lipinski_violations == 0
        assert desc.lipinski_pass is True

    def test_lipinski_violation_detection_atorvastatin(self):
        # Atorvastatin: MW ~ 558.65 Da (> 500 Da)
        atorvastatin_smiles = (
            "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
        )
        desc = calculate_descriptors(atorvastatin_smiles)
        assert desc.molecular_weight > 500.0
        assert desc.lipinski_violations >= 1
        assert desc.lipinski_pass is False
        assert any("Molecular weight" in v for v in desc.violation_details)

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_descriptors("NOT_A_VALID_SMILES")

        with pytest.raises(ValueError):
            calculate_descriptors("")


class TestMolecularFingerprints:
    """Tests for Module M03: Molecular Fingerprints."""

    def test_fingerprint_dimensions_and_properties(self):
        fp_res = generate_morgan_fingerprint("CCO", radius=2, n_bits=2048)
        assert isinstance(fp_res, FingerprintResult)
        assert fp_res.n_bits == 2048
        assert fp_res.radius == 2
        assert fp_res.numpy_vector.shape == (2048,)
        assert fp_res.numpy_vector.dtype == np.int8
        assert np.sum(fp_res.numpy_vector) > 0  # Active bits present

    def test_fingerprint_determinism(self):
        fp1 = generate_morgan_fingerprint("CC(=O)Nc1ccc(O)cc1", radius=2, n_bits=2048)
        fp2 = generate_morgan_fingerprint("CC(=O)Nc1ccc(O)cc1", radius=2, n_bits=2048)
        assert np.array_equal(fp1.numpy_vector, fp2.numpy_vector)

    def test_structural_distinction(self):
        fp_ethanol = generate_morgan_fingerprint("CCO", radius=2, n_bits=2048)
        fp_benzene = generate_morgan_fingerprint("c1ccccc1", radius=2, n_bits=2048)
        assert not np.array_equal(fp_ethanol.numpy_vector, fp_benzene.numpy_vector)

    def test_bit_info_atom_environment_mapping(self):
        fp_res = generate_morgan_fingerprint("CC(=O)Oc1ccccc1C(=O)O", return_bit_info=True)
        assert len(fp_res.bit_info) > 0
        # Check that bit_info keys are bit integers and values are collections of (atom_idx, radius)
        for bit_id, envs in fp_res.bit_info.items():
            assert isinstance(bit_id, int)
            assert isinstance(envs, (list, tuple))
            for atom_idx, radius in envs:
                assert isinstance(atom_idx, int)
                assert radius in [0, 1, 2]

    def test_invalid_input_handling(self):
        with pytest.raises(ValueError):
            generate_morgan_fingerprint("INVALID_SMILES")
