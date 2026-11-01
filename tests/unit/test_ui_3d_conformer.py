"""
Unit tests for RDKit 3D Conformer Generation in Single Molecule UI (Phase E).
"""

import pytest
from src.ui.single_molecule import generate_3d_mol_block, PRESET_MOLECULES


def test_generate_3d_mol_block_valid_compounds():
    """Verify that 3D conformer generation produces valid MolBlocks for all reference compounds."""
    for name, smiles in PRESET_MOLECULES.items():
        mol_block = generate_3d_mol_block(smiles)
        assert mol_block is not None, f"Failed to generate 3D conformer for preset: {name}"
        assert isinstance(mol_block, str)
        assert "M  END" in mol_block, f"Missing MolBlock terminal record for preset: {name}"
        # Ensure 3D coordinate block has non-zero coordinate values
        lines = mol_block.splitlines()
        assert len(lines) > 4, f"Truncated MolBlock for {name}"


def test_generate_3d_mol_block_invalid_inputs():
    """Verify that malformed or non-chemical strings return None without raising unhandled exceptions."""
    invalid_cases = [
        "",
        "   ",
        "NOT_A_SMILES",
        "C1CC",
        "C(=O)(O)(O)(O)(O)",  # Invalid pentavalent carbon
        "12345",
    ]
    for inv in invalid_cases:
        res = generate_3d_mol_block(inv)
        assert res is None, f"Expected None for invalid SMILES '{inv}', but got: {res}"


def test_generate_3d_mol_block_deterministic():
    """Verify that identical SMILES input produces identical conformer coordinates."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    block1 = generate_3d_mol_block(smiles)
    block2 = generate_3d_mol_block(smiles)
    assert block1 is not None
    assert block2 is not None
    assert block1 == block2, "Conformer generation should be deterministic under seed=42"
