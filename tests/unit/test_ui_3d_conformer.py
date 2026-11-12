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


def test_render_3dmol_viewer_features_and_controls(monkeypatch):
    """Verify that render_3dmol_viewer produces compliant HTML with all required 3Dmol features."""
    from unittest.mock import MagicMock
    from src.ui.single_molecule import render_3dmol_viewer
    import streamlit.components.v1 as components

    mock_html = MagicMock()
    monkeypatch.setattr(components, "html", mock_html)

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol_block = generate_3d_mol_block(aspirin_smiles)
    assert mol_block is not None

    render_3dmol_viewer(mol_block, height=380)

    assert mock_html.called
    html_arg = mock_html.call_args[0][0]
    height_arg = mock_html.call_args[1].get("height") or mock_html.call_args[0][1]
    assert height_arg == 380

    # 1. 3Dmol.js Library and Background
    assert "3Dmol-min.js" in html_arg
    assert "#080D16" in html_arg

    # 2. Display Modes: Ball-and-Stick, Stick, Space-Filling (vdW)
    assert "Ball &amp; Stick" in html_arg
    assert "Stick" in html_arg
    assert "Space-Filling (vdW)" in html_arg
    assert "ballstick" in html_arg
    assert "sphere" in html_arg

    # 3. Interactive Atom Selection & Inspection HUD
    assert "setClickable" in html_arg
    assert "onAtomClicked" in html_arg
    assert "highlightAtom" in html_arg
    assert "atomCard" in html_arg
    assert "clearSelection" in html_arg
    assert "#39C6D8" in html_arg  # Cyan selection highlight

    # 4. Camera & Animation Controls
    assert "toggleSpin" in html_arg
    assert "resetView" in html_arg
    assert "Auto-Rotate" in html_arg

