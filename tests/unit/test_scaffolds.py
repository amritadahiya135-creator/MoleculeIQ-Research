import pytest
from src.chemistry.scaffolds import (
    generate_bemis_murcko_scaffold,
    scaffold_train_test_split,
)


class TestBemisMurckoScaffolds:
    """Verifies Bemis-Murcko scaffold generation and scaffold-based dataset splitting."""

    def test_known_molecule_scaffolds(self):
        # Aspirin scaffold is benzene ring
        aspirin_scaf = generate_bemis_murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O")
        assert aspirin_scaf == "c1ccccc1"

        # Ethanol has no rings -> empty string
        ethanol_scaf = generate_bemis_murcko_scaffold("CCO")
        assert ethanol_scaf == ""

        # Empty or invalid SMILES returns empty string
        assert generate_bemis_murcko_scaffold("") == ""
        assert generate_bemis_murcko_scaffold("invalid_smiles_string") == ""

    def test_scaffold_train_test_split_zero_leakage(self):
        # Create a synthetic dataset with distinct scaffold clusters
        smiles_list = [
            # Benzene scaffold (6 molecules)
            "c1ccccc1C",
            "c1ccccc1CC",
            "c1ccccc1CCC",
            "c1ccccc1O",
            "c1ccccc1N",
            "c1ccccc1Cl",
            # Pyridine scaffold (4 molecules)
            "c1ccncc1C",
            "c1ccncc1CC",
            "c1ccncc1O",
            "c1ccncc1F",
            # Naphthalene scaffold (2 molecules)
            "c1ccc2ccccc2c1C",
            "c1ccc2ccccc2c1O",
            # Acyclics (3 molecules)
            "CCC",
            "CCCC",
            "CCCCC",
        ]

        train_idx, test_idx = scaffold_train_test_split(smiles_list, test_size=0.25, random_state=42)

        # 1. Total samples preserved
        assert len(train_idx) + len(test_idx) == len(smiles_list)

        # 2. Complete disjoint index sets
        assert set(train_idx).isdisjoint(set(test_idx))

        # 3. Complete disjoint cyclic scaffold sets (ZERO scaffold leakage)
        train_scaffolds = {
            generate_bemis_murcko_scaffold(smiles_list[i])
            for i in train_idx
            if generate_bemis_murcko_scaffold(smiles_list[i]) != ""
        }
        test_scaffolds = {
            generate_bemis_murcko_scaffold(smiles_list[i])
            for i in test_idx
            if generate_bemis_murcko_scaffold(smiles_list[i]) != ""
        }

        assert train_scaffolds.isdisjoint(test_scaffolds), (
            f"Scaffold leakage detected between train and test! Overlap: {train_scaffolds & test_scaffolds}"
        )
