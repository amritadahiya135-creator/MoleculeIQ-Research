"""Unit tests for ClinTox dataset ingestion, deduplication, and identity validation."""

import os
import pytest
import pandas as pd
import numpy as np

from src.models.clintox import (
    prepare_clintox_dataset,
    audit_clintox_duplicates,
)


class TestClinToxDataIntegrity:
    """Tests for ClinTox dataset deduplication and conflict resolution."""

    def test_audit_clintox_duplicates_identifies_conflicts(self):
        report = audit_clintox_duplicates()
        assert report["total_raw_rows"] == 1484
        assert report["invalid_smiles_count"] == 4
        assert report["unique_canonical_smiles"] == 1461
        assert report["duplicate_groups_count"] == 19
        assert report["conflicting_groups_count"] == 19
        assert len(report["conflicting_groups"]) == 19

    def test_prepare_clintox_conservative_toxic_policy(self):
        X, y, df_clean = prepare_clintox_dataset(duplicate_policy="conservative_toxic")
        assert len(X) == 1461
        assert len(y) == 1461
        assert len(df_clean) == 1461
        # In conservative_toxic, the 19 conflicting groups are assigned CT_TOX=1
        assert int(np.sum(y == 1)) == 112
        assert int(np.sum(y == 0)) == 1349

        # Check metadata in df_clean
        assert "duplicate_report" in df_clean.attrs
        dup_rep = df_clean.attrs["duplicate_report"]
        assert dup_rep["conflicts_resolved_count"] == 19
        assert dup_rep["duplicate_policy"] == "conservative_toxic"

        # Check provenance columns
        assert "canonical_smiles" in df_clean.columns
        assert "source_indices" in df_clean.columns
        assert "raw_smiles" in df_clean.columns
        assert "is_duplicate" in df_clean.columns
        assert "is_conflict" in df_clean.columns

    def test_prepare_clintox_drop_conflicts_policy(self):
        X, y, df_clean = prepare_clintox_dataset(duplicate_policy="drop_conflicts")
        # 1461 - 19 conflicting groups = 1442
        assert len(X) == 1442
        assert len(y) == 1442
        assert int(np.sum(y == 1)) == 93
        assert int(np.sum(y == 0)) == 1349

    def test_no_identical_smiles_leakage_across_splits(self):
        from sklearn.model_selection import train_test_split
        X, y, df_clean = prepare_clintox_dataset(duplicate_policy="conservative_toxic")
        smiles_list = df_clean["canonical_smiles"].tolist()

        # Random split
        train_smi, test_smi = train_test_split(smiles_list, test_size=0.2, random_state=42)
        overlap = set(train_smi).intersection(set(test_smi))
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping molecules across train and test!"

    def test_invalid_duplicate_policy_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported duplicate_policy"):
            prepare_clintox_dataset(duplicate_policy="invalid_policy_name")

    def test_malformed_csv_raises_error(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("random_col1,random_col2\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain 'smiles' and 'CT_TOX'"):
            prepare_clintox_dataset(csv_path=str(bad_csv))
