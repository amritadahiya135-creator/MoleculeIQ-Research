"""End-to-end integration test for Workflow B: Batch Molecular Screening."""

import io
import pytest
import pandas as pd
from src.services.orchestration import ScreeningOrchestrator
from src.services.contracts import BatchScreeningResult


class TestBatchScreeningWorkflow:
    """Verifies complete Workflow B execution and row fault isolation."""

    @pytest.fixture
    def orchestrator(self):
        return ScreeningOrchestrator()

    def test_mixed_valid_and_invalid_batch_screening(self, orchestrator):
        smiles_batch = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Valid (Aspirin)
            "MALFORMED_INPUT_123",    # Invalid
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Valid (Caffeine)
            "C1CC",                  # Invalid
        ]
        names = ["Aspirin", "Bad_1", "Caffeine", "Bad_2"]

        res = orchestrator.screen_batch(smiles_batch, names_list=names)

        assert isinstance(res, BatchScreeningResult)
        assert res.total_records == 4
        assert res.valid_records == 2
        assert res.invalid_records == 2

        df = res.results_dataframe
        assert len(df) == 4

        # Verify fault isolation: valid rows processed, invalid rows preserved with errors
        valid_rows = df[df["Status"] == "Valid"]
        invalid_rows = df[df["Status"] == "Invalid"]

        assert len(valid_rows) == 2
        assert len(invalid_rows) == 2

        # Check ranking assignment
        assert set(valid_rows["Rank"]) == {1, 2}
        assert all(invalid_rows["Rank"] == "-")

        # Verify export integrity
        csv_data = df.to_csv(index=False)
        reloaded_df = pd.read_csv(io.StringIO(csv_data))
        assert len(reloaded_df) == 4
        assert "Overall_Priority_Score" in reloaded_df.columns
        assert "Top_Reference_Drug" in reloaded_df.columns
        assert "Applicability_Domain_Status" in reloaded_df.columns
        assert "Applicability_Domain_Score" in reloaded_df.columns
        assert "Applicability_Warning" in reloaded_df.columns

        # Verify domain exposure on valid candidates
        asp_row = valid_rows[valid_rows["Name"] == "Aspirin"].iloc[0]
        assert asp_row["Applicability_Domain_Status"] == "WITHIN_DOMAIN"
        assert asp_row["Applicability_Domain_Score"] == 1.0

        # Verify domain status on invalid rows
        bad_row = invalid_rows.iloc[0]
        assert bad_row["Applicability_Domain_Status"] == "UNAVAILABLE"
