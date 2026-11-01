"""Unit and integration tests for Phase A.1: Batch Applicability Domain Exposure & CSV Export."""

import io
import pytest
import pandas as pd
import numpy as np

from src.services.orchestration import ScreeningOrchestrator
from src.services.contracts import BatchScreeningResult, BatchCandidateRecord
from src.utils.security import sanitize_dataframe_for_csv


class TestBatchApplicabilityDomainExposure:
    """Rigorous verification of applicability domain exposure in batch screening and CSV export."""

    @pytest.fixture
    def orchestrator(self):
        return ScreeningOrchestrator()

    def test_case_1_valid_in_domain_molecule(self, orchestrator):
        """Case 1: Exact training match (Aspirin) is WITHIN_DOMAIN with max score 1.0."""
        res = orchestrator.screen_batch(["CC(=O)Oc1ccccc1C(=O)O"], names_list=["Aspirin"])
        df = res.results_dataframe
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Status"] == "Valid"
        assert row["Applicability_Domain_Status"] == "WITHIN_DOMAIN"
        assert row["Applicability_Domain_Score"] == 1.0
        assert row["Applicability_Warning"] == ""

    def test_case_2_valid_borderline_domain_molecule(self, orchestrator):
        """Case 2: Structurally similar compound (Methyl salicylate) is NEAR_BOUNDARY or WITHIN_DOMAIN."""
        res = orchestrator.screen_batch(["COC(=O)c1ccccc1O"], names_list=["Methyl_salicylate"])
        df = res.results_dataframe
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Status"] == "Valid"
        assert row["Applicability_Domain_Status"] in ("WITHIN_DOMAIN", "NEAR_BOUNDARY")
        assert row["Applicability_Domain_Score"] is not None
        assert row["Applicability_Domain_Score"] >= 0.25
        if row["Applicability_Domain_Status"] == "NEAR_BOUNDARY":
            assert "Near applicability domain boundary" in row["Applicability_Warning"]

    def test_case_3_clearly_out_of_domain_molecule(self, orchestrator):
        """Case 3: Inorganics/atypical scaffolds (Cisplatin) are OUTSIDE_DOMAIN with warning."""
        res = orchestrator.screen_batch(["Cl[Pt](Cl)(N)N"], names_list=["Cisplatin"])
        df = res.results_dataframe
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Status"] == "Valid"
        assert row["Applicability_Domain_Status"] == "OUTSIDE_DOMAIN"
        assert row["Applicability_Domain_Score"] is not None
        assert row["Applicability_Domain_Score"] < 0.25
        assert "Outside applicability domain" in row["Applicability_Warning"]

    def test_case_4_malformed_smiles_isolated_without_crashing(self, orchestrator):
        """Case 4: Malformed SMILES string is isolated and tagged UNAVAILABLE."""
        res = orchestrator.screen_batch(["NOT_A_VALID_SMILES_999"], names_list=["Bad_Input"])
        df = res.results_dataframe
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Status"] == "Invalid"
        assert row["Applicability_Domain_Status"] == "UNAVAILABLE"
        assert pd.isna(row["Applicability_Domain_Score"]) or row["Applicability_Domain_Score"] is None
        assert row["Applicability_Warning"] == ""
        assert row["Error"] != ""

    def test_case_5_empty_null_smiles_isolated(self, orchestrator):
        """Case 5: Empty/whitespace SMILES is isolated with UNAVAILABLE domain status."""
        res = orchestrator.screen_batch(["", "   "], names_list=["Empty_1", "Empty_2"])
        df = res.results_dataframe
        assert len(df) == 2
        for _, row in df.iterrows():
            assert row["Status"] == "Invalid"
            assert row["Applicability_Domain_Status"] == "UNAVAILABLE"

    def test_case_6_duplicate_molecules_deterministic(self, orchestrator):
        """Case 6: Duplicate molecules are processed deterministically with identical domain properties."""
        smi = "CC(=O)Oc1ccccc1C(=O)O"
        res = orchestrator.screen_batch([smi, smi], names_list=["Aspirin_1", "Aspirin_2"])
        df = res.results_dataframe
        assert len(df) == 2
        assert df.iloc[0]["Applicability_Domain_Status"] == df.iloc[1]["Applicability_Domain_Status"] == "WITHIN_DOMAIN"
        assert df.iloc[0]["Applicability_Domain_Score"] == df.iloc[1]["Applicability_Domain_Score"] == 1.0

    def test_case_7_mixed_domain_batch_execution(self, orchestrator):
        """Case 7: Mixed batch of in-domain, out-of-domain, borderline, and invalid completes fully."""
        smiles_batch = [
            "CC(=O)Oc1ccccc1C(=O)O",         # In-domain (Aspirin)
            "Cl[Pt](Cl)(N)N",                # Out-of-domain (Cisplatin)
            "COC(=O)c1ccccc1O",              # Borderline / related (Methyl salicylate)
            "GARBAGE_SMILES_XYZ",            # Invalid
        ]
        names = ["Aspirin", "Cisplatin", "Methyl_Salicylate", "Invalid_Mol"]

        res = orchestrator.screen_batch(smiles_batch, names_list=names)
        assert isinstance(res, BatchScreeningResult)
        assert res.total_records == 4
        assert res.valid_records == 3
        assert res.invalid_records == 1

        df = res.results_dataframe
        assert len(df) == 4

        # Validate candidates individually by name
        asp = df[df["Name"] == "Aspirin"].iloc[0]
        assert asp["Applicability_Domain_Status"] == "WITHIN_DOMAIN"
        assert asp["Applicability_Domain_Score"] == 1.0
        assert asp["Applicability_Warning"] == ""

        cis = df[df["Name"] == "Cisplatin"].iloc[0]
        assert cis["Applicability_Domain_Status"] == "OUTSIDE_DOMAIN"
        assert cis["Applicability_Domain_Score"] < 0.25
        assert "Outside applicability domain" in cis["Applicability_Warning"]

        bad = df[df["Name"] == "Invalid_Mol"].iloc[0]
        assert bad["Status"] == "Invalid"
        assert bad["Applicability_Domain_Status"] == "UNAVAILABLE"

    def test_csv_export_schema_and_value_preservation(self, orchestrator):
        """Verify CSV export contains domain columns with exact values preserved."""
        smiles_batch = ["CC(=O)Oc1ccccc1C(=O)O", "Cl[Pt](Cl)(N)N"]
        res = orchestrator.screen_batch(smiles_batch, names_list=["Aspirin", "Cisplatin"])

        sanitized_df = sanitize_dataframe_for_csv(res.results_dataframe)
        csv_bytes = sanitized_df.to_csv(index=False)
        reloaded = pd.read_csv(io.StringIO(csv_bytes))

        # Check column presence
        assert "Applicability_Domain_Status" in reloaded.columns
        assert "Applicability_Domain_Score" in reloaded.columns
        assert "Applicability_Warning" in reloaded.columns

        # Verify values
        asp_reloaded = reloaded[reloaded["Name"] == "Aspirin"].iloc[0]
        assert asp_reloaded["Applicability_Domain_Status"] == "WITHIN_DOMAIN"
        assert asp_reloaded["Applicability_Domain_Score"] == 1.0

        cis_reloaded = reloaded[reloaded["Name"] == "Cisplatin"].iloc[0]
        assert cis_reloaded["Applicability_Domain_Status"] == "OUTSIDE_DOMAIN"
        assert "Outside applicability domain" in str(cis_reloaded["Applicability_Warning"])

    def test_ranking_unchanged_by_applicability_domain(self, orchestrator):
        """
        Verify that candidate prioritization ranking is NOT modified by Phase A.1.
        Candidates must remain ordered purely by Overall_Priority_Score descending.
        """
        smiles_batch = ["CC(=O)Oc1ccccc1C(=O)O", "Cl[Pt](Cl)(N)N", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        res = orchestrator.screen_batch(smiles_batch, names_list=["Aspirin", "Cisplatin", "Caffeine"])
        df = res.results_dataframe

        valid_df = df[df["Status"] == "Valid"]
        scores = valid_df["Overall_Priority_Score"].tolist()
        # Verify strictly monotonically non-increasing (descending)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_csv_formula_injection_defense_on_warning_column(self):
        """Verify that any formula-injection payload in applicability warning is neutralized."""
        df_attack = pd.DataFrame({
            "Name": ["Comp_1"],
            "Applicability_Domain_Status": ["=CMD|' /C calc'!A0"],
            "Applicability_Warning": ["+HYPERLINK('http://evil.com')"],
            "Overall_Priority_Score": [75.0],
        })
        sanitized = sanitize_dataframe_for_csv(df_attack)
        csv_text = sanitized.to_csv(index=False)

        reloaded = pd.read_csv(io.StringIO(csv_text))
        # Excel formula triggers neutralized with leading single quote
        assert reloaded["Applicability_Domain_Status"].iloc[0].startswith("'=")
        assert reloaded["Applicability_Warning"].iloc[0].startswith("'+")
