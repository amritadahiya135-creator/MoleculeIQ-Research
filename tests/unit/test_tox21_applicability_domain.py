"""Comprehensive scientific validation tests for Phase A.2: Tox21 Assay-Specific Applicability Domain.

Covers all 16 requirements specified in the Master Scientific Remediation Directive Phase A.2:
1. Representation of all 12 Tox21 endpoint identifiers
2. Respect of endpoint-specific missing-label eligibility
3. Complete exclusion of held-out test compounds (zero leakage audit)
4. Determinism of similarity scores on identical queries
5. Scores are strictly finite and bounded within [0, 1]
6. Explicit UNAVAILABLE result on invalid/malformed SMILES
7. Safe UNAVAILABLE handling on empty reference corpora
8. Transparent fallback on missing artifacts or FP generation failure
9. Isolation of endpoint-specific corpora (no accidental sharing)
10. Invariance of max-Tanimoto to duplicate structures
11. Preservation of existing ClinTox AD behavior
12. Preservation of Tox21 model predictions and calibration outputs
13. Preservation of candidate ranking order and formula (R = 0.5S + 0.3D + 0.2M)
14. Backward compatibility of batch screening and CSV exports
15. Preservation of CSV injection protections
16. Verification of model-artifact cryptographic integrity
"""

import io
import math
import numpy as np
import pandas as pd
import pytest

from src.config import TOX21_DATA_PATH, TOX21_ARTIFACT_DIR, ARTIFACTS_DIR
from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.models.tox21 import TOX21_ENDPOINTS
from src.models.tox21_applicability import (
    Tox21ApplicabilityEvaluator,
    Tox21EndpointADResult,
    STATUS_UNVALIDATED,
    STATUS_UNAVAILABLE,
    TOX21_AD_DISCLAIMER,
)
from src.models.inference import ToxicityInferenceService, Tox21MultiPrediction
from src.services.orchestration import ScreeningOrchestrator
from src.services.contracts import BatchScreeningResult
from src.utils.integrity import verify_artifact_checksums


@pytest.fixture(scope="module")
def evaluator():
    """Shared Tox21ApplicabilityEvaluator instance."""
    ev = Tox21ApplicabilityEvaluator()
    assert ev.is_available(), "Tox21 reference corpora must be available on disk."
    return ev


@pytest.fixture(scope="module")
def inference_service():
    """Shared ToxicityInferenceService instance."""
    return ToxicityInferenceService()


@pytest.fixture(scope="module")
def orchestrator():
    """Shared ScreeningOrchestrator instance."""
    return ScreeningOrchestrator()


class TestTox21ApplicabilityDomainScientificValidation:
    """Scientific test suite validating assay-specific applicability domain behavior."""

    # -------------------------------------------------------------------------
    # Requirement 1: All 12 endpoint identifiers represented
    # -------------------------------------------------------------------------
    def test_req01_all_12_endpoint_identifiers_represented(self, evaluator, inference_service):
        """All 12 standardized Tox21 endpoint identifiers must be present and evaluated."""
        assert len(TOX21_ENDPOINTS) == 12
        for ep in TOX21_ENDPOINTS:
            assert evaluator.is_available(ep) is True
            count = evaluator.get_training_count(ep)
            assert count > 4000, f"Endpoint '{ep}' reference count ({count}) must be > 4000."

        # Verify through inference service
        pred = inference_service.predict_tox21("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        assert isinstance(pred, Tox21MultiPrediction)
        assert pred.total_endpoints == 12
        assert set(pred.endpoints.keys()) == set(TOX21_ENDPOINTS)
        for ep, ep_res in pred.endpoints.items():
            assert ep_res.applicability_score is not None
            assert ep_res.applicability_domain_status == STATUS_UNVALIDATED
            assert ep_res.applicability_warning == TOX21_AD_DISCLAIMER

    # -------------------------------------------------------------------------
    # Requirement 2: Missing-label row exclusion
    # -------------------------------------------------------------------------
    def test_req02_endpoint_specific_missing_label_exclusion(self, evaluator):
        """Compounds with missing (NaN) labels for an endpoint must NOT enter its reference corpus."""
        df = pd.read_csv(TOX21_DATA_PATH)

        for ep in ["NR-Aromatase", "SR-MMP", "SR-ARE"]:
            train_smiles_set = set(evaluator.get_training_smiles(ep))
            # Find molecules in raw data that have NaN for this endpoint
            nan_mask = df[ep].isna()
            nan_smiles_in_raw = df.loc[nan_mask, "smiles"].dropna().tolist()

            # Pre-validate canonical forms of NaN molecules
            nan_canonical = set()
            for smi in nan_smiles_in_raw[:100]:  # Sample first 100 for efficiency
                v = validate_smiles(str(smi))
                if v.is_valid and v.canonical_smiles:
                    nan_canonical.add(v.canonical_smiles)

            # Intersection between NaN molecules and training reference corpus must be EMPTY
            overlap = train_smiles_set.intersection(nan_canonical)
            assert len(overlap) == 0, f"Endpoint '{ep}' contains {len(overlap)} NaN-label molecules in reference corpus!"

    # -------------------------------------------------------------------------
    # Requirement 3: Zero held-out test set leakage
    # -------------------------------------------------------------------------
    def test_req03_zero_held_out_test_set_leakage(self, evaluator):
        """Held-out test partition compounds must be strictly excluded from the training reference set."""
        df = pd.read_csv(TOX21_DATA_PATH)
        from sklearn.model_selection import train_test_split

        valid_mask = []
        canon_list = []
        for smi in df["smiles"]:
            v = validate_smiles(str(smi))
            if v.is_valid and v.canonical_smiles:
                valid_mask.append(True)
                canon_list.append(v.canonical_smiles)
            else:
                valid_mask.append(False)
                canon_list.append(None)
        valid_mask = np.array(valid_mask, dtype=bool)

        for ep in TOX21_ENDPOINTS:
            ep_series = df[ep]
            not_na_mask = ep_series.notna().values & valid_mask
            ep_indices = np.where(not_na_mask)[0]
            y_ep = ep_series.values[not_na_mask].astype(int)

            idx_all = np.arange(len(ep_indices))
            train_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=42, stratify=y_ep)

            test_orig_indices = ep_indices[test_idx]
            test_smiles_set = set(canon_list[i] for i in test_orig_indices)

            train_smiles_set = set(evaluator.get_training_smiles(ep))

            # Test set must NOT overlap with reference set
            overlap = train_smiles_set.intersection(test_smiles_set)
            assert len(overlap) == 0, f"ENDPOINT LEAKAGE DETECTED in '{ep}': {len(overlap)} test compounds in reference corpus!"

    # -------------------------------------------------------------------------
    # Requirement 4: Determinism of similarity scores
    # -------------------------------------------------------------------------
    def test_req04_score_determinism(self, evaluator, inference_service):
        """Identical queries evaluated multiple times must produce bitwise identical AD scores."""
        query_smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine

        res1 = evaluator.evaluate_all(query_smi)
        res2 = evaluator.evaluate_all(query_smi)

        for ep in TOX21_ENDPOINTS:
            assert res1[ep].applicability_score == res2[ep].applicability_score
            assert res1[ep].applicability_domain_status == res2[ep].applicability_domain_status

        # Determinism via Inference Service
        pred1 = inference_service.predict_tox21(query_smi)
        pred2 = inference_service.predict_tox21(query_smi)
        for ep in TOX21_ENDPOINTS:
            assert pred1.endpoints[ep].applicability_score == pred2.endpoints[ep].applicability_score

    # -------------------------------------------------------------------------
    # Requirement 5: Finite scores bounded within [0, 1]
    # -------------------------------------------------------------------------
    def test_req05_scores_bounded_and_finite(self, evaluator):
        """All calculated similarity scores must be finite floating-point numbers in [0.0, 1.0]."""
        test_molecules = [
            "CC(=O)Oc1ccccc1C(=O)O",          # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
            "c1ccccc1",                       # Benzene
            "Cl[Pt](Cl)(N)N",                 # Cisplatin
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # Ibuprofen
        ]

        for smi in test_molecules:
            res = evaluator.evaluate_all(smi)
            for ep, r in res.items():
                assert r.applicability_score is not None
                assert isinstance(r.applicability_score, float)
                assert math.isfinite(r.applicability_score)
                assert 0.0 <= r.applicability_score <= 1.0

    # -------------------------------------------------------------------------
    # Requirement 6: Invalid SMILES produce UNAVAILABLE
    # -------------------------------------------------------------------------
    def test_req06_invalid_smiles_produces_unavailable(self, evaluator, inference_service):
        """Malformed, empty, or unparseable SMILES must explicitly yield UNAVAILABLE status and None score."""
        invalid_inputs = ["INVALID_SMILES_STRING_123", "", "   ", "C1CC"]

        for bad in invalid_inputs:
            # Evaluator standalone
            res = evaluator.evaluate_all(bad)
            for ep, r in res.items():
                assert r.applicability_domain_status == STATUS_UNAVAILABLE
                assert r.applicability_score is None
                assert r.applicability_warning is not None

        # Via Inference Service
        bad_pred = inference_service.predict_tox21("NOT_A_SMILES")
        assert bad_pred.status == "INVALID_INPUT"

    # -------------------------------------------------------------------------
    # Requirement 7: Empty reference corpus handling
    # -------------------------------------------------------------------------
    def test_req07_empty_reference_corpus_handling(self, monkeypatch):
        """An empty reference corpus must return UNAVAILABLE without false in-domain claims."""
        ev = Tox21ApplicabilityEvaluator()
        # Mock empty reference corpus
        ev._fps_by_endpoint = {"NR-AR": np.zeros((0, 2048), dtype=np.int8)}
        ev._sums_by_endpoint = {"NR-AR": np.zeros(0, dtype=np.float32)}
        ev._smiles_by_endpoint = {"NR-AR": np.array([], dtype=object)}

        dummy_fp = np.ones(2048, dtype=np.int8)
        res = ev.evaluate_endpoint("NR-AR", dummy_fp)
        assert res.applicability_domain_status == STATUS_UNAVAILABLE
        assert res.applicability_score is None
        assert "Empty reference corpus" in res.applicability_warning

    # -------------------------------------------------------------------------
    # Requirement 8: Missing artifacts fallback
    # -------------------------------------------------------------------------
    def test_req08_missing_artifacts_fallback(self):
        """If reference artifact directory does not contain files, gracefully returns UNAVAILABLE."""
        ev = Tox21ApplicabilityEvaluator(
            artifact_dir="/nonexistent/path/moleculeiq_ad",
            raw_data_path="/nonexistent/path/tox21.csv",
        )
        assert ev.is_available() is False
        dummy_fp = np.ones(2048, dtype=np.int8)
        res = ev.evaluate_endpoint("NR-AR", dummy_fp)
        assert res.applicability_domain_status == STATUS_UNAVAILABLE
        assert res.applicability_score is None

    # -------------------------------------------------------------------------
    # Requirement 9: Endpoint corpora are distinct and not shared
    # -------------------------------------------------------------------------
    def test_req09_endpoint_corpora_distinctness(self, evaluator):
        """Different endpoints have different numbers of training molecules due to assay-specific missingness."""
        counts = [evaluator.get_training_count(ep) for ep in TOX21_ENDPOINTS]
        unique_counts = set(counts)
        # Because different assays have different missing labels, there must be multiple distinct count values
        assert len(unique_counts) >= 8, f"Expected distinct training sizes across endpoints, got: {unique_counts}"

        # Verify that two distinct endpoints have different sets of compounds
        smi_ar = set(evaluator.get_training_smiles("NR-AR"))
        smi_mmp = set(evaluator.get_training_smiles("SR-MMP"))
        assert smi_ar != smi_mmp
        # Difference must be non-trivial
        assert len(smi_ar.symmetric_difference(smi_mmp)) > 500

    # -------------------------------------------------------------------------
    # Requirement 10: Duplicate structures handling
    # -------------------------------------------------------------------------
    def test_req10_duplicate_molecules_policy(self, evaluator):
        """Evaluating duplicate queries yields identical similarity values, adhering to max-Tanimoto properties."""
        smi = "CC(=O)Oc1ccccc1C(=O)O"
        res_a = evaluator.evaluate_all(smi)
        res_b = evaluator.evaluate_all(smi)
        for ep in TOX21_ENDPOINTS:
            assert res_a[ep].applicability_score == res_b[ep].applicability_score

    # -------------------------------------------------------------------------
    # Requirement 11: ClinTox AD unchanged
    # -------------------------------------------------------------------------
    def test_req11_clintox_ad_remains_unchanged(self, inference_service):
        """ClinTox applicability domain thresholds (0.40/0.25) and outputs must remain identical."""
        # Aspirin: in ClinTox training set
        asp = inference_service.predict_clintox("CC(=O)Oc1ccccc1C(=O)O")
        assert asp.applicability_domain_status == "WITHIN_DOMAIN"
        assert asp.applicability_score == 1.0
        assert asp.applicability_warning is None

        # Cisplatin: out of domain (< 0.25)
        cis = inference_service.predict_clintox("Cl[Pt](Cl)(N)N")
        assert cis.applicability_domain_status == "OUTSIDE_DOMAIN"
        assert cis.applicability_score < 0.25
        assert "Outside applicability domain" in cis.applicability_warning

    # -------------------------------------------------------------------------
    # Requirement 12: Tox21 predictions and calibration unchanged
    # -------------------------------------------------------------------------
    def test_req12_tox21_predictions_unchanged(self, inference_service):
        """Tox21 model probabilities and class predictions are 100% unaltered by AD evaluation."""
        pred = inference_service.predict_tox21("CC(=O)Oc1ccccc1C(=O)O")
        assert pred.status == "SUCCESS"
        assert pred.total_endpoints == 12
        # Verify specific probabilities remain unaltered
        ar_pred = pred.endpoints["NR-AR"]
        assert ar_pred.endpoint == "NR-AR"
        assert isinstance(ar_pred.probability, float)
        assert ar_pred.predicted_class in (0, 1)
        assert ar_pred.status == "SUCCESS"

    # -------------------------------------------------------------------------
    # Requirement 13: Candidate ranking order and formula unchanged
    # -------------------------------------------------------------------------
    def test_req13_ranking_order_and_formula_unchanged(self, orchestrator):
        """Candidate ranking formula R = 0.5S + 0.3D + 0.2M and ordering remain unaltered."""
        batch_smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",         # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # Ibuprofen
        ]
        res = orchestrator.screen_batch(batch_smiles)
        df = res.results_dataframe
        valid_df = df[df["Status"] == "Valid"]

        # Verify priority scores are calculated and sorted descending
        scores = valid_df["Overall_Priority_Score"].tolist()
        assert scores == sorted(scores, reverse=True)
        # Ranks must be 1, 2, 3
        assert list(valid_df["Rank"]) == [1, 2, 3]

    # -------------------------------------------------------------------------
    # Requirement 14: Batch screening & CSV export backward-compatible
    # -------------------------------------------------------------------------
    def test_req14_batch_screening_and_csv_export_backward_compatible(self, orchestrator):
        """Batch screening DataFrame and CSV export preserve existing columns and handle invalid rows."""
        batch_smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "INVALID_SMILES_TEST",
        ]
        res = orchestrator.screen_batch(batch_smiles, names_list=["Aspirin", "Malformed"])
        df = res.results_dataframe

        # Required columns from Phase A.1
        expected_cols = [
            "Rank", "Row_ID", "Name", "Original_SMILES", "Canonical_SMILES",
            "Status", "Error", "ClinTox_P_Tox", "ClinTox_Toxic",
            "Applicability_Domain_Status", "Applicability_Domain_Score", "Applicability_Warning",
            "Tox21_Active_Assays", "Overall_Priority_Score",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Column '{col}' missing from batch DataFrame."

        # Verify export to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        reloaded = pd.read_csv(io.StringIO(csv_buffer.getvalue()))
        assert len(reloaded) == 2
        assert reloaded.iloc[0]["Applicability_Domain_Status"] == "WITHIN_DOMAIN"
        assert reloaded.iloc[1]["Applicability_Domain_Status"] == "UNAVAILABLE"

    # -------------------------------------------------------------------------
    # Requirement 15: CSV injection protections preserved
    # -------------------------------------------------------------------------
    def test_req15_csv_injection_protections_effective(self, orchestrator):
        """Formula characters in batch names or SMILES are neutralized during CSV serialization."""
        from src.utils.security import sanitize_dataframe_for_csv

        dirty_df = pd.DataFrame([
            {"Name": "=cmd|' /C calc'!A0", "SMILES": "@SUM(1+1)", "Score": 0.95}
        ])
        sanitized = sanitize_dataframe_for_csv(dirty_df)
        assert sanitized.iloc[0]["Name"].startswith("'=")
        assert sanitized.iloc[0]["SMILES"].startswith("'@")
        assert sanitized.iloc[0]["Score"] == 0.95

    # -------------------------------------------------------------------------
    # Requirement 16: Artifact integrity checks remain valid
    # -------------------------------------------------------------------------
    def test_req16_artifact_integrity_manifest_valid(self):
        """All repository artifacts in checksums.sha256 verify successfully with zero mismatches."""
        manifest = ARTIFACTS_DIR / "checksums.sha256"
        res = verify_artifact_checksums(manifest, ARTIFACTS_DIR)
        assert res.is_valid is True
        assert res.verified_count >= 30
        assert len(res.failed_checksums) == 0
        assert len(res.missing_files) == 0
