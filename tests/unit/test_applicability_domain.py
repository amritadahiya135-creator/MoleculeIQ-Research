"""Unit tests for dynamic applicability domain and chemical space boundaries."""

import os
import pytest
import numpy as np

from src.models.inference import ToxicityInferenceService, ClinToxPrediction
from src.chemistry.fingerprints import generate_morgan_fingerprint


class TestDynamicApplicabilityDomain:
    """Tests for dynamic applicability domain distance assessment."""

    @pytest.fixture
    def service(self):
        return ToxicityInferenceService()

    def test_exact_training_match_is_within_domain(self, service):
        # Aspirin is an exact training compound
        pred = service.predict_clintox("CC(=O)Oc1ccccc1C(=O)O")
        assert pred.status == "SUCCESS"
        assert pred.applicability_domain_status == "WITHIN_DOMAIN"
        assert pred.applicability_score == 1.0
        assert pred.applicability_warning is None

    def test_structurally_similar_molecule_is_within_domain(self, service):
        # Methyl salicylate (closely related to aspirin)
        pred = service.predict_clintox("COC(=O)c1ccccc1O")
        assert pred.status == "SUCCESS"
        assert pred.applicability_domain_status in ("WITHIN_DOMAIN", "NEAR_BOUNDARY")
        assert pred.applicability_score is not None
        assert pred.applicability_score >= 0.25

    def test_out_of_distribution_inorganics_are_outside_domain(self, service):
        # Cisplatin: platinum coordination complex
        pred_cis = service.predict_clintox("Cl[Pt](Cl)(N)N")
        assert pred_cis.status == "SUCCESS"
        assert pred_cis.applicability_domain_status == "OUTSIDE_DOMAIN"
        assert pred_cis.applicability_score is not None
        assert pred_cis.applicability_score < 0.25
        assert pred_cis.applicability_warning is not None
        assert "Outside applicability domain" in pred_cis.applicability_warning

        # Xenon difluoride: noble gas halide
        pred_xe = service.predict_clintox("F[Xe]F")
        assert pred_xe.status == "SUCCESS"
        assert pred_xe.applicability_domain_status == "OUTSIDE_DOMAIN"
        assert pred_xe.applicability_score is not None
        assert pred_xe.applicability_score < 0.25
        assert pred_xe.applicability_warning is not None
        assert "Outside applicability domain" in pred_xe.applicability_warning

    def test_invalid_smiles_reports_unavailable_domain(self, service):
        pred = service.predict_clintox("INVALID_SMILES_STRING")
        assert pred.status == "INVALID_INPUT"
        assert pred.applicability_domain_status == "UNAVAILABLE"

    def test_assess_applicability_domain_missing_fps_fallback(self):
        service = ToxicityInferenceService(clintox_artifact_dir="nonexistent_dir")
        status, score, warning = service.assess_applicability_domain(np.zeros(2048, dtype=np.int8))
        assert status == "UNAVAILABLE"
        assert score is None
        assert warning is None
