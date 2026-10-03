"""Integration tests for model training, persistence, and inference (Modules M04, M05)."""

import os
import pytest
from src.models.inference import ToxicityInferenceService, ClinToxPrediction, Tox21MultiPrediction


class TestToxicityInference:
    """Tests for model inference service."""

    @pytest.fixture
    def inference_service(self):
        return ToxicityInferenceService()

    def test_artifacts_exist(self, inference_service):
        assert inference_service.is_clintox_available() is True
        assert inference_service.is_tox21_available() is True

    def test_clintox_prediction_valid_molecule(self, inference_service):
        # Test with Aspirin
        res = inference_service.predict_clintox("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(res, ClinToxPrediction)
        assert res.status == "SUCCESS"
        assert 0.0 <= res.probability <= 1.0
        assert res.predicted_class in [0, 1]
        assert res.is_toxic == (res.predicted_class == 1)
        assert res.error_message is None

    def test_clintox_prediction_invalid_molecule(self, inference_service):
        res = inference_service.predict_clintox("MALFORMED_SMILES")
        assert res.status == "INVALID_INPUT"
        assert res.error_message is not None

    def test_tox21_multi_prediction_valid_molecule(self, inference_service):
        res = inference_service.predict_tox21("CC(=O)Nc1ccc(O)cc1")  # Paracetamol
        assert isinstance(res, Tox21MultiPrediction)
        assert res.status == "SUCCESS"
        assert res.total_endpoints == 12
        assert len(res.endpoints) == 12

        # Check an individual endpoint
        ar_res = res.endpoints["NR-AR"]
        assert ar_res.status == "SUCCESS"
        assert 0.0 <= ar_res.probability <= 1.0
        assert ar_res.predicted_class in [0, 1]

    def test_tox21_prediction_invalid_molecule(self, inference_service):
        res = inference_service.predict_tox21("NOT_A_SMILES")
        assert res.status == "INVALID_INPUT"
        assert res.total_endpoints == 0
