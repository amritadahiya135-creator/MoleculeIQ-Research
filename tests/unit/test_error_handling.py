"""Tests verifying clear error handling for missing datasets, missing artifacts, and invalid inputs."""

import pytest
from src.chemistry.validation import validate_smiles
from src.chemistry.descriptors import calculate_descriptors
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.models.clintox import prepare_clintox_dataset
from src.models.inference import ToxicityInferenceService
from src.explainability.shap_analysis import ClinToxExplainer
from src.services.orchestration import ScreeningOrchestrator


class TestErrorHandlingAndEdgeCases:
    """Verifies that missing datasets, missing artifacts, and invalid SMILES produce explicit error messages."""

    def test_missing_dataset_file_produces_clear_error(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            prepare_clintox_dataset(csv_path="nonexistent_dataset_path.csv")
        assert "ClinTox CSV dataset not found" in str(exc_info.value)

    def test_missing_clintox_artifact_produces_clear_error(self):
        service = ToxicityInferenceService(clintox_artifact_dir="nonexistent_artifacts_dir")
        pred = service.predict_clintox("CCO")
        assert pred.status == "UNAVAILABLE"
        assert pred.error_message is not None
        assert "missing in 'nonexistent_artifacts_dir'" in pred.error_message

    def test_missing_tox21_artifact_produces_clear_error(self):
        service = ToxicityInferenceService(tox21_artifact_dir="nonexistent_tox21_dir")
        res = service.predict_tox21("CCO")
        assert res.status == "SUCCESS"  # Service handles endpoint-level failure
        # All individual endpoints should report UNAVAILABLE with clear error
        for ep_name, ep_pred in res.endpoints.items():
            assert ep_pred.status == "UNAVAILABLE"
            assert ep_pred.error_message == "Model artifact not found"

    def test_missing_explainer_model_produces_clear_error(self):
        explainer = ClinToxExplainer(model_path="nonexistent_explainer_model.joblib")
        res = explainer.explain_molecule("CCO")
        assert res.status == "MODEL_UNAVAILABLE"
        assert res.error_message is not None
        assert "not found" in res.error_message

    def test_invalid_smiles_in_chemistry_validation(self):
        res1 = validate_smiles("C1CC")
        assert res1.is_valid is False
        assert res1.error_code == "E-001_MALFORMED_SMILES"
        assert "Failed to parse SMILES" in res1.error_message

        res2 = validate_smiles("")
        assert res2.is_valid is False
        assert res2.error_code == "E-001_EMPTY_INPUT"

        res3 = validate_smiles(12345)  # type: ignore
        assert res3.is_valid is False
        assert res3.error_code == "E-001_INVALID_TYPE"

    def test_invalid_smiles_in_descriptors(self):
        with pytest.raises(ValueError) as exc1:
            calculate_descriptors("INVALID_SMILES")
        assert "Invalid SMILES" in str(exc1.value)

        with pytest.raises(ValueError) as exc2:
            calculate_descriptors("   ")
        assert "empty SMILES string" in str(exc2.value)

    def test_invalid_smiles_in_fingerprints(self):
        with pytest.raises(ValueError) as exc1:
            generate_morgan_fingerprint("INVALID_SMILES")
        assert "Invalid SMILES" in str(exc1.value)

    def test_invalid_smiles_in_orchestration(self):
        orchestrator = ScreeningOrchestrator()
        res = orchestrator.analyze_single_molecule("INVALID_SMILES")
        assert res.is_valid is False
        assert res.status == "INVALID_MOLECULE"
        assert res.error_message is not None
