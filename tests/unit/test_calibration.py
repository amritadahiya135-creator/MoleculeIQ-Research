"""Unit tests for ClinTox probability calibration and statistical validation."""

import os
import json
import pytest
import numpy as np
import joblib

from src.models.clintox import (
    train_clintox_model,
    ClinToxTrainingResult,
    ClinToxEvaluationMetrics,
)
from src.config import CLINTOX_ARTIFACT_DIR


class TestClinToxProbabilityCalibration:
    """Tests for ClinTox probability calibration pipeline."""

    def test_calibrated_model_artifacts_exist_and_load(self):
        meta_path = os.path.join(CLINTOX_ARTIFACT_DIR, "metadata.json")
        model_path = os.path.join(CLINTOX_ARTIFACT_DIR, "model.joblib")
        uncal_path = os.path.join(CLINTOX_ARTIFACT_DIR, "uncalibrated_model.joblib")

        assert os.path.exists(meta_path)
        assert os.path.exists(model_path)
        assert os.path.exists(uncal_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["calibrated"] is True
        assert meta["calibration_method"] == "sigmoid"
        assert "calibration_curve" in meta
        assert meta["metrics"]["brier_score"] <= 0.08
        assert meta["metrics"]["expected_calibration_error"] <= 0.20

    def test_calibrated_predictions_bounded_in_unit_interval(self):
        from src.models.inference import ToxicityInferenceService
        service = ToxicityInferenceService()
        for smi in ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCO"]:
            pred = service.predict_clintox(smi)
            assert 0.0 <= pred.probability <= 1.0
            if pred.uncalibrated_probability is not None:
                assert 0.0 <= pred.uncalibrated_probability <= 1.0

    def test_train_clintox_model_with_and_without_calibration(self, tmp_path):
        out_dir = str(tmp_path / "clintox_test_run")
        # Train with calibration
        res_cal = train_clintox_model(
            output_dir=out_dir,
            n_estimators=50,
            calibration_method="sigmoid",
            random_seed=42,
        )
        assert isinstance(res_cal, ClinToxTrainingResult)
        assert res_cal.metrics.calibrated is True
        assert res_cal.metrics.calibration_method == "sigmoid"
        assert res_cal.metrics.brier_score > 0.0

        # Train without calibration
        res_uncal = train_clintox_model(
            output_dir=out_dir,
            n_estimators=50,
            calibration_method=None,
            random_seed=42,
        )
        assert res_uncal.metrics.calibrated is False
        assert res_uncal.metrics.calibration_method is None
