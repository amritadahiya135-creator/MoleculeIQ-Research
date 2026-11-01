"""Unit, regression, and adversarial tests for Phase B.1 Tox21 Probability Calibration (Module M05)."""

import copy
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
import pytest

from src.models.tox21 import TOX21_ENDPOINTS
from src.models.tox21_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    METHOD_SELECTION_RULE_VERSION,
    Tox21CalibrationService,
    apply_calibrator_params,
    build_3way_disjoint_splits,
    compute_ece_and_curve,
    evaluate_probability_predictions,
    extract_positive_class_probabilities,
    fit_isotonic_calibrator,
    fit_sigmoid_calibrator,
    generate_leakage_safe_oof_predictions,
    predict_with_isotonic_params,
    predict_with_sigmoid_params,
    select_calibration_method_via_3way_disjoint_cv,
    validate_3way_disjoint_fold_indices,
    validate_isotonic_params,
    validate_sigmoid_params,
)
from src.models.inference import ToxicityInferenceService
from src.ranking.scoring import calculate_prioritization_score
from src.services.orchestration import ScreeningOrchestrator
from src.utils.integrity import compute_file_sha256


class _DummyProbaModel:
    """Lightweight mock classifier for positive-class extraction tests."""

    def __init__(self, classes, probs):
        self.classes_ = np.array(classes)
        self._probs = np.array(probs, dtype=np.float64)

    def predict_proba(self, X):
        return self._probs[: len(X)]


def _make_valid_calibration_dict(endpoint: str, model_sha256: str) -> dict:
    """Construct a complete, schema-compliant v1.1.0 calibration artifact dictionary for testing."""
    safe_name = endpoint.replace("-", "_")
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "endpoint": endpoint,
        "created_at": "2026-10-10T08:30:00+00:00",
        "provenance": {
            "base_model_filename": f"{safe_name}_model.joblib",
            "base_model_sha256": model_sha256,
            "dataset_path": "data/raw/tox21.csv",
            "dataset_sha256": "0c5d8aa09dd4bec7858b37d9d26cadf7ce4ab8b6fbc4c290ac648407879b91b8",
            "split_strategy": "stratified_random_80_20",
            "method_selection_protocol": "5_fold_3way_disjoint_base60_cal20_val20",
            "calibration_protocol": "5_fold_stratified_out_of_fold_posthoc",
            "random_seed": 42,
            "positive_class_label": 1,
            "positive_class_index": 1,
            "library_versions": {
                "scikit_learn": "1.9.1",
                "numpy": "2.4.3",
                "pandas": "3.0.6",
            },
        },
        "data_summary": {
            "total_raw_records": 7831,
            "missing_or_invalid_excluded": 573,
            "usable_samples": 7258,
            "positive_count": 308,
            "negative_count": 6950,
            "prevalence": 0.0424,
            "train_count": 5806,
            "train_positive_count": 246,
            "train_negative_count": 5560,
            "test_count": 1452,
            "test_positive_count": 62,
            "test_negative_count": 1390,
        },
        "selection_governance": {
            "recommended_method": "sigmoid",
            "method_selection_rule": METHOD_SELECTION_RULE_VERSION,
            "selection_rationale": "Sigmoid selected via 3-way disjoint CV.",
            "isotonic_selection_eligible": False,
            "default_production_behavior": "raw_uncalibrated_preserved",
            "opt_in_calibrated_available": True,
            "scientific_disclaimer": "Benchmark disclaimer.",
        },
        "calibrator_parameters": {
            "sigmoid": {"method": "sigmoid", "a": -5.5, "b": 4.1},
            "isotonic": {
                "method": "isotonic",
                "X_thresholds": [0.0, 0.5, 1.0],
                "y_thresholds": [0.01, 0.15, 0.65],
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. Toy Metric Hand-Check Verification (Brier, Log Loss, ECE)
# ---------------------------------------------------------------------------

def test_toy_brier_log_loss_and_ece_exact_math():
    """Verify Brier, Log Loss, and 10-bin ECE against hand-calculated values."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.05, 0.85, 0.85])

    expected_brier = 0.0125
    expected_ll = -(2.0 * math.log(0.95) + 2.0 * math.log(0.85)) / 4.0
    expected_ece = 0.10

    ece_10, _, curve_10 = compute_ece_and_curve(y_true, y_prob, n_bins=10)
    assert math.isclose(ece_10, expected_ece, abs_tol=1e-6)
    assert curve_10.n_bins == 10
    assert curve_10.bin_counts[0] == 2
    assert curve_10.bin_counts[8] == 2

    metrics = evaluate_probability_predictions(
        y_true, y_prob, method_name="toy", n_bootstrap=20, random_seed=42
    )
    assert math.isclose(metrics.brier_score, expected_brier, abs_tol=1e-4)
    assert math.isclose(metrics.log_loss_value, expected_ll, abs_tol=1e-4)
    assert math.isclose(metrics.ece_10bin_weighted, expected_ece, abs_tol=1e-4)
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]


# ---------------------------------------------------------------------------
# 2. Positive-Class Probability Extraction
# ---------------------------------------------------------------------------

def test_positive_class_probability_extraction_normal_and_reversed():
    """Verify positive class (label=1) probability extraction for [0, 1] and [1, 0]."""
    X = np.zeros((3, 4))
    raw_matrix = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]

    model_standard = _DummyProbaModel([0, 1], raw_matrix)
    probs_std = extract_positive_class_probabilities(model_standard, X)
    np.testing.assert_allclose(probs_std, [0.2, 0.7, 0.1])

    model_reversed = _DummyProbaModel([1, 0], raw_matrix)
    probs_rev = extract_positive_class_probabilities(model_reversed, X)
    np.testing.assert_allclose(probs_rev, [0.8, 0.3, 0.9])

    model_missing_pos = _DummyProbaModel([0], [[1.0], [1.0], [1.0]])
    with pytest.raises(ValueError, match="Positive class"):
        extract_positive_class_probabilities(model_missing_pos, X)


# ---------------------------------------------------------------------------
# 3. Adversarial 3-Way Disjointness, Test Isolation & Injected Overlap Tests
# ---------------------------------------------------------------------------

def test_3way_disjoint_splits_and_method_selection_determinism():
    """
    Verify that `base_rf_idx`, `cal_fit_idx`, and `val_idx` are pairwise disjoint in every
    selection fold, cover 100% of `y_train`, never touch locked test indices, and are deterministic.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    n_total = 300
    all_global_idx = np.arange(n_total, dtype=np.int64)
    X_all = rng.integers(0, 2, size=(n_total, 24), dtype=np.int8)
    y_all = np.array([0] * 240 + [1] * 60, dtype=int)
    X_all[y_all == 1, :4] = 1

    train_global_idx, test_global_idx, y_train, _ = train_test_split(
        all_global_idx, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train = X_all[train_global_idx]
    assert len(np.intersect1d(train_global_idx, test_global_idx)) == 0

    splits = build_3way_disjoint_splits(y_train, n_splits=5, random_seed=42)
    assert len(splits) == 5

    all_val_seen = []
    for s in splits:
        b_idx = s["base_rf_idx"]
        c_idx = s["cal_fit_idx"]
        v_idx = s["val_idx"]

        # Pairwise disjointness within training partition
        assert len(np.intersect1d(b_idx, c_idx)) == 0
        assert len(np.intersect1d(b_idx, v_idx)) == 0
        assert len(np.intersect1d(c_idx, v_idx)) == 0

        # Complete partition of training set per rotation (60% + 20% + 20% = 100%)
        union_idx = np.concatenate([b_idx, c_idx, v_idx])
        assert len(union_idx) == len(y_train)
        assert len(np.unique(union_idx)) == len(y_train)

        # Zero overlap with locked 20% test partition
        assert len(np.intersect1d(train_global_idx[b_idx], test_global_idx)) == 0
        assert len(np.intersect1d(train_global_idx[c_idx], test_global_idx)) == 0
        assert len(np.intersect1d(train_global_idx[v_idx], test_global_idx)) == 0
        all_val_seen.append(v_idx)

    # Validation subsets across the 5 rotations must partition 100% of D_train
    pooled_val = np.concatenate(all_val_seen)
    assert len(pooled_val) == len(y_train)
    assert len(np.unique(pooled_val)) == len(y_train)

    # Determinism across repeated runs
    sel_1 = select_calibration_method_via_3way_disjoint_cv(
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        n_estimators=15,
        random_seed=42,
        train_global_idx=train_global_idx,
        locked_test_global_idx=test_global_idx,
    )
    sel_2 = select_calibration_method_via_3way_disjoint_cv(
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        n_estimators=15,
        random_seed=42,
        train_global_idx=train_global_idx,
        locked_test_global_idx=test_global_idx,
    )
    assert sel_1["selected_method"] == sel_2["selected_method"]
    assert sel_1["fold_provenance"] == sel_2["fold_provenance"]
    np.testing.assert_array_equal(sel_1["val_raw_probs_60pct"], sel_2["val_raw_probs_60pct"])


def test_adversarial_injected_index_overlaps_are_detected_and_fail():
    """
    Verify that deliberately injecting EVEN ONE overlapping index between:
      (a) `base_rf_idx` and `cal_fit_idx`
      (b) `base_rf_idx` and `val_idx`
      (c) `cal_fit_idx` and `val_idx`
      (d) `train_global_idx` and `locked_test_global_idx`
    immediately fails `validate_3way_disjoint_fold_indices` and `select_calibration_method_via_3way_disjoint_cv`.
    """
    rng = np.random.default_rng(7)
    X_train = rng.integers(0, 2, size=(150, 16), dtype=np.int8)
    y_train = np.array([0] * 120 + [1] * 30, dtype=int)
    valid_splits = build_3way_disjoint_splits(y_train, n_splits=5, random_seed=42)

    # Case A: Inject 1 overlapping index between base_rf_idx and cal_fit_idx
    bad_splits_a = copy.deepcopy(valid_splits)
    bad_splits_a[0]["cal_fit_idx"] = np.append(
        bad_splits_a[0]["cal_fit_idx"], bad_splits_a[0]["base_rf_idx"][0]
    )
    with pytest.raises(ValueError, match="base_rf_idx.*overlaps with.*cal_fit_idx"):
        select_calibration_method_via_3way_disjoint_cv(
            X_train, y_train, n_estimators=10, custom_splits=bad_splits_a
        )

    # Case B: Inject 1 overlapping index between base_rf_idx and val_idx
    bad_splits_b = copy.deepcopy(valid_splits)
    bad_splits_b[0]["val_idx"] = np.append(
        bad_splits_b[0]["val_idx"], bad_splits_b[0]["base_rf_idx"][0]
    )
    with pytest.raises(ValueError, match="base_rf_idx.*overlaps with.*val_idx"):
        select_calibration_method_via_3way_disjoint_cv(
            X_train, y_train, n_estimators=10, custom_splits=bad_splits_b
        )

    # Case C: Inject 1 overlapping index between cal_fit_idx and val_idx
    bad_splits_c = copy.deepcopy(valid_splits)
    bad_splits_c[0]["val_idx"] = np.append(
        bad_splits_c[0]["val_idx"], bad_splits_c[0]["cal_fit_idx"][0]
    )
    with pytest.raises(ValueError, match="cal_fit_idx.*overlaps with.*val_idx"):
        select_calibration_method_via_3way_disjoint_cv(
            X_train, y_train, n_estimators=10, custom_splits=bad_splits_c
        )

    # Case D: Inject 1 overlapping global index between training and locked test set
    train_global = np.arange(150, dtype=np.int64)
    leaked_test_global = np.array([149, 200, 201], dtype=np.int64)  # 149 is in train_global!
    with pytest.raises(ValueError, match="Test-set leakage violation"):
        select_calibration_method_via_3way_disjoint_cv(
            X_train,
            y_train,
            n_estimators=10,
            train_global_idx=train_global,
            locked_test_global_idx=leaked_test_global,
        )


def test_oof_generation_fold_disjointness_and_determinism():
    """Verify 5-fold stratified OOF has zero train/val overlap and is deterministic."""
    rng = np.random.default_rng(42)
    X_train = rng.integers(0, 2, size=(120, 32), dtype=np.int8)
    y_train = np.array([0] * 100 + [1] * 20)

    oof_1, meta_1 = generate_leakage_safe_oof_predictions(
        X_train, y_train, n_splits=5, n_estimators=15, random_seed=42
    )
    oof_2, _ = generate_leakage_safe_oof_predictions(
        X_train, y_train, n_splits=5, n_estimators=15, random_seed=42
    )

    np.testing.assert_array_equal(oof_1, oof_2)
    assert np.all(np.isfinite(oof_1))
    assert np.all((oof_1 >= 0.0) & (oof_1 <= 1.0))
    assert len(meta_1) == 5
    for fit_idx, val_idx in meta_1:
        assert len(np.intersect1d(fit_idx, val_idx)) == 0
    total_val = sum(len(val_idx) for _, val_idx in meta_1)
    assert total_val == len(y_train)


def test_single_class_and_rare_endpoint_rejection():
    """Verify OOF and 3-way split raise ValueError when minority class cannot support n_splits."""
    X_train = np.zeros((30, 16), dtype=np.int8)
    y_single = np.zeros(30, dtype=int)
    with pytest.raises(ValueError, match="Single-class"):
        generate_leakage_safe_oof_predictions(X_train, y_single, n_splits=5, n_estimators=10)
    with pytest.raises(ValueError, match="Single-class"):
        build_3way_disjoint_splits(y_single, n_splits=5)

    # Rare endpoint with only 3 positives cannot support 5-fold stratified splitting
    y_rare = np.array([0] * 27 + [1] * 3, dtype=int)
    with pytest.raises(ValueError, match="insufficient class counts"):
        build_3way_disjoint_splits(y_rare, n_splits=5)


# ---------------------------------------------------------------------------
# 4. Sigmoid & Isotonic Fit, Bounds, Monotonicity, and Invalid Parameter Rejection
# ---------------------------------------------------------------------------

def test_sigmoid_and_isotonic_fit_bounds_and_monotonicity():
    """Verify Sigmoid and Isotonic calibrators produce bounded, monotonic probabilities."""
    rng = np.random.default_rng(123)
    y_true = np.concatenate([np.zeros(280, dtype=int), np.ones(70, dtype=int)])
    raw_probs = np.clip(
        np.concatenate([rng.beta(1.5, 6.0, size=280), rng.beta(5.0, 2.0, size=70)]),
        0.0,
        1.0,
    )

    sig_params = fit_sigmoid_calibrator(raw_probs, y_true)
    iso_params = fit_isotonic_calibrator(raw_probs, y_true)

    grid = np.linspace(0.0, 1.0, 101)
    sig_grid = predict_with_sigmoid_params(grid, sig_params)
    iso_grid = predict_with_isotonic_params(grid, iso_params)

    assert np.all(np.isfinite(sig_grid))
    assert np.all((sig_grid >= 0.0) & (sig_grid <= 1.0))
    assert np.all(np.isfinite(iso_grid))
    assert np.all((iso_grid >= 0.0) & (iso_grid <= 1.0))

    assert np.all(np.diff(iso_grid) >= -1e-12)
    assert np.all(np.diff(sig_grid) >= -1e-12)


@pytest.mark.parametrize(
    "bad_probs",
    [
        np.array([0.2, np.nan, 0.5]),
        np.array([0.2, np.inf, 0.5]),
        np.array([0.2, -np.inf, 0.5]),
        np.array([-0.01, 0.5, 0.8]),
        np.array([0.1, 0.5, 1.05]),
    ],
)
def test_rejection_of_nonfinite_and_out_of_bounds_probabilities(bad_probs):
    """Verify calibration functions reject NaN, inf, -inf, and out-of-range inputs."""
    sig_params = {"method": "sigmoid", "a": -3.0, "b": 1.0}
    iso_params = {"method": "isotonic", "X_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]}

    with pytest.raises(ValueError):
        predict_with_sigmoid_params(bad_probs, sig_params)

    with pytest.raises(ValueError):
        predict_with_isotonic_params(bad_probs, iso_params)


@pytest.mark.parametrize(
    "bad_sig_params",
    [
        {"method": "sigmoid", "a": float("nan"), "b": 1.0},
        {"method": "sigmoid", "a": -3.0, "b": float("inf")},
        {"method": "sigmoid", "a": float("-inf"), "b": 1.0},
        {"method": "sigmoid", "a": True, "b": 1.0},
        {"method": "sigmoid", "a": -3.0},
    ],
)
def test_rejection_of_nan_and_infinite_sigmoid_parameters(bad_sig_params):
    """Verify NaN, infinite, boolean, or missing Sigmoid parameters are rejected."""
    with pytest.raises(ValueError):
        validate_sigmoid_params(bad_sig_params)
    with pytest.raises(ValueError):
        predict_with_sigmoid_params(0.5, bad_sig_params)


@pytest.mark.parametrize(
    "bad_iso_params",
    [
        {"method": "isotonic", "X_thresholds": [], "y_thresholds": []},
        {"method": "isotonic", "X_thresholds": [0.5], "y_thresholds": [0.5]},
        {"method": "isotonic", "X_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.1, 0.5]},
        {"method": "isotonic", "X_thresholds": [0.0, float("nan"), 1.0], "y_thresholds": [0.1, 0.5, 0.9]},
        {"method": "isotonic", "X_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.1, float("inf"), 0.9]},
        {"method": "isotonic", "X_thresholds": [0.0, 0.7, 0.4, 1.0], "y_thresholds": [0.1, 0.3, 0.5, 0.9]},
        {"method": "isotonic", "X_thresholds": [0.0, 0.4, 0.7, 1.0], "y_thresholds": [0.1, 0.6, 0.2, 0.9]},
        {"method": "isotonic", "X_thresholds": [-0.1, 0.5, 1.0], "y_thresholds": [0.1, 0.5, 0.9]},
        {"method": "isotonic", "X_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.1, 0.5, 1.2]},
    ],
)
def test_rejection_of_invalid_isotonic_thresholds(bad_iso_params):
    """Verify invalid, non-finite, non-monotonic, or out-of-range Isotonic thresholds are rejected."""
    with pytest.raises(ValueError):
        validate_isotonic_params(bad_iso_params)
    with pytest.raises(ValueError):
        predict_with_isotonic_params(0.5, bad_iso_params)


def test_isotonic_eligibility_guardrail_for_low_minority_support():
    """Verify Isotonic is excluded from selection when per-fold positive support is below threshold."""
    rng = np.random.default_rng(99)
    X_train = rng.integers(0, 2, size=(300, 16), dtype=np.int8)
    y_train = np.array([0] * 260 + [1] * 40, dtype=int)
    X_train[y_train == 1, :4] = 1

    sel = select_calibration_method_via_3way_disjoint_cv(
        X_train, y_train, n_splits=5, n_estimators=10, random_seed=42
    )
    assert sel["isotonic_selection_eligible"] is False
    assert sel["selected_method"] in ("sigmoid", "uncalibrated")
    assert sel["selected_method"] != "isotonic"


# ---------------------------------------------------------------------------
# 5. Artifact Contract, Schema Version, Corrupt Params & Raw Fallback Isolation
# ---------------------------------------------------------------------------

def test_missing_corrupt_unsupported_schema_and_hash_mismatch_safe_fallbacks(tmp_path):
    """
    Verify that missing, corrupt JSON, unsupported schema versions, hash mismatches,
    non-finite Sigmoid parameters, and invalid Isotonic thresholds fail safely without
    emitting NaN/inf or suppressing valid raw predictions.
    """
    ep = "NR-AR"
    safe_name = "NR_AR"

    # Case 1: Missing artifact
    service_missing = Tox21CalibrationService(artifact_dir=tmp_path)
    res_missing = service_missing.calibrate_probability(ep, 0.5)
    assert res_missing["calibration_status"] == "UNAVAILABLE"
    assert res_missing["calibrated_probability"] is None

    # Create dummy base model file and compute its actual SHA-256
    dummy_model_path = tmp_path / f"{safe_name}_model.joblib"
    dummy_model_path.write_bytes(b"deterministic dummy model content")
    actual_sha = compute_file_sha256(dummy_model_path)
    cal_file = tmp_path / f"{safe_name}_calibration.json"

    # Case 2: Corrupt JSON syntax
    cal_file.write_text("{corrupt json", encoding="utf-8")
    res_corrupt = Tox21CalibrationService(artifact_dir=tmp_path).calibrate_probability(ep, 0.5)
    assert res_corrupt["calibration_status"] == "CORRUPT_ARTIFACT"
    assert res_corrupt["calibrated_probability"] is None

    # Case 3: Unsupported schema version (e.g., "1.0.0" or "999.0.0")
    art_old_schema = _make_valid_calibration_dict(ep, actual_sha)
    art_old_schema["schema_version"] = "1.0.0"
    cal_file.write_text(json.dumps(art_old_schema), encoding="utf-8")
    res_schema = Tox21CalibrationService(artifact_dir=tmp_path).calibrate_probability(ep, 0.5)
    assert res_schema["calibration_status"] == "UNSUPPORTED_SCHEMA"
    assert res_schema["calibrated_probability"] is None

    # Case 4: Base model SHA-256 mismatch
    art_wrong_hash = _make_valid_calibration_dict(ep, "f" * 64)
    cal_file.write_text(json.dumps(art_wrong_hash), encoding="utf-8")
    res_mismatch = Tox21CalibrationService(artifact_dir=tmp_path).calibrate_probability(ep, 0.5)
    assert res_mismatch["calibration_status"] == "MODEL_HASH_MISMATCH"
    assert res_mismatch["calibrated_probability"] is None

    # Case 5: NaN / Infinity in Sigmoid parameters inside JSON
    art_nan_sig = _make_valid_calibration_dict(ep, actual_sha)
    art_nan_sig["calibrator_parameters"]["sigmoid"]["a"] = float("nan")
    cal_file.write_text(json.dumps(art_nan_sig, allow_nan=True), encoding="utf-8")
    res_nan_sig = Tox21CalibrationService(artifact_dir=tmp_path).calibrate_probability(ep, 0.5)
    assert res_nan_sig["calibration_status"] == "CORRUPT_ARTIFACT"
    assert res_nan_sig["calibrated_probability"] is None

    # Case 6: Non-monotonic Isotonic thresholds inside JSON
    art_bad_iso = _make_valid_calibration_dict(ep, actual_sha)
    art_bad_iso["calibrator_parameters"]["isotonic"]["y_thresholds"] = [0.0, 0.8, 0.2]
    cal_file.write_text(json.dumps(art_bad_iso), encoding="utf-8")
    res_bad_iso = Tox21CalibrationService(artifact_dir=tmp_path).calibrate_probability(ep, 0.5)
    assert res_bad_iso["calibration_status"] == "CORRUPT_ARTIFACT"
    assert res_bad_iso["calibrated_probability"] is None


def test_raw_predictions_survive_when_calibration_artifacts_fail(tmp_path):
    """
    Verify that `ToxicityInferenceService.predict_tox21()` preserves all 12 raw probabilities,
    predicted classes, and applicability domain outputs even when calibration artifacts are
    missing, corrupt, schema-mismatched, or hash-mismatched.
    """
    from src.config import TOX21_ARTIFACT_DIR, resolve_path

    prod_tox21_dir = resolve_path(TOX21_ARTIFACT_DIR)
    # Copy deployed models, metadata, and AD corpus to tmp_path without calibration files
    for fname in os.listdir(prod_tox21_dir):
        if fname.endswith("_model.joblib") or fname.endswith("_metadata.json") or fname == "training_fingerprints.npz":
            shutil.copy2(os.path.join(prod_tox21_dir, fname), tmp_path / fname)

    # Inject corrupt calibration file for NR-AR and wrong schema for NR-AhR
    (tmp_path / "NR_AR_calibration.json").write_text("{corrupt", encoding="utf-8")
    ah_sha = compute_file_sha256(tmp_path / "NR_AhR_model.joblib")
    bad_schema_art = _make_valid_calibration_dict("NR-AhR", ah_sha)
    bad_schema_art["schema_version"] = "999.0.0"
    (tmp_path / "NR_AhR_calibration.json").write_text(json.dumps(bad_schema_art), encoding="utf-8")

    prod_service = ToxicityInferenceService()
    isolated_service = ToxicityInferenceService(tox21_artifact_dir=str(tmp_path))

    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    prod_res = prod_service.predict_tox21(smiles)
    iso_res = isolated_service.predict_tox21(smiles)

    assert iso_res.status == "SUCCESS"
    assert iso_res.active_endpoints_count == prod_res.active_endpoints_count

    for ep in TOX21_ENDPOINTS:
        p_prod = prod_res.endpoints[ep]
        p_iso = iso_res.endpoints[ep]
        # Raw probability, class, activity flag, and AD score must be 100% identical!
        assert p_iso.probability == p_prod.probability
        assert p_iso.predicted_class == p_prod.predicted_class
        assert p_iso.is_active == p_prod.is_active
        assert p_iso.applicability_score == p_prod.applicability_score
        assert p_iso.calibrated_probability is None

    assert iso_res.endpoints["NR-AR"].calibration_status == "CORRUPT_ARTIFACT"
    assert iso_res.endpoints["NR-AhR"].calibration_status == "UNSUPPORTED_SCHEMA"
    assert iso_res.endpoints["SR-MMP"].calibration_status == "UNAVAILABLE"


def test_deployed_calibration_artifacts_and_hash_verification():
    """Verify all 12 Tox21 calibration artifacts exist, match v1.1.0 schema, and bind to base models."""
    cal_service = Tox21CalibrationService()
    for ep in TOX21_ENDPOINTS:
        art, status, warn = cal_service.load_endpoint_calibration(ep)
        assert status == "AVAILABLE", f"Failed loading {ep}: {warn}"
        assert art is not None
        assert art["schema_version"] == CALIBRATION_SCHEMA_VERSION
        assert art["endpoint"] == ep
        assert art["selection_governance"]["default_production_behavior"] == "raw_uncalibrated_preserved"
        assert art["selection_governance"]["method_selection_rule"] == METHOD_SELECTION_RULE_VERSION
        assert art["provenance"]["method_selection_protocol"] == "5_fold_3way_disjoint_base60_cal20_val20"
        assert art["provenance"]["train_test_disjoint_verified"] is True
        assert len(art["method_selection_fold_provenance"]) == 5
        for fold_info in art["method_selection_fold_provenance"]:
            assert fold_info["pairwise_disjoint_verified"] is True

        res = cal_service.calibrate_probability(ep, 0.42, method="recommended")
        assert res["calibration_status"] == "AVAILABLE"
        assert res["calibrated_probability"] is not None
        assert math.isfinite(res["calibrated_probability"])
        assert 0.0 <= res["calibrated_probability"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Inference Service Integration, Repeatability & Production Invariance
# ---------------------------------------------------------------------------

def test_inference_service_raw_probability_and_ad_preservation():
    """Verify ToxicityInferenceService preserves raw probabilities, classes, AD, and ClinTox."""
    service = ToxicityInferenceService()
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

    t21_a = service.predict_tox21(smiles)
    t21_b = service.predict_tox21(smiles)

    assert t21_a.status == "SUCCESS"
    assert t21_a.total_endpoints == 12

    for ep in TOX21_ENDPOINTS:
        p1 = t21_a.endpoints[ep]
        p2 = t21_b.endpoints[ep]
        assert p1.probability == p2.probability
        assert p1.predicted_class == p2.predicted_class
        assert p1.is_active == p2.is_active
        assert p1.calibrated_probability == p2.calibrated_probability
        assert p1.calibration_method == p2.calibration_method
        assert p1.calibration_status == "AVAILABLE"
        assert p1.applicability_domain_status == "UNVALIDATED"
        assert p1.applicability_score is not None

        assert p1.predicted_class == int(p1.probability >= 0.5)
        assert p1.is_active == (p1.predicted_class == 1)
        assert p1.calibrated_probability is not None
        assert 0.0 <= p1.calibrated_probability <= 1.0

    clin = service.predict_clintox(smiles)
    assert clin.status == "SUCCESS"
    assert clin.probability == 0.6659
    assert clin.uncalibrated_probability == 1.0
    assert clin.applicability_domain_status == "WITHIN_DOMAIN"


def test_candidate_ranking_invariance():
    """Verify candidate prioritization scores and batch ranking order remain 100% invariant."""
    score_aspirin = calculate_prioritization_score(0.6659, True, 1.0)
    assert math.isclose(score_aspirin.overall_score, 66.71, abs_tol=1e-2)

    orchestrator = ScreeningOrchestrator()
    batch_res = orchestrator.screen_batch(
        smiles_list=["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        names_list=["Aspirin", "Caffeine"],
    )
    assert batch_res.valid_records == 2
    assert [c.name for c in batch_res.candidates] == ["Aspirin", "Caffeine"]
    assert math.isclose(batch_res.candidates[0].overall_priority_score, 66.71, abs_tol=1e-2)


# ---------------------------------------------------------------------------
# 7. QC-2 Remediation Tests (M01, M02, L01, L02, L03, L04)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("non_negative_a", [0.0, -0.0, 1e-8, 2.5, 10.0])
def test_sigmoid_slope_sign_validation_rejects_zero_and_positive_slopes(non_negative_a, tmp_path):
    """
    Verify validate_sigmoid_params and Tox21CalibrationService reject a >= 0.0
    as CORRUPT_ARTIFACT while all 12 deployed calibration artifacts have a < 0.0 (QC2-L01).
    """
    bad_params = {"method": "sigmoid", "a": non_negative_a, "b": 1.0}
    with pytest.raises(ValueError, match="requires a < 0.0"):
        validate_sigmoid_params(bad_params)
    with pytest.raises(ValueError, match="requires a < 0.0"):
        predict_with_sigmoid_params(0.5, bad_params)

    ep = "NR-AR"
    safe_name = "NR_AR"
    dummy_model_path = tmp_path / f"{safe_name}_model.joblib"
    dummy_model_path.write_bytes(b"dummy model bytes for slope test")
    actual_sha = compute_file_sha256(dummy_model_path)

    art = _make_valid_calibration_dict(ep, actual_sha)
    art["calibrator_parameters"]["sigmoid"]["a"] = non_negative_a
    (tmp_path / f"{safe_name}_calibration.json").write_text(json.dumps(art), encoding="utf-8")

    svc = Tox21CalibrationService(artifact_dir=tmp_path)
    res = svc.calibrate_probability(ep, 0.45)
    assert res["calibration_status"] == "CORRUPT_ARTIFACT"
    assert res["calibrated_probability"] is None
    assert "requires a < 0.0" in (res["calibration_warning"] or "")

    # Confirm all 12 deployed artifacts continue to load cleanly and have a < 0.0
    prod_svc = Tox21CalibrationService()
    for prod_ep in TOX21_ENDPOINTS:
        prod_art, prod_status, _ = prod_svc.load_endpoint_calibration(prod_ep)
        assert prod_status == "AVAILABLE"
        assert prod_art["calibrator_parameters"]["sigmoid"]["a"] < 0.0


def test_single_molecule_json_export_and_12_assay_table_contract():
    """
    Verify Single Molecule JSON export includes all 12 Tox21 endpoints with distinct
    raw and calibrated probabilities, preserves valid 0.0 vs None/null, and exposes
    Calibration Status in the 12-assay table (QC2-M02, QC2-L02).
    """
    from dataclasses import replace
    from src.ui.single_molecule import (
        build_single_molecule_export_dict,
        build_tox21_endpoint_table_rows,
    )

    orchestrator = ScreeningOrchestrator()
    res = orchestrator.analyze_single_molecule("CC(=O)Oc1ccccc1C(=O)O")
    export_dict = build_single_molecule_export_dict(res)

    # All existing top-level keys plus tox21 must be present
    expected_top_keys = {
        "smiles",
        "formula",
        "descriptors",
        "clintox",
        "tox21",
        "prioritization",
        "timestamp",
    }
    assert set(export_dict.keys()) == expected_top_keys
    assert export_dict["tox21"] is not None
    assert export_dict["tox21"]["status"] == "SUCCESS"
    assert export_dict["tox21"]["total_endpoints"] == 12
    assert set(export_dict["tox21"]["endpoints"].keys()) == set(TOX21_ENDPOINTS)

    # Verify JSON serialization roundtrip and per-endpoint schema
    serialized_json = json.dumps(export_dict, indent=2)
    parsed_json = json.loads(serialized_json)
    for ep in TOX21_ENDPOINTS:
        ep_rec = parsed_json["tox21"]["endpoints"][ep]
        assert ep_rec["endpoint"] == ep
        assert isinstance(ep_rec["probability"], float)
        assert isinstance(ep_rec["calibrated_probability"], float)
        assert ep_rec["probability"] != ep_rec["calibrated_probability"]
        assert ep_rec["calibration_method"] == "sigmoid"
        assert ep_rec["calibration_status"] == "AVAILABLE"
        assert ep_rec["applicability_domain_status"] == "UNVALIDATED"
        assert isinstance(ep_rec["applicability_score"], float)

    # Verify valid 0.0 calibrated probability vs unavailable (None -> null / "N/A")
    modified_endpoints = dict(res.tox21.endpoints)
    modified_endpoints["NR-AR"] = replace(
        modified_endpoints["NR-AR"],
        calibrated_probability=0.0,
        calibration_method="sigmoid",
        calibration_status="AVAILABLE",
    )
    modified_endpoints["NR-AhR"] = replace(
        modified_endpoints["NR-AhR"],
        calibrated_probability=None,
        calibration_method=None,
        calibration_status="UNAVAILABLE",
        calibration_warning="Missing calibration artifact",
    )
    mod_t21 = replace(res.tox21, endpoints=modified_endpoints)
    mod_res = replace(res, tox21=mod_t21)

    mod_export = json.loads(json.dumps(build_single_molecule_export_dict(mod_res)))
    assert mod_export["tox21"]["endpoints"]["NR-AR"]["calibrated_probability"] == 0.0
    assert mod_export["tox21"]["endpoints"]["NR-AR"]["calibrated_probability"] is not None
    assert mod_export["tox21"]["endpoints"]["NR-AhR"]["calibrated_probability"] is None
    assert mod_export["tox21"]["endpoints"]["NR-AhR"]["calibration_status"] == "UNAVAILABLE"
    assert mod_export["tox21"]["endpoints"]["NR-AhR"]["probability"] == res.tox21.endpoints["NR-AhR"].probability

    # Verify 12-assay table rows include Calibration Status and distinguish "0.0%" from "N/A"
    table_rows = build_tox21_endpoint_table_rows(mod_t21)
    assert len(table_rows) == 12
    row_by_ep = {r["Assay Endpoint"]: r for r in table_rows}
    assert row_by_ep["NR-AR"]["Calibrated Prob."] == "0.0%"
    assert row_by_ep["NR-AR"]["Calibration Status"] == "AVAILABLE"
    assert row_by_ep["NR-AhR"]["Calibrated Prob."] == "N/A"
    assert row_by_ep["NR-AhR"]["Calibration Method"] == "N/A"
    assert row_by_ep["NR-AhR"]["Calibration Status"] == "UNAVAILABLE"


def test_model_insights_tox21_12_endpoint_loader_and_fault_isolation(tmp_path):
    """
    Verify Model Insights Tox21 performance loader returns all 12 distinct endpoints from
    existing per-endpoint metadata artifacts without modifying summary.json, and handles
    missing or malformed metadata files safely without crashing or fabricating metrics (QC2-M01).
    """
    from src.config import TOX21_ARTIFACT_DIR, resolve_path
    from src.ui.model_insights import load_tox21_endpoint_performance_rows

    prod_dir = resolve_path(TOX21_ARTIFACT_DIR)
    rows, skipped = load_tox21_endpoint_performance_rows(prod_dir)
    assert len(rows) == 12
    assert skipped == []
    assert [r["Endpoint"] for r in rows] == list(TOX21_ENDPOINTS)
    assert len({r["Endpoint"] for r in rows}) == 12

    # Copy metadata files to tmp_path and corrupt/remove specific files
    for fname in os.listdir(prod_dir):
        if fname.endswith("_metadata.json"):
            shutil.copy2(os.path.join(prod_dir, fname), tmp_path / fname)

    # 1. Remove NR-AR metadata file
    (tmp_path / "NR_AR_metadata.json").unlink()
    # 2. Corrupt JSON syntax in NR-AhR metadata file
    (tmp_path / "NR_AhR_metadata.json").write_text("{broken json", encoding="utf-8")
    # 3. Inject missing metric field in SR-MMP metadata file
    mmp_meta = json.loads((tmp_path / "SR_MMP_metadata.json").read_text(encoding="utf-8"))
    del mmp_meta["metrics"]["roc_auc"]
    (tmp_path / "SR_MMP_metadata.json").write_text(json.dumps(mmp_meta), encoding="utf-8")

    partial_rows, partial_skipped = load_tox21_endpoint_performance_rows(str(tmp_path))
    assert len(partial_rows) == 9
    assert set(partial_skipped) == {"NR-AR", "NR-AhR", "SR-MMP"}
    loaded_eps = [r["Endpoint"] for r in partial_rows]
    assert len(loaded_eps) == len(set(loaded_eps))
    for bad_ep in ("NR-AR", "NR-AhR", "SR-MMP"):
        assert bad_ep not in loaded_eps


def test_documentation_and_ui_governance_consistency():
    """
    Verify documentation reports the exact 15,577 test observation total (not 15,629)
    and Model Insights UI text distinguishes 3-way disjoint selection from held-out test
    evaluation (QC2-L03, QC2-L04).
    """
    from src.config import resolve_path

    model_card = Path(resolve_path("docs/MODEL_CARD_TOX21.md")).read_text(encoding="utf-8")
    known_limits = Path(resolve_path("docs/KNOWN_LIMITATIONS.md")).read_text(encoding="utf-8")
    model_insights_src = Path(resolve_path("src/ui/model_insights.py")).read_text(encoding="utf-8")

    assert "15,577" in model_card
    assert "15,629" not in model_card
    assert "15,577" in known_limits
    assert "15,629" not in known_limits
    assert "3-Way Disjoint Selection & 5-Fold OOF Fitting" in model_insights_src
    assert "5-fold 3-way disjoint cross-validation" in model_insights_src


def test_platt_sigmoid_monotonicity_and_auc_invariance():
    """
    Phase C.4 Priority One:
    Verify that under the deployed Platt scaling parameterization:
        P(y=1|f) = 1 / (1 + exp(a * f + b))
    a strictly negative slope (a < 0.0) yields a strictly monotonically increasing
    transformation (dP/df > 0), preserving pairwise ranking and leaving ROC-AUC
    and PR-AUC invariant on non-tied samples.

    Also verifies that a positive slope (a > 0.0) would yield a strictly decreasing
    transformation (dP/df < 0) that inverts score ordering (ROC_decreasing = 1 - ROC_raw),
    confirming why validate_sigmoid_params strictly rejects a >= 0.0.

    Does NOT load or modify production model artifacts.
    """
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Deterministic synthetic non-tied test probabilities and binary labels
    rng = np.random.default_rng(20261016)
    n_samples = 120
    # Create strictly non-tied raw probabilities in [0.01, 0.99]
    raw_scores = np.sort(np.linspace(0.02, 0.98, n_samples))
    y_true = rng.binomial(1, 0.25, size=n_samples)

    # 1. Test across representative negative slopes (a < 0.0)
    for test_a in [-6.5, -4.0, -1.5, -0.5]:
        test_b = 3.5
        params = {"method": "sigmoid", "a": test_a, "b": test_b}

        # Predict using verified codebase function
        calibrated_probs = predict_with_sigmoid_params(raw_scores, params)
        assert isinstance(calibrated_probs, np.ndarray)
        assert len(calibrated_probs) == n_samples
        assert np.all((calibrated_probs >= 0.0) & (calibrated_probs <= 1.0))

        # Check strict monotonicity: f_i < f_j ==> P_i < P_j
        raw_diffs = np.diff(raw_scores)
        cal_diffs = np.diff(calibrated_probs)
        assert np.all(raw_diffs > 0.0)
        assert np.all(cal_diffs > 0.0), f"Calibrated probs must be strictly increasing for a={test_a}"

        # Check ranking discrimination invariance
        raw_roc = float(roc_auc_score(y_true, raw_scores))
        cal_roc = float(roc_auc_score(y_true, calibrated_probs))
        assert math.isclose(raw_roc, cal_roc, abs_tol=1e-12), f"ROC-AUC changed for a={test_a}"

        raw_pr = float(average_precision_score(y_true, raw_scores))
        cal_pr = float(average_precision_score(y_true, calibrated_probs))
        assert math.isclose(raw_pr, cal_pr, abs_tol=1e-12), f"PR-AUC changed for a={test_a}"

    # 2. Test that a hypothetical positive slope (a > 0.0) is strictly decreasing
    # (Note: validate_sigmoid_params rejects a >= 0 with ValueError, so we test the math directly)
    hypothetical_pos_a = 2.5
    hypothetical_b = -1.0
    with pytest.raises(ValueError, match="requires a < 0.0"):
        validate_sigmoid_params({"method": "sigmoid", "a": hypothetical_pos_a, "b": hypothetical_b})

    # Under the same formula P = 1 / (1 + exp(a*f + b)) = expit(-(a*f + b)), a > 0 is decreasing
    decreasing_probs = expit(-(hypothetical_pos_a * raw_scores + hypothetical_b))
    dec_diffs = np.diff(decreasing_probs)
    assert np.all(dec_diffs < 0.0), "Positive slope a > 0.0 produces strictly decreasing probabilities"
    dec_roc = float(roc_auc_score(y_true, decreasing_probs))
    assert math.isclose(dec_roc, 1.0 - raw_roc, abs_tol=1e-12), "Decreasing transformation inverts ROC-AUC"
