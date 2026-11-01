"""Unit, integration, and scientific regression tests for Phase B.2 Tox21 Reliability Diagnostics."""

import copy
import json
import math
import os
import shutil
from pathlib import Path
import numpy as np
import pytest

from src.config import TOX21_ARTIFACT_DIR, resolve_path
from src.models.inference import ToxicityInferenceService
from src.models.tox21 import TOX21_ENDPOINTS
from src.models.tox21_reliability_diagnostics import (
    AD_STRATIFICATION_DISCLAIMER,
    MIN_POSITIVES_FOR_STRATUM_SUPPORT,
    PR_AUC_DEFINITION,
    TOX21_ENDPOINT_NAMES,
    build_reliability_series_from_benchmark_dict,
    compute_ad_stratified_diagnostics,
    compute_diagnostic_metrics,
    compute_paired_brier_bootstrap,
    compute_reliability_diagram,
    filter_valid_paired_observations,
    load_endpoint_reliability_diagnostics,
)
from src.ranking.scoring import calculate_prioritization_score
from src.services.orchestration import ScreeningOrchestrator
from src.ui.model_insights import (
    build_ad_stratification_table_rows,
    build_all_endpoints_diagnostic_table_rows,
    build_endpoint_metric_comparison_rows,
    build_reliability_bin_table_rows,
    get_tox21_endpoint_selector_options,
    load_tox21_endpoint_performance_rows,
    render_reliability_diagram_svg,
)
from src.ui.single_molecule import (
    build_single_molecule_export_dict,
    build_tox21_endpoint_table_rows,
)
from src.utils.security import sanitize_dataframe_for_csv


# ---------------------------------------------------------------------------
# 1. Reliability Diagram Computation Tests (Implementation A)
# ---------------------------------------------------------------------------

def test_reliability_diagram_equal_width_bin_assignment_and_boundary_0_and_1():
    """Verify exact 10-bin equal-width boundary assignment, including 0.0 and 1.0."""
    # Exact boundary test points:
    # 0.0 -> bin 0 [0.0, 0.1)
    # 0.1 -> bin 1 [0.1, 0.2)
    # 0.5 -> bin 5 [0.5, 0.6)
    # 0.9 -> bin 9 [0.9, 1.0]
    # 1.0 -> bin 9 [0.9, 1.0] (right-closed final bin)
    y_true = np.array([0, 0, 1, 1, 1], dtype=int)
    y_prob = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=float)

    rel = compute_reliability_diagram(y_true, y_prob, series_name="boundary_check", n_bins=10)
    assert rel.n_valid == 5
    assert rel.n_positive == 3
    assert rel.n_negative == 2
    assert math.isclose(rel.prevalence, 0.6)
    assert sum(rel.bin_counts) == 5

    expected_counts = [1, 1, 0, 0, 0, 1, 0, 0, 0, 2]
    assert rel.bin_counts == expected_counts

    # Bin 0 has p=0.0, y=0
    assert rel.bins[0].count == 1
    assert rel.bins[0].mean_predicted_prob == 0.0
    assert rel.bins[0].observed_positive_fraction == 0.0
    assert rel.bins[0].is_right_closed is False

    # Bin 1 has p=0.1, y=0
    assert rel.bins[1].count == 1
    assert rel.bins[1].mean_predicted_prob == 0.1
    assert rel.bins[1].observed_positive_fraction == 0.0

    # Bin 5 has p=0.5, y=1
    assert rel.bins[5].count == 1
    assert rel.bins[5].mean_predicted_prob == 0.5
    assert rel.bins[5].observed_positive_fraction == 1.0

    # Bin 9 has p=[0.9, 1.0], y=[1, 1] -> mean_p = 0.95, obs = 1.0
    assert rel.bins[9].count == 2
    assert rel.bins[9].is_right_closed is True
    assert rel.bins[9].interval_notation == "[0.90, 1.00]"
    assert math.isclose(rel.bins[9].mean_predicted_prob, 0.95, abs_tol=1e-6)
    assert math.isclose(rel.bins[9].observed_positive_fraction, 1.0, abs_tol=1e-6)

    # Empty bins must be None, never 0.0
    for empty_idx in (2, 3, 4, 6, 7, 8):
        assert rel.bins[empty_idx].is_empty is True
        assert rel.bins[empty_idx].count == 0
        assert rel.bins[empty_idx].mean_predicted_prob is None
        assert rel.bins[empty_idx].observed_positive_fraction is None
        assert rel.bins[empty_idx].abs_calibration_gap is None
        assert rel.mean_predicted_probability[empty_idx] is None
        assert rel.observed_frequency[empty_idx] is None


def test_reliability_diagram_all_predictions_equal():
    """Verify reliability diagram when every prediction has the exact same value."""
    y_true = np.array([0] * 15 + [1] * 5, dtype=int)
    y_prob = np.full(20, 0.25, dtype=float)

    rel = compute_reliability_diagram(y_true, y_prob, series_name="constant")
    assert rel.n_valid == 20
    assert sum(rel.bin_counts) == 20
    assert rel.non_empty_bin_count == 1
    assert rel.bin_counts[2] == 20
    assert rel.bins[2].mean_predicted_prob == 0.25
    assert rel.bins[2].observed_positive_fraction == 0.25
    assert rel.bins[2].abs_calibration_gap == 0.0
    assert rel.bins[2].is_sparse is False

    for b_idx in range(10):
        if b_idx != 2:
            assert rel.bins[b_idx].is_empty is True
            assert rel.bins[b_idx].mean_predicted_prob is None


def test_reliability_diagram_perfectly_calibrated_vs_deliberately_miscalibrated():
    """Verify perfectly calibrated toy data (ECE=0, MACE=0) vs inverted miscalibrated toy data."""
    # Perfectly calibrated:
    # Bin 1 ([0.1, 0.2)): 10 samples with p=0.10, 1 positive (obs=0.10)
    # Bin 8 ([0.8, 0.9)): 10 samples with p=0.80, 8 positives (obs=0.80)
    y_perf = np.array([1] + [0] * 9 + [1] * 8 + [0] * 2, dtype=int)
    p_perf = np.array([0.10] * 10 + [0.80] * 10, dtype=float)

    m_perf = compute_diagnostic_metrics(y_perf, p_perf, method_name="perfect")
    assert m_perf.ece_10bin_weighted == 0.0
    assert m_perf.mace_10bin_max == 0.0
    assert m_perf.reliability_diagram_10bin.bins[1].abs_calibration_gap == 0.0
    assert m_perf.reliability_diagram_10bin.bins[8].abs_calibration_gap == 0.0

    # Deliberately miscalibrated:
    # Bin 0 ([0.0, 0.1)): 10 samples with p=0.05, all 10 are positive (obs=1.0 -> gap=0.95)
    # Bin 9 ([0.9, 1.0]): 10 samples with p=0.90, all 10 are negative (obs=0.0 -> gap=0.90)
    y_mis = np.array([1] * 10 + [0] * 10, dtype=int)
    p_mis = np.array([0.05] * 10 + [0.90] * 10, dtype=float)

    m_mis = compute_diagnostic_metrics(y_mis, p_mis, method_name="miscalibrated")
    # Weighted ECE = 0.5 * 0.95 + 0.5 * 0.90 = 0.925
    assert math.isclose(m_mis.ece_10bin_weighted, 0.925, abs_tol=1e-4)
    # Max bin gap MACE = max(0.95, 0.90) = 0.95
    assert math.isclose(m_mis.mace_10bin_max, 0.95, abs_tol=1e-4)


@pytest.mark.parametrize(
    "bad_y,bad_p",
    [
        ([], []),
        ([0, 1], [0.1, 0.2, 0.3]),
        ([0, 2], [0.1, 0.8]),
        ([0, -1], [0.1, 0.8]),
        ([0, 0.5], [0.1, 0.8]),
        ([0, float("nan")], [0.1, 0.8]),
        ([0, 1], [0.1, float("nan")]),
        ([0, 1], [0.1, float("inf")]),
        ([0, 1], [-0.01, 0.8]),
        ([0, 1], [0.1, 1.01]),
        (True, [0.5]),
        ([0, 1], True),
    ],
)
def test_reliability_and_metrics_reject_invalid_inputs(bad_y, bad_p):
    """Verify invalid labels, out-of-range probabilities, NaNs, and mismatched lengths raise ValueError."""
    with pytest.raises(ValueError):
        compute_reliability_diagram(bad_y, bad_p)
    with pytest.raises(ValueError):
        compute_diagnostic_metrics(bad_y, bad_p)


def test_filter_valid_paired_observations_handles_missing_entries_cleanly():
    """Verify missing (None / NaN) paired observations are counted and excluded without imputation."""
    raw_y = [0, 1, None, 0, float("nan"), 1]
    raw_p = [0.1, 0.9, 0.5, float("nan"), 0.2, 0.8]

    vy, vp, excluded = filter_valid_paired_observations(raw_y, raw_p)
    assert excluded == 3
    np.testing.assert_array_equal(vy, [0, 1, 1])
    np.testing.assert_allclose(vp, [0.1, 0.9, 0.8])


# ---------------------------------------------------------------------------
# 2. Endpoint-Level Metrics & Single-Class Handling Tests (Implementation B)
# ---------------------------------------------------------------------------

def test_metrics_hand_calculated_brier_logloss_ece_mace_roc_and_pr_auc():
    """Verify Brier, Log loss, ECE, MACE, ROC-AUC, and PR-AUC against exact hand calculations."""
    # 4 observations across 2 non-empty bins:
    # Bin 0 [0.0, 0.1): y=[0, 0], p=[0.05, 0.05] -> n=2, mean_p=0.05, obs=0.0, gap=0.05
    # Bin 7 [0.7, 0.8): y=[0, 1], p=[0.75, 0.75] -> n=2, mean_p=0.75, obs=0.5, gap=0.25
    y_true = np.array([0, 0, 0, 1], dtype=int)
    y_prob = np.array([0.05, 0.05, 0.75, 0.75], dtype=float)
    y_prob_orig = y_prob.copy()

    m = compute_diagnostic_metrics(y_true, y_prob, method_name="hand_calc")
    # Ensure input probability array was never mutated
    np.testing.assert_array_equal(y_prob, y_prob_orig)

    # Hand-calculated Brier = ((0.05)^2 + (0.05)^2 + (0.75)^2 + (0.25)^2) / 4
    # = (0.0025 + 0.0025 + 0.5625 + 0.0625) / 4 = 0.6300 / 4 = 0.1575
    assert math.isclose(m.brier_score, 0.1575, abs_tol=1e-4)

    # Hand-calculated Log Loss = -(2*ln(0.95) + ln(0.25) + ln(0.75)) / 4
    expected_ll = -(2.0 * math.log(0.95) + math.log(0.25) + math.log(0.75)) / 4.0
    assert math.isclose(m.log_loss_value, round(expected_ll, 4), abs_tol=1e-4)

    # Hand-calculated 10-bin weighted ECE = (2/4)*0.05 + (2/4)*0.25 = 0.15
    assert math.isclose(m.ece_10bin_weighted, 0.15, abs_tol=1e-4)

    # Hand-calculated 10-bin MACE (max over non-empty bins) = max(0.05, 0.25) = 0.25
    assert math.isclose(m.mace_10bin_max, 0.25, abs_tol=1e-4)

    # Hand-calculated 5-bin MACE (max) = 0.25, and 5-bin unweighted mean = (0.05 + 0.25)/2 = 0.15
    assert math.isclose(m.mace_5bin_max, 0.25, abs_tol=1e-4)
    assert math.isclose(m.mace_5bin_unweighted_mean, 0.15, abs_tol=1e-4)

    # Counts & prevalence
    assert m.n_valid == 4
    assert m.n_positive == 1
    assert m.n_negative == 3
    assert m.prevalence == 0.25
    assert m.is_single_class is False
    assert m.roc_auc is not None and 0.8 <= m.roc_auc <= 1.0
    assert m.pr_auc is not None and 0.4 <= m.pr_auc <= 1.0
    assert m.pr_auc_definition == PR_AUC_DEFINITION


def test_single_class_labels_report_roc_and_pr_auc_as_unavailable_none():
    """Verify single-class label cohorts return None for ROC-AUC and PR-AUC, never fabricated zeros."""
    # All negative (0)
    y_all_neg = np.zeros(12, dtype=int)
    p_test = np.linspace(0.05, 0.60, 12)
    m_neg = compute_diagnostic_metrics(y_all_neg, p_test, method_name="single_neg")

    assert m_neg.is_single_class is True
    assert m_neg.n_positive == 0
    assert m_neg.n_negative == 12
    assert m_neg.roc_auc is None
    assert m_neg.pr_auc is None
    assert m_neg.single_class_reason is not None
    assert "class 1 absent" in m_neg.single_class_reason
    # Brier and ECE remain mathematically defined
    assert m_neg.brier_score is not None

    # All positive (1)
    y_all_pos = np.ones(10, dtype=int)
    m_pos = compute_diagnostic_metrics(y_all_pos, p_test[:10], method_name="single_pos")
    assert m_pos.is_single_class is True
    assert m_pos.n_positive == 10
    assert m_pos.n_negative == 0
    assert m_pos.roc_auc is None
    assert m_pos.pr_auc is None
    assert "class 0 absent" in (m_pos.single_class_reason or "")


# ---------------------------------------------------------------------------
# 3. Paired Bootstrap Analysis Tests (Implementation C)
# ---------------------------------------------------------------------------

def test_paired_bootstrap_reproducibility_sign_convention_and_degenerate_handling():
    """
    Verify paired bootstrap reproducibility for seed=42, sign convention
    `brier_improvement = Brier(raw) - Brier(calibrated) > 0` when calibrated improves,
    and safe handling of degenerate/unavailable inputs.
    """
    y_true = np.array([0] * 80 + [1] * 20, dtype=int)
    # Raw probabilities systematically over-predict (0.40 on negatives, 0.70 on positives)
    p_raw = np.array([0.40] * 80 + [0.70] * 20, dtype=float)
    # Calibrated probabilities are much closer to true labels (0.10 on negatives, 0.80 on positives)
    p_cal = np.array([0.10] * 80 + [0.80] * 20, dtype=float)

    b1 = compute_paired_brier_bootstrap(
        y_true, p_raw, p_cal, endpoint="toy", n_bootstrap=500, random_seed=42
    )
    b2 = compute_paired_brier_bootstrap(
        y_true, p_raw, p_cal, endpoint="toy", n_bootstrap=500, random_seed=42
    )

    assert b1 == b2
    assert b1.status == "AVAILABLE"
    assert b1.n_requested_replicates == 500
    assert b1.n_valid_replicates == 500
    assert b1.brier_improvement is not None
    assert b1.brier_improvement > 0.0
    assert b1.delta_brier_vs_raw == round(-b1.brier_improvement, 6)
    assert b1.brier_improvement_ci_95[0] > 0.0
    assert b1.ci_includes_zero is False

    # Degenerate single-class cohort
    b_single = compute_paired_brier_bootstrap(
        np.zeros(30, dtype=int), p_raw[:30], p_cal[:30], endpoint="single_cls"
    )
    assert b_single.status == "DEGENERATE_SAMPLE"
    assert b_single.brier_improvement is None
    assert b_single.brier_improvement_ci_95 is None

    # Unavailable None input
    b_none = compute_paired_brier_bootstrap(None, None, None, endpoint="missing")
    assert b_none.status == "UNAVAILABLE"
    assert b_none.brier_improvement is None

    # Mismatched length input
    b_mismatch = compute_paired_brier_bootstrap(y_true[:10], p_raw[:12], p_cal[:10], endpoint="mismatch")
    assert b_mismatch.status == "INVALID_INPUT"
    assert b_mismatch.brier_improvement is None


# ---------------------------------------------------------------------------
# 4. Applicability-Domain Stratification Tests (Implementation D)
# ---------------------------------------------------------------------------

def test_ad_stratification_insufficient_support_flagging_and_missing_score_preservation():
    """
    Verify AD stratification flags strata with `< 10` positives as INSUFFICIENT_SUPPORT,
    suppresses ROC-AUC/PR-AUC for sparse strata, and isolates missing (`None`/`NaN`) AD scores.
    """
    rng = np.random.default_rng(42)
    # 90 valid AD scores + 5 missing AD scores = 95 observations
    # Construct 3 tertiles of 30 valid observations each:
    #   Tertile 1 (sim in [0.1, 0.3]): 15 positives (>= 10 -> SUFFICIENT_SUPPORT)
    #   Tertile 2 (sim in [0.4, 0.6]): 12 positives (>= 10 -> SUFFICIENT_SUPPORT)
    #   Tertile 3 (sim in [0.7, 0.9]): 4 positives (< 10 -> INSUFFICIENT_SUPPORT)
    y_t1 = np.array([1] * 15 + [0] * 15, dtype=int)
    y_t2 = np.array([1] * 12 + [0] * 18, dtype=int)
    y_t3 = np.array([1] * 4 + [0] * 26, dtype=int)
    y_miss = np.array([1] * 2 + [0] * 3, dtype=int)

    y_all = np.concatenate([y_t1, y_t2, y_t3, y_miss])
    p_raw = np.clip(y_all * 0.6 + (1 - y_all) * 0.3 + rng.uniform(-0.05, 0.05, size=len(y_all)), 0.01, 0.99)
    p_sig = np.clip(y_all * 0.75 + (1 - y_all) * 0.1 + rng.uniform(-0.03, 0.03, size=len(y_all)), 0.01, 0.99)

    ad_scores = (
        list(np.linspace(0.10, 0.30, 30))
        + list(np.linspace(0.40, 0.60, 30))
        + list(np.linspace(0.70, 0.90, 30))
        + [None, float("nan"), None, None, float("nan")]
    )

    rep = compute_ad_stratified_diagnostics(
        y_true=y_all,
        y_prob_raw=p_raw,
        y_prob_sigmoid=p_sig,
        ad_scores=ad_scores,
        endpoint="NR-PPAR-gamma",
        training_corpus_size=5156,
        custom_quantile_edges=[0.10, 0.30, 0.60, 0.90],
    )

    assert rep.is_exploratory is True
    assert rep.domain_thresholds_validated is False
    assert rep.min_positives_support_rule == MIN_POSITIVES_FOR_STRATUM_SUPPORT
    assert rep.n_total_test == 95
    assert rep.n_valid_ad_scores == 90
    assert rep.n_unavailable_ad_scores == 5
    assert rep.status == "AVAILABLE_WITH_SPARSE_STRATA"

    assert len(rep.strata) == 3
    s1, s2, s3 = rep.strata
    assert s1.n_total == 30 and s1.n_positive == 15
    assert s1.support_status == "SUFFICIENT_SUPPORT"
    assert s1.sigmoid_roc_auc is not None
    assert s1.sigmoid_pr_auc is not None

    assert s2.n_total == 30 and s2.n_positive == 12
    assert s2.support_status == "SUFFICIENT_SUPPORT"
    assert s2.sigmoid_roc_auc is not None

    assert s3.n_total == 30 and s3.n_positive == 4
    assert s3.support_status == "INSUFFICIENT_SUPPORT"
    assert s3.sigmoid_roc_auc is None
    assert s3.sigmoid_pr_auc is None
    assert "4 positive" in (s3.support_warning or "")

    # Missing AD scores are preserved in unavailable_stratum, never forced into T1..T3
    assert rep.unavailable_stratum is not None
    assert rep.unavailable_stratum.n_total == 5
    assert rep.unavailable_stratum.n_positive == 2
    assert rep.unavailable_stratum.support_status == "INSUFFICIENT_SUPPORT"


# ---------------------------------------------------------------------------
# 5. Benchmark Reconciliation & Real 12-Endpoint Verification
# ---------------------------------------------------------------------------

def test_all_12_endpoints_reconcile_with_benchmark_and_retain_ci_caveats():
    """
    Verify that across all 12 Tox21 endpoints:
      1. `load_endpoint_reliability_diagnostics` loads cleanly and reconciles 100% with
         `artifacts/tox21/calibration_benchmark.json`.
      2. Exactly 2 endpoints (`NR-PPAR-gamma`, `SR-ATAD5`) have 95% paired bootstrap Brier
         improvement CIs that include zero, while the other 10 exclude zero.
      3. `NR-PPAR-gamma` Upper Similarity Tertile (`pos = 4 < 10`) is flagged `INSUFFICIENT_SUPPORT`.
    """
    ci_zero_endpoints = []
    for ep in TOX21_ENDPOINTS:
        bundle = load_endpoint_reliability_diagnostics(ep, reconstruct_sample_level=True)
        assert bundle.overall_status == "AVAILABLE"
        assert bundle.calibration_artifact_status == "AVAILABLE"
        assert bundle.sample_level_status == "AVAILABLE_SHA256_VERIFIED"
        assert bundle.reconciled_with_benchmark is True, (ep, bundle.reconciliation_details)

        # Verify all 3 probability series exist and isotonic is marked evaluation-only
        assert set(bundle.series_metrics.keys()) == {"uncalibrated", "sigmoid", "isotonic"}
        assert bundle.series_metrics["sigmoid"].is_deployed_method is True
        assert bundle.series_metrics["isotonic"].is_evaluation_only is True

        # Verify bin counts sum to n_valid for every series
        for s_name, rel_s in bundle.reliability_series.items():
            assert sum(rel_s.bin_counts) == rel_s.n_valid, (ep, s_name)

        # Verify paired bootstrap reconciliation
        boot = bundle.bootstrap_sigmoid
        assert boot.status == "AVAILABLE"
        assert boot.n_valid_replicates == 500
        assert boot.reconciled_with_benchmark is True
        assert boot.brier_improvement > 0.0
        if boot.ci_includes_zero:
            ci_zero_endpoints.append(ep)

    assert set(ci_zero_endpoints) == {"NR-PPAR-gamma", "SR-ATAD5"}

    # Check NR-PPAR-gamma AD stratification specifically flags Q3 Upper Tertile (< 10 positives)
    ppar_bundle = load_endpoint_reliability_diagnostics("NR-PPAR-gamma", reconstruct_sample_level=True)
    assert ppar_bundle.ad_stratification.status == "AVAILABLE_WITH_SPARSE_STRATA"
    q3_stratum = ppar_bundle.ad_stratification.strata[2]
    assert q3_stratum.n_positive < 10
    assert q3_stratum.support_status == "INSUFFICIENT_SUPPORT"
    assert q3_stratum.sigmoid_roc_auc is None
    assert q3_stratum.sigmoid_pr_auc is None


# ---------------------------------------------------------------------------
# 6. UI Helper, Fault Isolation & Regression Tests (Implementation E)
# ---------------------------------------------------------------------------

def test_ui_selector_covers_all_12_endpoints_and_preserves_canonical_ids():
    """Verify the Model Insights endpoint selector includes all 12 canonical Tox21 endpoints."""
    options = get_tox21_endpoint_selector_options()
    assert len(options) == 12
    assert [ep for ep, _ in options] == list(TOX21_ENDPOINTS)
    for ep, label in options:
        assert ep in label
        assert ep in TOX21_ENDPOINT_NAMES


def test_ui_svg_and_table_builders_handle_empty_sparse_and_single_class_safely():
    """
    Verify SVG diagram and table builders never plot empty bins as zero, never fabricate
    ROC-AUC/PR-AUC for single-class endpoints, and return clean unavailable states when empty.
    """
    # 1. Empty series dictionary -> clean unavailable card, no <polyline>
    empty_svg = render_reliability_diagram_svg({}, endpoint="NR-AR")
    assert "unavailable" in empty_svg.lower()
    assert "<polyline" not in empty_svg

    # 2. Single-class series with empty bins -> valid SVG without NaN, table shows "N/A (Empty Bin)"
    y_single = np.zeros(15, dtype=int)
    p_single = np.array([0.05] * 12 + [0.85] * 3, dtype=float)
    rel_single = compute_reliability_diagram(y_single, p_single, series_name="sigmoid")
    svg_str = render_reliability_diagram_svg({"sigmoid": rel_single}, endpoint="NR-AR")
    assert "<svg" in svg_str
    assert "nan" not in svg_str.lower()

    bin_rows = build_reliability_bin_table_rows(rel_single)
    assert len(bin_rows) == 10
    assert bin_rows[0]["Bin Support Status"] == "SUPPORTED (≥ 10 obs)"
    assert bin_rows[8]["Bin Support Status"] == "SPARSE (< 10 obs)"
    assert bin_rows[1]["Mean Predicted Prob"] == "N/A (Empty Bin)"
    assert bin_rows[1]["Observed Positive Fraction"] == "N/A (Empty Bin)"


def test_missing_or_corrupt_endpoint_diagnostics_do_not_break_other_endpoints(tmp_path):
    """
    Verify fault isolation: if one endpoint's calibration artifact is missing or corrupt,
    that endpoint returns an explicit unavailable status while other endpoints load normally.
    """
    prod_dir = resolve_path(TOX21_ARTIFACT_DIR)
    for fname in os.listdir(prod_dir):
        if fname.startswith("NR_AR_") or fname.startswith("NR_AhR_") or fname == "training_fingerprints.npz":
            shutil.copy2(os.path.join(prod_dir, fname), tmp_path / fname)

    # Corrupt NR-AR calibration artifact
    (tmp_path / "NR_AR_calibration.json").write_text("{broken_json", encoding="utf-8")

    b_corrupt = load_endpoint_reliability_diagnostics("NR-AR", artifact_dir=tmp_path)
    assert b_corrupt.overall_status == "CORRUPT_ARTIFACT"
    assert b_corrupt.series_metrics == {}
    assert b_corrupt.bootstrap_sigmoid.status == "UNAVAILABLE"
    assert b_corrupt.ad_stratification.status == "UNAVAILABLE"

    # Meanwhile NR-AhR in the same directory loads and reconciles normally
    b_valid = load_endpoint_reliability_diagnostics("NR-AhR", artifact_dir=tmp_path)
    assert b_valid.overall_status == "AVAILABLE"
    assert b_valid.reconciled_with_benchmark is True

    # Also test benchmark-only fallback when raw CSV is unavailable
    b_bench_only = load_endpoint_reliability_diagnostics(
        "NR-AhR",
        artifact_dir=tmp_path,
        csv_path=str(tmp_path / "nonexistent_tox21.csv"),
        reconstruct_sample_level=True,
    )
    assert b_bench_only.overall_status == "AVAILABLE"
    assert "BENCHMARK_ONLY" in b_bench_only.sample_level_status
    assert b_bench_only.ad_stratification.status == "UNAVAILABLE"

    comp_rows = build_endpoint_metric_comparison_rows(b_valid)
    assert len(comp_rows) == 10
    ad_rows = build_ad_stratification_table_rows(b_valid.ad_stratification)
    assert len(ad_rows) == 3

    # Verify 12-row endpoint-level diagnostic table on both full production dir and partial/corrupt dir
    full_12_rows = build_all_endpoints_diagnostic_table_rows(artifact_dir=prod_dir)
    assert len(full_12_rows) == 12
    assert [r["Endpoint"] for r in full_12_rows] == TOX21_ENDPOINTS
    assert all(r["Recommended Method"] == "sigmoid" for r in full_12_rows)
    assert all("AVAILABLE" in r["Availability & Support Status"] for r in full_12_rows)

    partial_12_rows = build_all_endpoints_diagnostic_table_rows(artifact_dir=str(tmp_path))
    assert len(partial_12_rows) == 12
    row_by_ep = {r["Endpoint"]: r for r in partial_12_rows}
    assert "UNAVAILABLE" in row_by_ep["NR-AR"]["Availability & Support Status"]
    assert row_by_ep["NR-AR"]["Raw Brier"] == "Unavailable"
    assert row_by_ep["NR-AR"]["Sigmoid Brier"] == "Unavailable"
    assert "AVAILABLE" in row_by_ep["NR-AhR"]["Availability & Support Status"]
    assert row_by_ep["NR-AhR"]["Raw Brier"] == "0.0769"
    assert row_by_ep["NR-AhR"]["Sigmoid Brier"] == "0.0653"


def test_production_inference_ranking_and_export_invariance():
    """
    Verify that Phase B.2 introduces zero changes to molecular inference outputs,
    predicted classes, activity flags, applicability-domain scores, candidate ranking,
    or JSON/CSV exports.
    """
    orchestrator = ScreeningOrchestrator()
    aspirin_res = orchestrator.analyze_single_molecule("CC(=O)Oc1ccccc1C(=O)O")
    assert aspirin_res.is_valid is True
    assert aspirin_res.clintox.probability == 0.6659
    assert aspirin_res.clintox.uncalibrated_probability == 1.0
    assert aspirin_res.clintox.applicability_domain_status == "WITHIN_DOMAIN"
    assert aspirin_res.tox21.active_endpoints_count == 0
    assert math.isclose(aspirin_res.prioritization.overall_score, 66.7, abs_tol=1e-2)

    for ep in TOX21_ENDPOINTS:
        ep_res = aspirin_res.tox21.endpoints[ep]
        assert ep_res.predicted_class == int(ep_res.probability >= 0.5)
        assert ep_res.is_active == (ep_res.predicted_class == 1)
        assert ep_res.calibration_status == "AVAILABLE"
        assert ep_res.applicability_domain_status == "UNVALIDATED"

    export_dict = build_single_molecule_export_dict(aspirin_res)
    assert set(export_dict.keys()) == {
        "smiles",
        "formula",
        "descriptors",
        "clintox",
        "tox21",
        "prioritization",
        "timestamp",
    }
    assert len(export_dict["tox21"]["endpoints"]) == 12
    assert len(build_tox21_endpoint_table_rows(aspirin_res.tox21)) == 12

    batch_res = orchestrator.screen_batch(
        smiles_list=["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        names_list=["Aspirin", "Caffeine"],
    )
    assert batch_res.valid_records == 2
    assert math.isclose(batch_res.candidates[0].overall_priority_score, 66.71, abs_tol=1e-2)
    assert math.isclose(batch_res.candidates[1].overall_priority_score, 95.97, abs_tol=1e-2)
    sanitized_df = sanitize_dataframe_for_csv(batch_res.results_dataframe)
    assert len(sanitized_df) == 2
