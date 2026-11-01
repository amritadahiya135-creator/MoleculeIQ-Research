"""Tox21 12-Endpoint Reliability Diagnostics, Bootstrap Uncertainty & AD Stratification (Phase B.2).

Provides a pure, testable, read-only scientific diagnostics layer extending Phase B.1
Tox21 probability calibration without retraining models, regenerating artifacts, or
modifying production inference/ranking behavior:
  - Implementation A: 10-bin equal-width reliability diagrams (`[0.0, 1.0]`, right-closed final bin)
    for Raw RF, Sigmoid-calibrated (selected), and Isotonic-calibrated (evaluation-only comparator).
  - Implementation B: Endpoint-level calibration & discrimination metrics with explicit single-class
    handling (`None` for undefined ROC-AUC / PR-AUC), probability clipping (`1e-15`) restricted to
    log-loss calculation, finite-sample MACE (`max |obs - pred|` across non-empty bins), and
    reconciliation against `artifacts/tox21/calibration_benchmark.json`.
  - Implementation C: Paired bootstrap (`B = 500`, `seed = 42`) confidence intervals for Brier
    improvement (`Brier(raw) - Brier(calibrated)`, positive = improvement) reconciled with benchmark
    `delta_brier_vs_raw` (`Brier(calibrated) - Brier(raw)`).
  - Implementation D: Exploratory applicability-domain (Tanimoto similarity) stratification over
    cryptographically verified held-out test splits, without borrowing ClinTox cutoffs or inventing
    validated thresholds, enforcing `INSUFFICIENT_SUPPORT` when positive count < 10.
"""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.chemistry.validation import validate_smiles
from src.config import TOX21_ARTIFACT_DIR, TOX21_DATA_PATH, resolve_path
from src.models.tox21 import TOX21_ENDPOINTS
from src.models.tox21_applicability import TOX21_AD_DISCLAIMER
from src.models.tox21_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    PROB_CLIP_EPS,
    TOX21_CALIBRATION_DISCLAIMER,
    Tox21CalibrationService,
    _hash_index_array,
    _validate_probability_array,
    extract_positive_class_probabilities,
    predict_with_isotonic_params,
    predict_with_sigmoid_params,
)
from src.utils.integrity import compute_file_sha256

logger = logging.getLogger(__name__)

# Conservative minimum positive sample count required within a stratum for
# positive-class-dependent interpretation (Section 7 reporting rule).
MIN_POSITIVES_FOR_STRATUM_SUPPORT = 10
MIN_BIN_COUNT_FOR_WELL_SUPPORTED = 10
DEFAULT_RELIABILITY_BINS = 10
DEFAULT_BOOTSTRAP_REPLICATES = 500
DEFAULT_RANDOM_SEED = 42

PR_AUC_DEFINITION = (
    "Average Precision (sklearn.metrics.average_precision_score): step-wise precision-recall "
    "integral (sum_n (R_n - R_{n-1}) * P_n) without linear-trapezoidal interpolation optimism."
)

ECE_DEFINITION = (
    "10-bin sample-weighted Expected Calibration Error over equal-width bins on [0.0, 1.0] "
    "(bins 0..8 half-open [l, u), bin 9 closed [0.9, 1.0]): sum_{b: n_b > 0} (n_b / N) * |obs_b - pred_b|."
)

MACE_DEFINITION = (
    "Finite-sample Maximum Calibration Error across eligible non-empty equal-width bins: "
    "max_{b: n_b > 0} |obs_b - pred_b|. Because MACE takes the unweighted maximum across non-empty "
    "bins, a single sparse high-probability bin with 1-5 observations can dominate the value."
)

AD_STRATIFICATION_DISCLAIMER = (
    "EXPLORATORY ANALYSIS ONLY: Tox21 applicability-domain strata are empirical quantile tertiles "
    "(Q1: 0-33.3%, Q2: 33.3-66.7%, Q3: 66.7-100%) of maximum Morgan fingerprint Tanimoto similarity "
    "to each endpoint's training set. They do not use ClinTox cutoffs, do not represent validated "
    "chemical-domain thresholds, and high Tanimoto similarity does not establish biological or clinical safety."
)

TOX21_ENDPOINT_NAMES: Dict[str, Dict[str, str]] = {
    "NR-AR": {
        "name": "Androgen Receptor (Full-Length)",
        "family": "Nuclear Receptor (NR)",
        "selector_label": "NR-AR — Androgen Receptor (Full-Length)",
    },
    "NR-AR-LBD": {
        "name": "Androgen Receptor (Ligand-Binding Domain)",
        "family": "Nuclear Receptor (NR)",
        "selector_label": "NR-AR-LBD — Androgen Receptor (LBD)",
    },
    "NR-AhR": {
        "name": "Aryl Hydrocarbon Receptor",
        "family": "Nuclear Receptor (NR)",
        "selector_label": "NR-AhR — Aryl Hydrocarbon Receptor",
    },
    "NR-Aromatase": {
        "name": "Aromatase (CYP19A1) Inhibition",
        "family": "Nuclear Receptor (NR)",
        "selector_label": "NR-Aromatase — Aromatase (CYP19A1)",
    },
    "NR-ER": {
        "name": "Estrogen Receptor Alpha (Full-Length)",
        "family": "Nuclear Receptor (NR)",
        "selector_label": "NR-ER — Estrogen Receptor Alpha (Full-Length)",
    },
    "NR-ER-LBD": {
        "name": "Estrogen Receptor Alpha (Ligand-Binding Domain)",
        "family": "Nuclear Receptor (NR)",
        "selector_label": "NR-ER-LBD — Estrogen Receptor Alpha (LBD)",
    },
    "NR-PPAR-gamma": {
        "name": "Peroxisome Proliferator-Activated Receptor Gamma",
        "family": "Nuclear Receptor (NR)",
        "selector_label": "NR-PPAR-gamma — PPAR-gamma Receptor",
    },
    "SR-ARE": {
        "name": "Antioxidant Response Element (Nrf2/ARE)",
        "family": "Stress Response (SR)",
        "selector_label": "SR-ARE — Antioxidant Response Element",
    },
    "SR-ATAD5": {
        "name": "ATAD5 Genotoxicity / DNA Replication Stress",
        "family": "Stress Response (SR)",
        "selector_label": "SR-ATAD5 — ATAD5 Genotoxic Stress",
    },
    "SR-HSE": {
        "name": "Heat Shock Factor Response Element",
        "family": "Stress Response (SR)",
        "selector_label": "SR-HSE — Heat Shock Response Element",
    },
    "SR-MMP": {
        "name": "Mitochondrial Membrane Potential Disruption",
        "family": "Stress Response (SR)",
        "selector_label": "SR-MMP — Mitochondrial Membrane Potential",
    },
    "SR-p53": {
        "name": "p53 Tumor Suppressor DNA-Damage Pathway",
        "family": "Stress Response (SR)",
        "selector_label": "SR-p53 — p53 DNA-Damage Pathway",
    },
}


# ---------------------------------------------------------------------------
# Dataclasses & Structured Return Schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReliabilityBin:
    """Single probability interval bin in a reliability diagram."""
    bin_index: int
    lower_edge: float
    upper_edge: float
    bin_center: float
    interval_notation: str
    is_right_closed: bool
    count: int
    positive_count: int
    negative_count: int
    mean_predicted_prob: Optional[float]
    observed_positive_fraction: Optional[float]
    abs_calibration_gap: Optional[float]
    is_empty: bool
    is_sparse: bool


@dataclass(frozen=True)
class ReliabilityDiagramSeries:
    """Complete reliability diagram data for a single endpoint and probability series."""
    series_name: str
    display_label: str
    is_evaluation_only: bool
    n_valid: int
    n_positive: int
    n_negative: int
    prevalence: float
    n_bins: int
    bin_strategy: str
    bin_edges: List[float]
    bin_centers: List[float]
    bin_counts: List[int]
    mean_predicted_probability: List[Optional[float]]
    observed_frequency: List[Optional[float]]
    bins: List[ReliabilityBin]
    non_empty_bin_count: int
    sparse_bin_count: int


@dataclass(frozen=True)
class DiagnosticMethodMetrics:
    """Comprehensive calibration and discrimination metrics for a single probability series."""
    method: str
    display_label: str
    calibration_status: str
    is_deployed_method: bool
    is_evaluation_only: bool
    n_valid: int
    n_positive: int
    n_negative: int
    prevalence: float
    is_single_class: bool
    single_class_reason: Optional[str]
    brier_score: Optional[float]
    log_loss_value: Optional[float]
    ece_10bin_weighted: Optional[float]
    mace_10bin_max: Optional[float]
    mace_5bin_max: Optional[float]
    mace_5bin_unweighted_mean: Optional[float]
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    pr_auc_definition: str
    ece_definition: str
    mace_definition: str
    reliability_diagram_10bin: Optional[ReliabilityDiagramSeries]
    mace_10bin_unweighted_mean: Optional[float] = None


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired bootstrap analysis for Brier score improvement (`Brier(raw) - Brier(calibrated)`)."""
    endpoint: str
    comparator_method: str
    status: str
    status_message: Optional[str]
    n_samples: int
    n_positive: int
    n_negative: int
    n_requested_replicates: int
    n_valid_replicates: int
    n_two_class_replicates: int
    n_degenerate_replicates: int
    random_seed: int
    raw_brier: Optional[float]
    raw_brier_ci_95: Optional[List[float]]
    calibrated_brier: Optional[float]
    calibrated_brier_ci_95: Optional[List[float]]
    # Positive values of brier_improvement indicate lower (improved) calibrated Brier score:
    # brier_improvement = Brier(raw) - Brier(calibrated)
    brier_improvement: Optional[float]
    brier_improvement_ci_95: Optional[List[float]]
    # Benchmark sign convention for direct reconciliation:
    # delta_brier_vs_raw = Brier(calibrated) - Brier(raw) = -brier_improvement
    delta_brier_vs_raw: Optional[float]
    delta_brier_ci_95: Optional[List[float]]
    ci_includes_zero: Optional[bool]
    reconciled_with_benchmark: Optional[bool]
    interpretation_note: str


@dataclass(frozen=True)
class ADStratumDiagnostics:
    """Calibration & support diagnostics for a single applicability-domain similarity stratum."""
    stratum_id: str
    stratum_label: str
    similarity_min: Optional[float]
    similarity_max: Optional[float]
    n_total: int
    n_positive: int
    n_negative: int
    prevalence: Optional[float]
    support_status: str
    support_warning: Optional[str]
    raw_brier: Optional[float]
    sigmoid_brier: Optional[float]
    brier_improvement: Optional[float]
    raw_log_loss: Optional[float]
    sigmoid_log_loss: Optional[float]
    raw_ece_10bin: Optional[float]
    sigmoid_ece_10bin: Optional[float]
    raw_mace_10bin: Optional[float]
    sigmoid_mace_10bin: Optional[float]
    raw_roc_auc: Optional[float]
    sigmoid_roc_auc: Optional[float]
    raw_pr_auc: Optional[float]
    sigmoid_pr_auc: Optional[float]


@dataclass(frozen=True)
class EndpointADStratificationReport:
    """Exploratory applicability-domain stratification report for a single Tox21 endpoint."""
    endpoint: str
    status: str
    status_message: Optional[str]
    is_exploratory: bool
    stratification_method: str
    domain_thresholds_validated: bool
    min_positives_support_rule: int
    n_total_test: int
    n_valid_ad_scores: int
    n_unavailable_ad_scores: int
    training_corpus_size: int
    quantile_edges: List[float]
    strata: List[ADStratumDiagnostics]
    unavailable_stratum: Optional[ADStratumDiagnostics]
    scientific_disclaimer: str


@dataclass(frozen=True)
class EndpointReliabilityDiagnosticBundle:
    """Complete Phase B.2 diagnostic bundle for a single Tox21 endpoint."""
    endpoint: str
    display_name: str
    pathway_family: str
    selector_label: str
    overall_status: str
    status_message: Optional[str]
    calibration_artifact_status: str
    benchmark_status: str
    sample_level_status: str
    reconciled_with_benchmark: bool
    reconciliation_details: Dict[str, bool]
    recommended_method: str
    selection_rationale: str
    isotonic_selection_eligible: bool
    data_summary: Dict[str, Any]
    score_distribution_comparison: Dict[str, Any]
    series_metrics: Dict[str, DiagnosticMethodMetrics]
    reliability_series: Dict[str, ReliabilityDiagramSeries]
    bootstrap_sigmoid: PairedBootstrapResult
    bootstrap_isotonic: Optional[PairedBootstrapResult]
    ad_stratification: EndpointADStratificationReport
    scientific_disclaimer: str


# ---------------------------------------------------------------------------
# Input Validation Helpers
# ---------------------------------------------------------------------------

def _validate_binary_labels(y_true: Any, context: str = "y_true") -> np.ndarray:
    """
    Validate that `y_true` is a non-empty 1D array of finite binary integers in `{0, 1}`.
    Rejects booleans, NaNs, infinities, strings, and non-binary numbers.
    """
    if isinstance(y_true, (bool, np.bool_)):
        raise ValueError(f"Boolean scalar is not a valid {context} array.")
    arr = np.asarray(y_true)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Non-numeric or boolean dtype ({arr.dtype}) encountered in {context}.")
    if arr.ndim != 1:
        raise ValueError(f"{context} must be 1-dimensional (got shape {arr.shape}).")
    if arr.size == 0:
        raise ValueError(f"Empty array encountered in {context}.")
    f_arr = arr.astype(np.float64)
    if not np.all(np.isfinite(f_arr)):
        raise ValueError(f"Non-finite (NaN or inf) values encountered in {context}.")
    if not np.all((f_arr == 0.0) | (f_arr == 1.0)):
        bad_vals = np.unique(f_arr[(f_arr != 0.0) & (f_arr != 1.0)])
        raise ValueError(f"Invalid non-binary label values {bad_vals.tolist()} in {context}; expected {0, 1}.")
    return f_arr.astype(np.int64)


def _validate_1d_probabilities(y_prob: Any, context: str = "y_prob") -> np.ndarray:
    """Validate that `y_prob` is a non-empty 1D float64 array strictly in `[0.0, 1.0]`."""
    p_arr = _validate_probability_array(y_prob, context=context)
    if p_arr.ndim != 1:
        raise ValueError(f"{context} must be 1-dimensional (got shape {p_arr.shape}).")
    return p_arr


def filter_valid_paired_observations(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Explicitly separate missing (`None` / `NaN`) observations from valid paired `(y_true, y_prob)`
    records without fabricating or imputing values.

    Returns
    -------
    valid_y : np.ndarray (int64)
    valid_p : np.ndarray (float64)
    excluded_count : int
    """
    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_prob)={len(y_prob)}.")

    valid_y_list: List[int] = []
    valid_p_list: List[float] = []
    excluded = 0

    for y_val, p_val in zip(y_true, y_prob):
        if y_val is None or p_val is None:
            excluded += 1
            continue
        if isinstance(y_val, (bool, np.bool_)) or isinstance(p_val, (bool, np.bool_)):
            raise ValueError("Boolean values are not valid labels or probabilities.")
        try:
            yf = float(y_val)
            pf = float(p_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric observation encountered: {exc}") from exc

        if math.isnan(yf) or math.isnan(pf):
            excluded += 1
            continue
        if not (math.isfinite(yf) and math.isfinite(pf)):
            raise ValueError("Infinite values are not valid labels or probabilities.")
        if yf not in (0.0, 1.0):
            raise ValueError(f"Invalid binary label {y_val}; expected 0 or 1.")
        if pf < 0.0 or pf > 1.0:
            raise ValueError(f"Probability {p_val} out of bounds [0.0, 1.0].")

        valid_y_list.append(int(yf))
        valid_p_list.append(float(pf))

    if not valid_y_list:
        raise ValueError("No valid paired observations remain after excluding missing entries.")

    return np.asarray(valid_y_list, dtype=np.int64), np.asarray(valid_p_list, dtype=np.float64), excluded


# ---------------------------------------------------------------------------
# Implementation A — Reliability Diagrams
# ---------------------------------------------------------------------------

def compute_reliability_diagram(
    y_true: Any,
    y_prob: Any,
    series_name: str = "uncalibrated",
    n_bins: int = DEFAULT_RELIABILITY_BINS,
    display_label: Optional[str] = None,
    is_evaluation_only: bool = False,
) -> ReliabilityDiagramSeries:
    """
    Compute equal-width reliability diagram statistics over `[0.0, 1.0]`.

    Boundary Rule (`equal_width_closed_right_final`):
      - `n_bins` equal-width bins with edges `np.linspace(0.0, 1.0, n_bins + 1)`.
      - Bins `0 .. n_bins - 2` are half-open `[edge_b, edge_{b+1})`.
      - Final bin `n_bins - 1` is closed `[edge_{n_bins - 1}, 1.0]` so that `p == 0.0` falls
        into bin `0` and `p == 1.0` falls into bin `n_bins - 1`.
      - Empty bins (`count == 0`) have `mean_predicted_prob = None`,
        `observed_positive_fraction = None`, and `abs_calibration_gap = None` without division by zero.
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError(f"n_bins must be an integer >= 2 (got {n_bins}).")

    y_arr = _validate_binary_labels(y_true, context=f"{series_name} y_true")
    p_arr = _validate_1d_probabilities(y_prob, context=f"{series_name} y_prob")

    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Length mismatch in compute_reliability_diagram for '{series_name}': "
            f"len(y_true)={len(y_arr)} != len(y_prob)={len(p_arr)}."
        )

    n_valid = int(len(y_arr))
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    prevalence = round(float(n_pos / n_valid), 4)

    raw_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = [round(float(e), 4) for e in raw_edges]
    bin_centers = [round(float((raw_edges[i] + raw_edges[i + 1]) / 2.0), 4) for i in range(n_bins)]

    # np.digitize with internal edges raw_edges[1:-1] and right=False assigns:
    #   [0.0, e_1) -> 0, [e_1, e_2) -> 1, ..., [e_{n_bins-1}, 1.0] -> n_bins - 1
    bin_indices = np.digitize(p_arr, raw_edges[1:-1], right=False)

    bins_list: List[ReliabilityBin] = []
    bin_counts: List[int] = []
    mean_preds: List[Optional[float]] = []
    obs_freqs: List[Optional[float]] = []

    non_empty_count = 0
    sparse_count = 0

    for b in range(n_bins):
        mask = bin_indices == b
        cnt = int(np.sum(mask))
        pos_cnt = int(np.sum(y_arr[mask] == 1)) if cnt > 0 else 0
        neg_cnt = int(cnt - pos_cnt)
        low_e = bin_edges[b]
        high_e = bin_edges[b + 1]
        is_right_closed = b == (n_bins - 1)
        notation = f"[{low_e:.2f}, {high_e:.2f}]" if is_right_closed else f"[{low_e:.2f}, {high_e:.2f})"

        bin_counts.append(cnt)
        if cnt > 0:
            non_empty_count += 1
            is_sparse = cnt < MIN_BIN_COUNT_FOR_WELL_SUPPORTED
            if is_sparse:
                sparse_count += 1
            p_bar = float(np.mean(p_arr[mask]))
            o_bar = float(np.mean(y_arr[mask]))
            gap = abs(o_bar - p_bar)
            mean_p_round = round(p_bar, 4)
            obs_f_round = round(o_bar, 4)
            gap_round = round(gap, 4)
            mean_preds.append(mean_p_round)
            obs_freqs.append(obs_f_round)
            bins_list.append(
                ReliabilityBin(
                    bin_index=b,
                    lower_edge=low_e,
                    upper_edge=high_e,
                    bin_center=bin_centers[b],
                    interval_notation=notation,
                    is_right_closed=is_right_closed,
                    count=cnt,
                    positive_count=pos_cnt,
                    negative_count=neg_cnt,
                    mean_predicted_prob=mean_p_round,
                    observed_positive_fraction=obs_f_round,
                    abs_calibration_gap=gap_round,
                    is_empty=False,
                    is_sparse=is_sparse,
                )
            )
        else:
            mean_preds.append(None)
            obs_freqs.append(None)
            bins_list.append(
                ReliabilityBin(
                    bin_index=b,
                    lower_edge=low_e,
                    upper_edge=high_e,
                    bin_center=bin_centers[b],
                    interval_notation=notation,
                    is_right_closed=is_right_closed,
                    count=0,
                    positive_count=0,
                    negative_count=0,
                    mean_predicted_prob=None,
                    observed_positive_fraction=None,
                    abs_calibration_gap=None,
                    is_empty=True,
                    is_sparse=False,
                )
            )

    if sum(bin_counts) != n_valid:
        raise RuntimeError(
            f"Bin count conservation failure: sum(bin_counts)={sum(bin_counts)} != n_valid={n_valid}."
        )

    default_labels = {
        "uncalibrated": "Raw Random Forest (Uncalibrated)",
        "sigmoid": "Sigmoid / Platt Scaling (Selected Calibrator)",
        "isotonic": "Isotonic Regression (Evaluation-Only Comparator)",
    }
    resolved_label = display_label or default_labels.get(series_name, series_name)

    return ReliabilityDiagramSeries(
        series_name=series_name,
        display_label=resolved_label,
        is_evaluation_only=bool(is_evaluation_only),
        n_valid=n_valid,
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=prevalence,
        n_bins=n_bins,
        bin_strategy="equal_width_closed_right_final",
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_counts=bin_counts,
        mean_predicted_probability=mean_preds,
        observed_frequency=obs_freqs,
        bins=bins_list,
        non_empty_bin_count=non_empty_count,
        sparse_bin_count=sparse_count,
    )


def build_reliability_series_from_benchmark_dict(
    curve_dict: Dict[str, Any],
    n_valid: int,
    n_positive: int,
    n_negative: int,
    series_name: str,
    display_label: Optional[str] = None,
    is_evaluation_only: bool = False,
) -> ReliabilityDiagramSeries:
    """
    Construct a `ReliabilityDiagramSeries` from a validated `calibration_curve_10bin`
    dictionary inside `calibration_benchmark.json` when only benchmark JSON is loaded.
    """
    n_bins = int(curve_dict.get("n_bins", 10))
    bin_edges = [round(float(x), 4) for x in curve_dict["bin_edges"]]
    bin_centers = [round(float(x), 4) for x in curve_dict["bin_centers"]]
    bin_counts = [int(c) for c in curve_dict["bin_counts"]]
    mean_preds = [
        round(float(p), 4) if p is not None else None
        for p in curve_dict["mean_predicted_probability"]
    ]
    obs_freqs = [
        round(float(o), 4) if o is not None else None
        for o in curve_dict["observed_frequency"]
    ]

    if len(bin_counts) != n_bins or len(mean_preds) != n_bins or len(obs_freqs) != n_bins:
        raise ValueError(f"Malformed calibration_curve_10bin lengths for '{series_name}'.")
    if sum(bin_counts) != int(n_valid):
        raise ValueError(
            f"Benchmark bin_counts sum ({sum(bin_counts)}) does not match test_count ({n_valid})."
        )

    bins_list: List[ReliabilityBin] = []
    non_empty_count = 0
    sparse_count = 0

    for b in range(n_bins):
        cnt = bin_counts[b]
        low_e = bin_edges[b]
        high_e = bin_edges[b + 1]
        is_right_closed = b == (n_bins - 1)
        notation = f"[{low_e:.2f}, {high_e:.2f}]" if is_right_closed else f"[{low_e:.2f}, {high_e:.2f})"
        p_val = mean_preds[b]
        o_val = obs_freqs[b]

        if cnt > 0 and p_val is not None and o_val is not None:
            non_empty_count += 1
            is_sparse = cnt < MIN_BIN_COUNT_FOR_WELL_SUPPORTED
            if is_sparse:
                sparse_count += 1
            pos_est = int(round(cnt * o_val))
            neg_est = max(0, cnt - pos_est)
            gap = round(abs(o_val - p_val), 4)
            bins_list.append(
                ReliabilityBin(
                    bin_index=b,
                    lower_edge=low_e,
                    upper_edge=high_e,
                    bin_center=bin_centers[b],
                    interval_notation=notation,
                    is_right_closed=is_right_closed,
                    count=cnt,
                    positive_count=pos_est,
                    negative_count=neg_est,
                    mean_predicted_prob=p_val,
                    observed_positive_fraction=o_val,
                    abs_calibration_gap=gap,
                    is_empty=False,
                    is_sparse=is_sparse,
                )
            )
        else:
            bins_list.append(
                ReliabilityBin(
                    bin_index=b,
                    lower_edge=low_e,
                    upper_edge=high_e,
                    bin_center=bin_centers[b],
                    interval_notation=notation,
                    is_right_closed=is_right_closed,
                    count=0,
                    positive_count=0,
                    negative_count=0,
                    mean_predicted_prob=None,
                    observed_positive_fraction=None,
                    abs_calibration_gap=None,
                    is_empty=True,
                    is_sparse=False,
                )
            )

    default_labels = {
        "uncalibrated": "Raw Random Forest (Uncalibrated)",
        "sigmoid": "Sigmoid / Platt Scaling (Selected Calibrator)",
        "isotonic": "Isotonic Regression (Evaluation-Only Comparator)",
    }
    resolved_label = display_label or default_labels.get(series_name, series_name)
    prevalence = round(float(n_positive / n_valid), 4) if n_valid > 0 else 0.0

    return ReliabilityDiagramSeries(
        series_name=series_name,
        display_label=resolved_label,
        is_evaluation_only=bool(is_evaluation_only),
        n_valid=int(n_valid),
        n_positive=int(n_positive),
        n_negative=int(n_negative),
        prevalence=prevalence,
        n_bins=n_bins,
        bin_strategy=str(curve_dict.get("bin_strategy", "equal_width_closed_right_final")),
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_counts=bin_counts,
        mean_predicted_probability=mean_preds,
        observed_frequency=obs_freqs,
        bins=bins_list,
        non_empty_bin_count=non_empty_count,
        sparse_bin_count=sparse_count,
    )


# ---------------------------------------------------------------------------
# Implementation B — Endpoint-Level Metrics
# ---------------------------------------------------------------------------

def _compute_ece_mace_from_arrays(
    y_arr: np.ndarray,
    p_arr: np.ndarray,
    n_bins: int,
) -> Tuple[float, float, float]:
    """
    Compute:
      1. `ece_weighted`: sample-weighted Expected Calibration Error `sum_{b: n_b>0} (n_b/N) * |o_b - p_b|`
      2. `mace_max`: Maximum Calibration Error `max_{b: n_b>0} |o_b - p_b|` among non-empty bins
      3. `ace_unweighted_mean`: unweighted mean absolute bin error `mean_{b: n_b>0} |o_b - p_b|`
    using exact unrounded per-bin means before rounding the final summary statistic.
    """
    n_total = len(y_arr)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p_arr, bin_edges[1:-1], right=False)

    weighted_sum = 0.0
    non_empty_diffs: List[float] = []

    for b in range(n_bins):
        mask = bin_indices == b
        cnt = int(np.sum(mask))
        if cnt > 0:
            o_bar = float(np.mean(y_arr[mask]))
            p_bar = float(np.mean(p_arr[mask]))
            diff = abs(o_bar - p_bar)
            weighted_sum += (cnt / n_total) * diff
            non_empty_diffs.append(diff)

    ece_val = float(weighted_sum)
    mace_max_val = float(max(non_empty_diffs)) if non_empty_diffs else 0.0
    ace_mean_val = float(np.mean(non_empty_diffs)) if non_empty_diffs else 0.0
    return ece_val, mace_max_val, ace_mean_val


def compute_diagnostic_metrics(
    y_true: Any,
    y_prob: Any,
    method_name: str = "uncalibrated",
    calibration_status: str = "AVAILABLE",
    is_deployed_method: bool = False,
    is_evaluation_only: bool = False,
    display_label: Optional[str] = None,
) -> DiagnosticMethodMetrics:
    """
    Compute endpoint-level probability reliability and discrimination metrics.

    Statistical guarantees:
      1. `log_loss_value` clips probabilities to `[1e-15, 1 - 1e-15]` strictly on a local copy
         for metric computation only; input `y_prob` is never mutated.
      2. `ece_10bin_weighted` uses 10 equal-width bins over `[0.0, 1.0]` (`equal_width_closed_right_final`)
         weighted by bin sample proportion `n_b / N`.
      3. `mace_10bin_max` and `mace_5bin_max` compute the maximum absolute difference
         `max_{b: n_b > 0} |obs_b - pred_b|` across non-empty bins. `mace_5bin_unweighted_mean`
         is also retained for exact cross-reference with the legacy field in `calibration_benchmark.json`.
      4. `pr_auc` is explicitly defined as `sklearn.metrics.average_precision_score`.
      5. Single-class labels (`n_positive == 0` or `n_negative == 0`) return `roc_auc = None` and
         `pr_auc = None` with an explicit `single_class_reason`, never fabricated zeros.
    """
    y_arr = _validate_binary_labels(y_true, context=f"{method_name} y_true")
    p_arr = _validate_1d_probabilities(y_prob, context=f"{method_name} y_prob")

    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Length mismatch in compute_diagnostic_metrics for '{method_name}': "
            f"len(y_true)={len(y_arr)} != len(y_prob)={len(p_arr)}."
        )

    n_valid = int(len(y_arr))
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    prevalence = round(float(n_pos / n_valid), 4)

    # Local copy for log_loss clipping; p_arr remains completely untouched
    p_clipped = np.clip(p_arr.copy(), PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

    brier = round(float(brier_score_loss(y_arr, p_arr)), 4)
    ll = round(float(log_loss(y_arr, p_clipped, labels=[0, 1])), 4)

    ece_10, mace_10_max, mace_10_mean = _compute_ece_mace_from_arrays(y_arr, p_arr, n_bins=10)
    _, mace_5_max, mace_5_mean = _compute_ece_mace_from_arrays(y_arr, p_arr, n_bins=5)

    rel_10 = compute_reliability_diagram(
        y_arr,
        p_arr,
        series_name=method_name,
        n_bins=10,
        display_label=display_label,
        is_evaluation_only=is_evaluation_only,
    )

    is_single_class = (n_pos == 0) or (n_neg == 0)
    if is_single_class:
        missing_cls = 1 if n_pos == 0 else 0
        single_reason = (
            f"Single-class observation cohort (n_positive={n_pos}, n_negative={n_neg}; "
            f"class {missing_cls} absent). ROC-AUC and PR-AUC are mathematically undefined."
        )
        roc_val: Optional[float] = None
        pr_val: Optional[float] = None
    else:
        single_reason = None
        roc_val = round(float(roc_auc_score(y_arr, p_arr)), 4)
        pr_val = round(float(average_precision_score(y_arr, p_arr)), 4)

    return DiagnosticMethodMetrics(
        method=method_name,
        display_label=rel_10.display_label,
        calibration_status=calibration_status,
        is_deployed_method=bool(is_deployed_method),
        is_evaluation_only=bool(is_evaluation_only),
        n_valid=n_valid,
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=prevalence,
        is_single_class=is_single_class,
        single_class_reason=single_reason,
        brier_score=brier,
        log_loss_value=ll,
        ece_10bin_weighted=round(ece_10, 4),
        mace_10bin_max=round(mace_10_max, 4),
        mace_5bin_max=round(mace_5_max, 4),
        mace_5bin_unweighted_mean=round(mace_5_mean, 4),
        roc_auc=roc_val,
        pr_auc=pr_val,
        pr_auc_definition=PR_AUC_DEFINITION,
        ece_definition=ECE_DEFINITION,
        mace_definition=MACE_DEFINITION,
        reliability_diagram_10bin=rel_10,
        mace_10bin_unweighted_mean=round(mace_10_mean, 4),
    )


# ---------------------------------------------------------------------------
# Implementation C — Paired Bootstrap Analysis
# ---------------------------------------------------------------------------

def compute_paired_brier_bootstrap(
    y_true: Any,
    y_prob_raw: Any,
    y_prob_calibrated: Any,
    endpoint: str = "unknown",
    comparator_method: str = "sigmoid",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    benchmark_delta_brier: Optional[float] = None,
    benchmark_delta_ci_95: Optional[Sequence[float]] = None,
) -> PairedBootstrapResult:
    """
    Compute paired bootstrap 95% confidence intervals for Brier score improvement.

    Sign Convention:
      - `brier_improvement = Brier(raw) - Brier(calibrated)`
        (Positive values indicate calibration reduced Brier error.)
      - `delta_brier_vs_raw = Brier(calibrated) - Brier(raw) = -brier_improvement`
        (Retained alongside `brier_improvement` for exact reconciliation with
        `artifacts/tox21/calibration_benchmark.json`.)

    Resampling Protocol:
      - Uses `np.random.default_rng(random_seed)` and draws `idx = rng.integers(0, n_samples, size=n_samples)`
        once per replicate `b in range(n_bootstrap)`, keeping `(y_true[idx], y_prob_raw[idx], y_prob_calibrated[idx])`
        strictly paired.
    """
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        return PairedBootstrapResult(
            endpoint=endpoint,
            comparator_method=comparator_method,
            status="INVALID_CONFIG",
            status_message=f"n_bootstrap must be a positive integer (got {n_bootstrap}).",
            n_samples=0,
            n_positive=0,
            n_negative=0,
            n_requested_replicates=0,
            n_valid_replicates=0,
            n_two_class_replicates=0,
            n_degenerate_replicates=0,
            random_seed=random_seed,
            raw_brier=None,
            raw_brier_ci_95=None,
            calibrated_brier=None,
            calibrated_brier_ci_95=None,
            brier_improvement=None,
            brier_improvement_ci_95=None,
            delta_brier_vs_raw=None,
            delta_brier_ci_95=None,
            ci_includes_zero=None,
            reconciled_with_benchmark=None,
            interpretation_note="Bootstrap configuration invalid.",
        )

    if y_true is None or y_prob_raw is None or y_prob_calibrated is None:
        return PairedBootstrapResult(
            endpoint=endpoint,
            comparator_method=comparator_method,
            status="UNAVAILABLE",
            status_message="Sample-level labels or paired probability arrays are unavailable.",
            n_samples=0,
            n_positive=0,
            n_negative=0,
            n_requested_replicates=n_bootstrap,
            n_valid_replicates=0,
            n_two_class_replicates=0,
            n_degenerate_replicates=0,
            random_seed=random_seed,
            raw_brier=None,
            raw_brier_ci_95=None,
            calibrated_brier=None,
            calibrated_brier_ci_95=None,
            brier_improvement=None,
            brier_improvement_ci_95=None,
            delta_brier_vs_raw=None,
            delta_brier_ci_95=None,
            ci_includes_zero=None,
            reconciled_with_benchmark=None,
            interpretation_note="Paired bootstrap unavailable because sample-level inputs were not provided.",
        )

    try:
        y_arr = _validate_binary_labels(y_true, context="bootstrap y_true")
        raw_arr = _validate_1d_probabilities(y_prob_raw, context="bootstrap y_prob_raw")
        cal_arr = _validate_1d_probabilities(y_prob_calibrated, context="bootstrap y_prob_calibrated")
    except ValueError as exc:
        return PairedBootstrapResult(
            endpoint=endpoint,
            comparator_method=comparator_method,
            status="INVALID_INPUT",
            status_message=str(exc),
            n_samples=0,
            n_positive=0,
            n_negative=0,
            n_requested_replicates=n_bootstrap,
            n_valid_replicates=0,
            n_two_class_replicates=0,
            n_degenerate_replicates=0,
            random_seed=random_seed,
            raw_brier=None,
            raw_brier_ci_95=None,
            calibrated_brier=None,
            calibrated_brier_ci_95=None,
            brier_improvement=None,
            brier_improvement_ci_95=None,
            delta_brier_vs_raw=None,
            delta_brier_ci_95=None,
            ci_includes_zero=None,
            reconciled_with_benchmark=None,
            interpretation_note=f"Bootstrap input validation failed: {exc}",
        )

    if len(y_arr) != len(raw_arr) or len(y_arr) != len(cal_arr):
        return PairedBootstrapResult(
            endpoint=endpoint,
            comparator_method=comparator_method,
            status="INVALID_INPUT",
            status_message=(
                f"Paired array length mismatch: len(y_true)={len(y_arr)}, "
                f"len(raw)={len(raw_arr)}, len(calibrated)={len(cal_arr)}."
            ),
            n_samples=0,
            n_positive=0,
            n_negative=0,
            n_requested_replicates=n_bootstrap,
            n_valid_replicates=0,
            n_two_class_replicates=0,
            n_degenerate_replicates=0,
            random_seed=random_seed,
            raw_brier=None,
            raw_brier_ci_95=None,
            calibrated_brier=None,
            calibrated_brier_ci_95=None,
            brier_improvement=None,
            brier_improvement_ci_95=None,
            delta_brier_vs_raw=None,
            delta_brier_ci_95=None,
            ci_includes_zero=None,
            reconciled_with_benchmark=None,
            interpretation_note="Paired bootstrap requires equal-length label and prediction arrays.",
        )

    n_samples = int(len(y_arr))
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))

    if n_samples < 2 or n_pos == 0 or n_neg == 0:
        return PairedBootstrapResult(
            endpoint=endpoint,
            comparator_method=comparator_method,
            status="DEGENERATE_SAMPLE",
            status_message=(
                f"Degenerate or single-class cohort (n={n_samples}, pos={n_pos}, neg={n_neg}); "
                "paired bootstrap confidence intervals require at least 2 observations and both binary classes."
            ),
            n_samples=n_samples,
            n_positive=n_pos,
            n_negative=n_neg,
            n_requested_replicates=n_bootstrap,
            n_valid_replicates=0,
            n_two_class_replicates=0,
            n_degenerate_replicates=n_bootstrap,
            random_seed=random_seed,
            raw_brier=round(float(brier_score_loss(y_arr, raw_arr)), 4) if n_samples > 0 else None,
            raw_brier_ci_95=None,
            calibrated_brier=round(float(brier_score_loss(y_arr, cal_arr)), 4) if n_samples > 0 else None,
            calibrated_brier_ci_95=None,
            brier_improvement=None,
            brier_improvement_ci_95=None,
            delta_brier_vs_raw=None,
            delta_brier_ci_95=None,
            ci_includes_zero=None,
            reconciled_with_benchmark=None,
            interpretation_note="Single-class or <2 observation sample cannot support paired bootstrap inference.",
        )

    raw_brier_exact = float(brier_score_loss(y_arr, raw_arr))
    cal_brier_exact = float(brier_score_loss(y_arr, cal_arr))
    delta_exact = round(cal_brier_exact - raw_brier_exact, 6)
    improvement_exact = round(raw_brier_exact - cal_brier_exact, 6)

    rng = np.random.default_rng(random_seed)
    boot_raw_briers: List[float] = []
    boot_cal_briers: List[float] = []
    boot_deltas: List[float] = []
    boot_improvements: List[float] = []
    n_two_class = 0
    n_degenerate = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_b = y_arr[idx]
        if np.any(y_b == 1) and np.any(y_b == 0):
            n_two_class += 1
        else:
            n_degenerate += 1

        b_cal = float(brier_score_loss(y_b, cal_arr[idx]))
        b_raw = float(brier_score_loss(y_b, raw_arr[idx]))
        boot_cal_briers.append(b_cal)
        boot_raw_briers.append(b_raw)
        boot_deltas.append(b_cal - b_raw)
        boot_improvements.append(b_raw - b_cal)

    n_valid_reps = len(boot_improvements)
    raw_ci = [
        round(float(np.percentile(boot_raw_briers, 2.5)), 4),
        round(float(np.percentile(boot_raw_briers, 97.5)), 4),
    ]
    cal_ci = [
        round(float(np.percentile(boot_cal_briers, 2.5)), 4),
        round(float(np.percentile(boot_cal_briers, 97.5)), 4),
    ]
    delta_ci = [
        round(float(np.percentile(boot_deltas, 2.5)), 6),
        round(float(np.percentile(boot_deltas, 97.5)), 6),
    ]
    # Note: percentile(boot_improvements, [2.5, 97.5]) == [-delta_ci[1], -delta_ci[0]]
    imp_ci = [
        round(float(np.percentile(boot_improvements, 2.5)), 6),
        round(float(np.percentile(boot_improvements, 97.5)), 6),
    ]

    ci_includes_zero = bool(imp_ci[0] <= 0.0 <= imp_ci[1])

    reconciled: Optional[bool] = None
    if benchmark_delta_brier is not None and benchmark_delta_ci_95 is not None and len(benchmark_delta_ci_95) == 2:
        reconciled = bool(
            math.isclose(delta_exact, float(benchmark_delta_brier), abs_tol=1e-6)
            and math.isclose(delta_ci[0], float(benchmark_delta_ci_95[0]), abs_tol=1e-6)
            and math.isclose(delta_ci[1], float(benchmark_delta_ci_95[1]), abs_tol=1e-6)
        )

    if ci_includes_zero:
        interp = (
            f"Point estimate shows Brier improvement of +{improvement_exact:.6f} "
            f"(Delta Brier = {delta_exact:+.6f}), but the 95% paired bootstrap CI "
            f"[{imp_ci[0]:+.6f}, {imp_ci[1]:+.6f}] spans zero (n_pos={n_pos} test positives). "
            "A confidence interval spanning zero reflects finite-sample uncertainty on a low-prevalence "
            "assay and is NOT proof that calibration is ineffective."
        )
    elif imp_ci[0] > 0.0:
        interp = (
            f"Point estimate shows Brier improvement of +{improvement_exact:.6f} "
            f"(Delta Brier = {delta_exact:+.6f}) with 95% paired bootstrap CI "
            f"[{imp_ci[0]:+.6f}, {imp_ci[1]:+.6f}] strictly above zero across {n_valid_reps} replicates. "
            "Statistical improvement in in-vitro Brier reliability does NOT establish clinical safety."
        )
    else:
        interp = (
            f"Calibrated Brier score is higher than raw Brier score on this sample "
            f"(Brier improvement = {improvement_exact:+.6f}, 95% CI [{imp_ci[0]:+.6f}, {imp_ci[1]:+.6f}])."
        )

    return PairedBootstrapResult(
        endpoint=endpoint,
        comparator_method=comparator_method,
        status="AVAILABLE",
        status_message=None,
        n_samples=n_samples,
        n_positive=n_pos,
        n_negative=n_neg,
        n_requested_replicates=n_bootstrap,
        n_valid_replicates=n_valid_reps,
        n_two_class_replicates=n_two_class,
        n_degenerate_replicates=n_degenerate,
        random_seed=random_seed,
        raw_brier=round(raw_brier_exact, 4),
        raw_brier_ci_95=raw_ci,
        calibrated_brier=round(cal_brier_exact, 4),
        calibrated_brier_ci_95=cal_ci,
        brier_improvement=improvement_exact,
        brier_improvement_ci_95=imp_ci,
        delta_brier_vs_raw=delta_exact,
        delta_brier_ci_95=delta_ci,
        ci_includes_zero=ci_includes_zero,
        reconciled_with_benchmark=reconciled,
        interpretation_note=interp,
    )


# ---------------------------------------------------------------------------
# Implementation D — Applicability-Domain Stratification
# ---------------------------------------------------------------------------

def _evaluate_single_ad_stratum(
    stratum_id: str,
    stratum_label: str,
    y_sub: np.ndarray,
    raw_sub: np.ndarray,
    sig_sub: np.ndarray,
    sim_sub: Optional[np.ndarray],
    min_positives_rule: int = MIN_POSITIVES_FOR_STRATUM_SUPPORT,
) -> ADStratumDiagnostics:
    """Compute calibration and support diagnostics for a single AD similarity subset."""
    n_total = int(len(y_sub))
    if n_total == 0:
        return ADStratumDiagnostics(
            stratum_id=stratum_id,
            stratum_label=stratum_label,
            similarity_min=None,
            similarity_max=None,
            n_total=0,
            n_positive=0,
            n_negative=0,
            prevalence=None,
            support_status="INSUFFICIENT_SUPPORT",
            support_warning="Empty stratum (0 observations); no metrics computed.",
            raw_brier=None,
            sigmoid_brier=None,
            brier_improvement=None,
            raw_log_loss=None,
            sigmoid_log_loss=None,
            raw_ece_10bin=None,
            sigmoid_ece_10bin=None,
            raw_mace_10bin=None,
            sigmoid_mace_10bin=None,
            raw_roc_auc=None,
            sigmoid_roc_auc=None,
            raw_pr_auc=None,
            sigmoid_pr_auc=None,
        )

    n_pos = int(np.sum(y_sub == 1))
    n_neg = int(np.sum(y_sub == 0))
    prevalence = round(float(n_pos / n_total), 4)
    sim_min = round(float(np.min(sim_sub)), 4) if sim_sub is not None and len(sim_sub) > 0 else None
    sim_max = round(float(np.max(sim_sub)), 4) if sim_sub is not None and len(sim_sub) > 0 else None

    raw_m = compute_diagnostic_metrics(y_sub, raw_sub, method_name="uncalibrated")
    sig_m = compute_diagnostic_metrics(y_sub, sig_sub, method_name="sigmoid", is_deployed_method=True)

    brier_imp = (
        round(float(raw_m.brier_score - sig_m.brier_score), 4)
        if raw_m.brier_score is not None and sig_m.brier_score is not None
        else None
    )

    if n_pos < min_positives_rule or n_neg < min_positives_rule:
        support_status = "INSUFFICIENT_SUPPORT"
        support_warning = (
            f"Stratum has {n_pos} positive and {n_neg} negative observations "
            f"(< {min_positives_rule} required by conservative reporting policy). "
            "Positive-class-dependent discrimination metrics (ROC-AUC, PR-AUC) are suppressed "
            "to prevent over-interpreting noisy small-sample estimates."
        )
        raw_roc = None
        sig_roc = None
        raw_pr = None
        sig_pr = None
    else:
        support_status = "SUFFICIENT_SUPPORT"
        support_warning = None
        raw_roc = raw_m.roc_auc
        sig_roc = sig_m.roc_auc
        raw_pr = raw_m.pr_auc
        sig_pr = sig_m.pr_auc

    return ADStratumDiagnostics(
        stratum_id=stratum_id,
        stratum_label=stratum_label,
        similarity_min=sim_min,
        similarity_max=sim_max,
        n_total=n_total,
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=prevalence,
        support_status=support_status,
        support_warning=support_warning,
        raw_brier=raw_m.brier_score,
        sigmoid_brier=sig_m.brier_score,
        brier_improvement=brier_imp,
        raw_log_loss=raw_m.log_loss_value,
        sigmoid_log_loss=sig_m.log_loss_value,
        raw_ece_10bin=raw_m.ece_10bin_weighted,
        sigmoid_ece_10bin=sig_m.ece_10bin_weighted,
        raw_mace_10bin=raw_m.mace_10bin_max,
        sigmoid_mace_10bin=sig_m.mace_10bin_max,
        raw_roc_auc=raw_roc,
        sigmoid_roc_auc=sig_roc,
        raw_pr_auc=raw_pr,
        sigmoid_pr_auc=sig_pr,
    )


def compute_ad_stratified_diagnostics(
    y_true: Any,
    y_prob_raw: Any,
    y_prob_sigmoid: Any,
    ad_scores: Sequence[Optional[float]],
    endpoint: str = "unknown",
    training_corpus_size: int = 0,
    min_positives_rule: int = MIN_POSITIVES_FOR_STRATUM_SUPPORT,
    custom_quantile_edges: Optional[Sequence[float]] = None,
) -> EndpointADStratificationReport:
    """
    Compute exploratory calibration diagnostics across endpoint-specific Tanimoto similarity strata.

    Scientific Governance:
      - Does NOT borrow ClinTox applicability-domain cutoffs (`0.25`, `0.40`).
      - Does NOT invent fixed validated Tanimoto thresholds; partitions valid test observations
        into 3 empirical quantile tertiles (`Q1: 0-33.3%`, `Q2: 33.3-66.7%`, `Q3: 66.7-100%`)
        of the observed endpoint-specific maximum Tanimoto similarity distribution (unless
        explicit `custom_quantile_edges` are supplied for testing).
      - Preserves missing (`None` or `NaN`) AD scores in a separate `unavailable_stratum` record
        instead of forcing them into a similarity stratum.
      - Flags any stratum with `< min_positives_rule` (default 10) positives as `INSUFFICIENT_SUPPORT`.
    """
    y_arr = _validate_binary_labels(y_true, context=f"{endpoint} AD y_true")
    raw_arr = _validate_1d_probabilities(y_prob_raw, context=f"{endpoint} AD y_prob_raw")
    sig_arr = _validate_1d_probabilities(y_prob_sigmoid, context=f"{endpoint} AD y_prob_sigmoid")

    n_total = len(y_arr)
    if len(raw_arr) != n_total or len(sig_arr) != n_total or len(ad_scores) != n_total:
        raise ValueError(
            f"Length mismatch in compute_ad_stratified_diagnostics for '{endpoint}': "
            f"len(y)={n_total}, len(raw)={len(raw_arr)}, len(sig)={len(sig_arr)}, len(ad)={len(ad_scores)}."
        )

    valid_mask = np.zeros(n_total, dtype=bool)
    sim_values = np.full(n_total, np.nan, dtype=np.float64)

    for i, val in enumerate(ad_scores):
        if val is None:
            continue
        if isinstance(val, (bool, np.bool_)):
            raise ValueError("Boolean value encountered in ad_scores.")
        fval = float(val)
        if math.isnan(fval):
            continue
        if not math.isfinite(fval) or fval < 0.0 or fval > 1.0:
            raise ValueError(f"Invalid Tanimoto similarity score {val} outside [0.0, 1.0].")
        valid_mask[i] = True
        sim_values[i] = fval

    n_valid_ad = int(np.sum(valid_mask))
    n_unavail_ad = int(n_total - n_valid_ad)

    unavail_stratum: Optional[ADStratumDiagnostics] = None
    if n_unavail_ad > 0:
        unavail_mask = ~valid_mask
        unavail_stratum = _evaluate_single_ad_stratum(
            stratum_id="UNAVAILABLE_AD",
            stratum_label="Missing / Unavailable AD Score",
            y_sub=y_arr[unavail_mask],
            raw_sub=raw_arr[unavail_mask],
            sig_sub=sig_arr[unavail_mask],
            sim_sub=None,
            min_positives_rule=min_positives_rule,
        )

    if n_valid_ad < 6:
        return EndpointADStratificationReport(
            endpoint=endpoint,
            status="UNAVAILABLE",
            status_message=(
                f"Insufficient valid applicability-domain similarity scores ({n_valid_ad} valid of {n_total}) "
                "to construct empirical tertile strata."
            ),
            is_exploratory=True,
            stratification_method="EXPLORATORY_EMPIRICAL_TERTILES",
            domain_thresholds_validated=False,
            min_positives_support_rule=min_positives_rule,
            n_total_test=n_total,
            n_valid_ad_scores=n_valid_ad,
            n_unavailable_ad_scores=n_unavail_ad,
            training_corpus_size=int(training_corpus_size),
            quantile_edges=[],
            strata=[],
            unavailable_stratum=unavail_stratum,
            scientific_disclaimer=AD_STRATIFICATION_DISCLAIMER,
        )

    valid_sims = sim_values[valid_mask]
    valid_y = y_arr[valid_mask]
    valid_raw = raw_arr[valid_mask]
    valid_sig = sig_arr[valid_mask]

    if custom_quantile_edges is not None:
        edges = [float(x) for x in custom_quantile_edges]
        if len(edges) != 4 or any(edges[i] > edges[i + 1] for i in range(3)):
            raise ValueError("custom_quantile_edges must provide 4 non-decreasing boundary values [min, q1, q2, max].")
        q1, q2 = edges[1], edges[2]
        q_edges = [round(edges[0], 4), round(q1, 4), round(q2, 4), round(edges[3], 4)]
    else:
        q1_raw, q2_raw = np.quantile(valid_sims, [1.0 / 3.0, 2.0 / 3.0])
        q1 = float(q1_raw)
        q2 = float(q2_raw)
        q_edges = [
            round(float(np.min(valid_sims)), 4),
            round(q1, 4),
            round(q2, 4),
            round(float(np.max(valid_sims)), 4),
        ]

    # Partition valid AD observations into 3 mutually exclusive, collectively exhaustive tertiles
    mask_t1 = valid_sims <= q1
    mask_t2 = (valid_sims > q1) & (valid_sims <= q2)
    mask_t3 = valid_sims > q2

    strata_defs = [
        (
            "TERTILE_1_LOWER",
            f"Lower Similarity Tertile (Q1: <= {q1:.3f})",
            mask_t1,
        ),
        (
            "TERTILE_2_MIDDLE",
            f"Middle Similarity Tertile (Q2: ({q1:.3f}, {q2:.3f}])",
            mask_t2,
        ),
        (
            "TERTILE_3_UPPER",
            f"Upper Similarity Tertile (Q3: > {q2:.3f})",
            mask_t3,
        ),
    ]

    strata_list: List[ADStratumDiagnostics] = []
    for s_id, s_label, s_mask in strata_defs:
        strata_list.append(
            _evaluate_single_ad_stratum(
                stratum_id=s_id,
                stratum_label=s_label,
                y_sub=valid_y[s_mask],
                raw_sub=valid_raw[s_mask],
                sig_sub=valid_sig[s_mask],
                sim_sub=valid_sims[s_mask],
                min_positives_rule=min_positives_rule,
            )
        )

    if sum(s.n_total for s in strata_list) != n_valid_ad:
        raise RuntimeError("AD stratification count mismatch across tertiles.")

    any_insufficient = any(s.support_status == "INSUFFICIENT_SUPPORT" for s in strata_list)
    status_code = "AVAILABLE_WITH_SPARSE_STRATA" if any_insufficient else "AVAILABLE"
    status_msg = (
        "One or more empirical similarity tertiles contain fewer than "
        f"{min_positives_rule} positive observations and are flagged INSUFFICIENT_SUPPORT."
        if any_insufficient
        else "All three empirical similarity tertiles meet the >= 10 positive observation support rule."
    )

    return EndpointADStratificationReport(
        endpoint=endpoint,
        status=status_code,
        status_message=status_msg,
        is_exploratory=True,
        stratification_method="EXPLORATORY_EMPIRICAL_TERTILES",
        domain_thresholds_validated=False,
        min_positives_support_rule=min_positives_rule,
        n_total_test=n_total,
        n_valid_ad_scores=n_valid_ad,
        n_unavailable_ad_scores=n_unavail_ad,
        training_corpus_size=int(training_corpus_size),
        quantile_edges=q_edges,
        strata=strata_list,
        unavailable_stratum=unavail_stratum,
        scientific_disclaimer=AD_STRATIFICATION_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Sample-Level Held-Out Test Reconstruction & Caching Service
# ---------------------------------------------------------------------------

_DATASET_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_FP_ROW_CACHE: Dict[Tuple[str, int], np.ndarray] = {}
_ENDPOINT_BUNDLE_CACHE: Dict[Tuple[str, str, str, str, str], EndpointReliabilityDiagnosticBundle] = {}
_FILE_SHA256_CACHE: Dict[Tuple[str, int, int], str] = {}


def _get_cached_file_sha256(path: str) -> str:
    """Compute file SHA-256 cached by `(abspath, st_mtime_ns, st_size)`."""
    if not os.path.exists(path):
        return "missing"
    st = os.stat(path)
    key = (os.path.abspath(path), int(st.st_mtime_ns), int(st.st_size))
    if key not in _FILE_SHA256_CACHE:
        _FILE_SHA256_CACHE[key] = compute_file_sha256(path)
    return _FILE_SHA256_CACHE[key]


def _get_cached_tox21_dataframe_and_valid_mask(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """
    Load `data/raw/tox21.csv` and compute `valid_mask_arr` once per `(csv_path, dataset_sha256)`.
    """
    dataset_sha = _get_cached_file_sha256(csv_path)
    cache_key = (csv_path, dataset_sha)
    if cache_key in _DATASET_CACHE:
        entry = _DATASET_CACHE[cache_key]
        return entry["df"], entry["valid_mask_arr"], dataset_sha

    df = pd.read_csv(csv_path)
    valid_mask = np.array(
        [
            bool(validate_smiles(str(smi)).is_valid and validate_smiles(str(smi)).canonical_smiles)
            for smi in df["smiles"]
        ],
        dtype=bool,
    )
    _DATASET_CACHE[cache_key] = {
        "df": df,
        "valid_mask_arr": valid_mask,
    }
    return df, valid_mask, dataset_sha


def _featurize_global_indices(df: pd.DataFrame, dataset_sha: str, global_indices: np.ndarray) -> np.ndarray:
    """Generate or retrieve cached 2048-bit Morgan fingerprints for specific row indices of `df`."""
    rows: List[np.ndarray] = []
    smiles_col = df["smiles"].values
    for idx in global_indices:
        row_idx = int(idx)
        key = (dataset_sha, row_idx)
        if key not in _FP_ROW_CACHE:
            val_res = validate_smiles(str(smiles_col[row_idx]))
            if not val_res.is_valid or not val_res.canonical_smiles:
                raise ValueError(f"Invalid SMILES at global index {row_idx}.")
            fp_res = generate_morgan_fingerprint(
                val_res.canonical_smiles, radius=2, n_bits=2048, return_bit_info=False
            )
            _FP_ROW_CACHE[key] = fp_res.numpy_vector
        rows.append(_FP_ROW_CACHE[key])
    return np.asarray(rows, dtype=np.int8)


def reconstruct_verified_test_samples(
    endpoint: str,
    artifact_dir: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Deterministically reconstruct the locked 20% held-out test partition for `endpoint`
    and verify its SHA-256 index digests, dataset hash, and base model hash against the
    read-only `{safe_ep}_calibration.json` artifact without retraining or refitting anything.
    """
    if endpoint not in TOX21_ENDPOINTS:
        return {
            "status": "INVALID_ENDPOINT",
            "error": f"Unknown Tox21 endpoint '{endpoint}'.",
        }

    resolved_dir = resolve_path(artifact_dir or TOX21_ARTIFACT_DIR)
    resolved_csv = resolve_path(csv_path or TOX21_DATA_PATH)
    safe_ep = endpoint.replace("-", "_")

    model_path = os.path.join(resolved_dir, f"{safe_ep}_model.joblib")
    cal_service = Tox21CalibrationService(artifact_dir=resolved_dir)
    if os.path.exists(model_path):
        cal_service._model_hashes[endpoint] = _get_cached_file_sha256(model_path)
    cal_art, cal_status, cal_warn = cal_service.load_endpoint_calibration(endpoint)
    if cal_art is None:
        return {
            "status": cal_status,
            "error": cal_warn or f"Calibration artifact unavailable for '{endpoint}'.",
        }

    if not os.path.exists(resolved_csv):
        return {
            "status": "DATASET_UNAVAILABLE",
            "error": f"Raw Tox21 dataset not found at '{resolved_csv}'.",
            "calibration_artifact": cal_art,
        }

    model_path = os.path.join(resolved_dir, f"{safe_ep}_model.joblib")
    if not os.path.exists(model_path):
        return {
            "status": "MODEL_UNAVAILABLE",
            "error": f"Deployed model not found at '{model_path}'.",
            "calibration_artifact": cal_art,
        }

    try:
        df, valid_mask_arr, dataset_sha = _get_cached_tox21_dataframe_and_valid_mask(resolved_csv)
    except Exception as exc:
        return {
            "status": "DATASET_ERROR",
            "error": f"Failed reading Tox21 dataset: {exc}",
            "calibration_artifact": cal_art,
        }

    prov = cal_art.get("provenance", {})
    expected_ds_sha = prov.get("dataset_sha256")
    if expected_ds_sha and dataset_sha.lower() != str(expected_ds_sha).lower():
        return {
            "status": "DATASET_HASH_MISMATCH",
            "error": f"Dataset SHA-256 mismatch for '{endpoint}'.",
            "calibration_artifact": cal_art,
        }

    if endpoint not in df.columns:
        return {
            "status": "DATASET_ERROR",
            "error": f"Column '{endpoint}' missing from dataset.",
            "calibration_artifact": cal_art,
        }

    ep_series = df[endpoint]
    not_na_mask = ep_series.notna().values & valid_mask_arr
    ep_global_indices = np.where(not_na_mask)[0]
    y_ep = ep_series.values[not_na_mask].astype(np.int64)
    rel_indices = np.arange(len(y_ep), dtype=np.int64)

    seed = int(prov.get("random_seed", DEFAULT_RANDOM_SEED))
    train_rel_idx, test_rel_idx, y_train, y_test = train_test_split(
        rel_indices,
        y_ep,
        test_size=0.2,
        random_state=seed,
        stratify=y_ep,
    )
    train_global_idx = ep_global_indices[train_rel_idx]
    test_global_idx = ep_global_indices[test_rel_idx]

    train_idx_sha = _hash_index_array(train_global_idx)
    test_idx_sha = _hash_index_array(test_global_idx)

    expected_train_sha = prov.get("train_global_indices_sha256")
    expected_test_sha = prov.get("test_global_indices_sha256")
    if expected_train_sha and train_idx_sha != expected_train_sha:
        return {
            "status": "SPLIT_HASH_MISMATCH",
            "error": f"Reconstructed train split index hash mismatch for '{endpoint}'.",
            "calibration_artifact": cal_art,
        }
    if expected_test_sha and test_idx_sha != expected_test_sha:
        return {
            "status": "SPLIT_HASH_MISMATCH",
            "error": f"Reconstructed test split index hash mismatch for '{endpoint}'.",
            "calibration_artifact": cal_art,
        }

    X_test = _featurize_global_indices(df, dataset_sha, test_global_idx)
    base_model = joblib.load(model_path)
    if hasattr(base_model, "n_jobs"):
        base_model.n_jobs = 1

    y_prob_raw = extract_positive_class_probabilities(base_model, X_test)
    cal_params = cal_art.get("calibrator_parameters", {})
    sig_params = cal_params.get("sigmoid")
    iso_params = cal_params.get("isotonic")

    y_prob_sig = (
        np.asarray(predict_with_sigmoid_params(y_prob_raw, sig_params), dtype=np.float64)
        if sig_params
        else None
    )
    y_prob_iso = (
        np.asarray(predict_with_isotonic_params(y_prob_raw, iso_params), dtype=np.float64)
        if iso_params
        else None
    )

    # Compute endpoint-specific applicability-domain Tanimoto similarity scores against
    # isolated training fingerprints (`training_fingerprints.npz`) if available.
    npz_path = os.path.join(resolved_dir, "training_fingerprints.npz")
    ad_scores: Optional[List[Optional[float]]] = None
    training_corpus_size = 0
    if os.path.exists(npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as npz_data:
                fp_key = f"{safe_ep}_fingerprints"
                if fp_key in npz_data:
                    ref_fps = np.asarray(npz_data[fp_key], dtype=np.float32)
                    training_corpus_size = int(ref_fps.shape[0])
                    if training_corpus_size > 0:
                        q_fps = X_test.astype(np.float32)
                        inter = q_fps @ ref_fps.T
                        q_sum = np.sum(q_fps, axis=1, keepdims=True)
                        r_sum = np.sum(ref_fps, axis=1, keepdims=True).T
                        union = q_sum + r_sum - inter
                        sims = np.where(union > 0.0, inter / union, 0.0)
                        max_sims = np.round(np.max(sims, axis=1), 4)
                        ad_scores = [float(x) for x in max_sims]
        except Exception as exc:
            logger.warning(f"Failed loading AD reference corpus for '{endpoint}': {exc}")
            ad_scores = None

    return {
        "status": "AVAILABLE",
        "error": None,
        "endpoint": endpoint,
        "calibration_artifact": cal_art,
        "y_test": y_test,
        "y_prob_raw": y_prob_raw,
        "y_prob_sigmoid": y_prob_sig,
        "y_prob_isotonic": y_prob_iso,
        "ad_scores": ad_scores,
        "training_corpus_size": training_corpus_size,
        "train_global_indices_sha256": train_idx_sha,
        "test_global_indices_sha256": test_idx_sha,
    }


def load_endpoint_reliability_diagnostics(
    endpoint: str,
    artifact_dir: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
    reconstruct_sample_level: bool = True,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> EndpointReliabilityDiagnosticBundle:
    """
    Load and verify the complete Phase B.2 reliability diagnostics bundle for `endpoint`.

    Combines:
      1. Read-only benchmark reference (`calibration_benchmark.json` / `{safe_ep}_calibration.json`)
      2. Cryptographically verified sample-level held-out test reconstruction (when available)
      3. Reliability diagrams (10-bin equal-width) for Raw, Sigmoid, and Isotonic (evaluation-only)
      4. Endpoint-level calibration & discrimination metrics with reconciliation verification
      5. Paired bootstrap 95% CIs (`B = 500, seed = 42`) for `Brier(raw) - Brier(calibrated)`
      6. Exploratory applicability-domain Tanimoto similarity tertile stratification
    """
    ep_meta = TOX21_ENDPOINT_NAMES.get(
        endpoint,
        {
            "name": endpoint,
            "family": "Nuclear Receptor (NR)" if endpoint.startswith("NR") else "Stress Response (SR)",
            "selector_label": endpoint,
        },
    )

    if endpoint not in TOX21_ENDPOINTS:
        empty_boot = compute_paired_brier_bootstrap(None, None, None, endpoint=endpoint)
        empty_ad = EndpointADStratificationReport(
            endpoint=endpoint,
            status="UNAVAILABLE",
            status_message=f"Unknown Tox21 endpoint '{endpoint}'.",
            is_exploratory=True,
            stratification_method="EXPLORATORY_EMPIRICAL_TERTILES",
            domain_thresholds_validated=False,
            min_positives_support_rule=MIN_POSITIVES_FOR_STRATUM_SUPPORT,
            n_total_test=0,
            n_valid_ad_scores=0,
            n_unavailable_ad_scores=0,
            training_corpus_size=0,
            quantile_edges=[],
            strata=[],
            unavailable_stratum=None,
            scientific_disclaimer=AD_STRATIFICATION_DISCLAIMER,
        )
        return EndpointReliabilityDiagnosticBundle(
            endpoint=endpoint,
            display_name=ep_meta["name"],
            pathway_family=ep_meta["family"],
            selector_label=ep_meta["selector_label"],
            overall_status="INVALID_ENDPOINT",
            status_message=f"Unknown Tox21 endpoint '{endpoint}'.",
            calibration_artifact_status="UNAVAILABLE",
            benchmark_status="UNAVAILABLE",
            sample_level_status="UNAVAILABLE",
            reconciled_with_benchmark=False,
            reconciliation_details={},
            recommended_method="uncalibrated",
            selection_rationale="",
            isotonic_selection_eligible=False,
            data_summary={},
            score_distribution_comparison={},
            series_metrics={},
            reliability_series={},
            bootstrap_sigmoid=empty_boot,
            bootstrap_isotonic=None,
            ad_stratification=empty_ad,
            scientific_disclaimer=TOX21_CALIBRATION_DISCLAIMER,
        )

    resolved_dir = resolve_path(artifact_dir or TOX21_ARTIFACT_DIR)
    resolved_csv = resolve_path(csv_path or TOX21_DATA_PATH)
    safe_ep = endpoint.replace("-", "_")

    cal_path = os.path.join(resolved_dir, f"{safe_ep}_calibration.json")
    model_path = os.path.join(resolved_dir, f"{safe_ep}_model.joblib")
    npz_path = os.path.join(resolved_dir, "training_fingerprints.npz")

    cal_sha = _get_cached_file_sha256(cal_path)
    model_sha = _get_cached_file_sha256(model_path)
    csv_sha = _get_cached_file_sha256(resolved_csv) if reconstruct_sample_level else "skipped"
    npz_sha = _get_cached_file_sha256(npz_path) if reconstruct_sample_level else "skipped"

    cache_key = (endpoint, cal_sha, model_sha, csv_sha, f"{npz_sha}:{reconstruct_sample_level}:{n_bootstrap}:{random_seed}")
    if cache_key in _ENDPOINT_BUNDLE_CACHE:
        return _ENDPOINT_BUNDLE_CACHE[cache_key]

    cal_service = Tox21CalibrationService(artifact_dir=resolved_dir)
    if model_sha != "missing":
        cal_service._model_hashes[endpoint] = model_sha
    cal_art, cal_status, cal_warn = cal_service.load_endpoint_calibration(endpoint)

    # Also check calibration_benchmark.json if present
    bench_path = os.path.join(resolved_dir, "calibration_benchmark.json")
    bench_ep_data: Optional[Dict[str, Any]] = None
    bench_status = "UNAVAILABLE"
    if os.path.exists(bench_path):
        try:
            with open(bench_path, "r", encoding="utf-8") as f:
                bench_root = json.load(f)
            if isinstance(bench_root, dict) and bench_root.get("schema_version") == CALIBRATION_SCHEMA_VERSION:
                ep_map = bench_root.get("endpoints", {})
                if isinstance(ep_map, dict) and endpoint in ep_map:
                    bench_ep_data = ep_map[endpoint]
                    bench_status = "AVAILABLE"
            else:
                bench_status = "UNSUPPORTED_SCHEMA"
        except Exception as exc:
            logger.warning(f"Failed reading calibration_benchmark.json for '{endpoint}': {exc}")
            bench_status = "CORRUPT_ARTIFACT"

    ref_art = cal_art if cal_art is not None else bench_ep_data
    if ref_art is None:
        empty_boot = compute_paired_brier_bootstrap(None, None, None, endpoint=endpoint)
        empty_ad = EndpointADStratificationReport(
            endpoint=endpoint,
            status="UNAVAILABLE",
            status_message=cal_warn or f"No verified calibration or benchmark artifact available for '{endpoint}'.",
            is_exploratory=True,
            stratification_method="EXPLORATORY_EMPIRICAL_TERTILES",
            domain_thresholds_validated=False,
            min_positives_support_rule=MIN_POSITIVES_FOR_STRATUM_SUPPORT,
            n_total_test=0,
            n_valid_ad_scores=0,
            n_unavailable_ad_scores=0,
            training_corpus_size=0,
            quantile_edges=[],
            strata=[],
            unavailable_stratum=None,
            scientific_disclaimer=AD_STRATIFICATION_DISCLAIMER,
        )
        return EndpointReliabilityDiagnosticBundle(
            endpoint=endpoint,
            display_name=ep_meta["name"],
            pathway_family=ep_meta["family"],
            selector_label=ep_meta["selector_label"],
            overall_status=cal_status if cal_status != "AVAILABLE" else bench_status,
            status_message=cal_warn or f"Calibration diagnostics unavailable for '{endpoint}'.",
            calibration_artifact_status=cal_status,
            benchmark_status=bench_status,
            sample_level_status="UNAVAILABLE",
            reconciled_with_benchmark=False,
            reconciliation_details={},
            recommended_method="uncalibrated",
            selection_rationale="",
            isotonic_selection_eligible=False,
            data_summary={},
            score_distribution_comparison={},
            series_metrics={},
            reliability_series={},
            bootstrap_sigmoid=empty_boot,
            bootstrap_isotonic=None,
            ad_stratification=empty_ad,
            scientific_disclaimer=TOX21_CALIBRATION_DISCLAIMER,
        )

    gov = ref_art.get("selection_governance", {})
    rec_method = str(gov.get("recommended_method", "sigmoid"))
    rationale = str(gov.get("selection_rationale", ""))
    iso_eligible = bool(gov.get("isotonic_selection_eligible", False))
    data_summary = dict(ref_art.get("data_summary", {}))
    score_dist = dict(ref_art.get("score_distribution_comparison", {}))
    held_out_bench = ref_art.get("held_out_test_metrics", {})

    series_metrics: Dict[str, DiagnosticMethodMetrics] = {}
    reliability_series: Dict[str, ReliabilityDiagramSeries] = {}
    reconciliation_details: Dict[str, bool] = {}

    sample_rec = (
        reconstruct_verified_test_samples(endpoint, artifact_dir=resolved_dir, csv_path=resolved_csv)
        if reconstruct_sample_level and cal_art is not None
        else {"status": "SKIPPED", "error": "Sample-level reconstruction not requested or calibration artifact invalid."}
    )

    if sample_rec.get("status") == "AVAILABLE":
        sample_status = "AVAILABLE_SHA256_VERIFIED"
        y_test = sample_rec["y_test"]
        p_raw = sample_rec["y_prob_raw"]
        p_sig = sample_rec["y_prob_sigmoid"]
        p_iso = sample_rec["y_prob_isotonic"]

        # 1. Raw uncalibrated series
        raw_diag = compute_diagnostic_metrics(
            y_test,
            p_raw,
            method_name="uncalibrated",
            calibration_status="AVAILABLE",
            is_deployed_method=(rec_method == "uncalibrated"),
            is_evaluation_only=False,
        )
        series_metrics["uncalibrated"] = raw_diag
        if raw_diag.reliability_diagram_10bin is not None:
            reliability_series["uncalibrated"] = raw_diag.reliability_diagram_10bin

        # 2. Sigmoid series
        if p_sig is not None:
            sig_diag = compute_diagnostic_metrics(
                y_test,
                p_sig,
                method_name="sigmoid",
                calibration_status=cal_status,
                is_deployed_method=(rec_method == "sigmoid"),
                is_evaluation_only=(rec_method != "sigmoid"),
            )
            series_metrics["sigmoid"] = sig_diag
            if sig_diag.reliability_diagram_10bin is not None:
                reliability_series["sigmoid"] = sig_diag.reliability_diagram_10bin

        # 3. Isotonic series (explicitly marked as evaluation-only comparator when sigmoid is selected)
        if p_iso is not None:
            iso_diag = compute_diagnostic_metrics(
                y_test,
                p_iso,
                method_name="isotonic",
                calibration_status="EVALUATION_ONLY" if rec_method != "isotonic" else cal_status,
                is_deployed_method=(rec_method == "isotonic"),
                is_evaluation_only=(rec_method != "isotonic"),
            )
            series_metrics["isotonic"] = iso_diag
            if iso_diag.reliability_diagram_10bin is not None:
                reliability_series["isotonic"] = iso_diag.reliability_diagram_10bin

        # Reconcile computed metrics with held_out_test_metrics in benchmark JSON
        for m_key, diag_obj in series_metrics.items():
            b_m = held_out_bench.get(m_key, {})
            if b_m:
                b_curve = b_m.get("calibration_curve_10bin", {})
                reconciliation_details[f"{m_key}_brier"] = (diag_obj.brier_score == b_m.get("brier_score"))
                reconciliation_details[f"{m_key}_log_loss"] = (diag_obj.log_loss_value == b_m.get("log_loss_value"))
                reconciliation_details[f"{m_key}_ece_10bin"] = (
                    diag_obj.ece_10bin_weighted == b_m.get("ece_10bin_weighted")
                )
                reconciliation_details[f"{m_key}_mace_5bin_mean"] = (
                    diag_obj.mace_5bin_unweighted_mean == b_m.get("mace_5bin_unweighted")
                )
                reconciliation_details[f"{m_key}_roc_auc"] = (diag_obj.roc_auc == b_m.get("roc_auc"))
                reconciliation_details[f"{m_key}_pr_auc"] = (diag_obj.pr_auc == b_m.get("average_precision"))
                if diag_obj.reliability_diagram_10bin and b_curve:
                    reconciliation_details[f"{m_key}_bin_counts"] = (
                        diag_obj.reliability_diagram_10bin.bin_counts == b_curve.get("bin_counts")
                    )

        # Paired bootstrap for Sigmoid and Isotonic
        b_sig_ref = held_out_bench.get("sigmoid", {})
        boot_sig = compute_paired_brier_bootstrap(
            y_test,
            p_raw,
            p_sig,
            endpoint=endpoint,
            comparator_method="sigmoid",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
            benchmark_delta_brier=b_sig_ref.get("delta_brier_vs_raw"),
            benchmark_delta_ci_95=b_sig_ref.get("delta_brier_ci_95"),
        )
        if boot_sig.reconciled_with_benchmark is not None:
            reconciliation_details["sigmoid_bootstrap_ci"] = bool(boot_sig.reconciled_with_benchmark)

        b_iso_ref = held_out_bench.get("isotonic", {})
        boot_iso = (
            compute_paired_brier_bootstrap(
                y_test,
                p_raw,
                p_iso,
                endpoint=endpoint,
                comparator_method="isotonic",
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
                benchmark_delta_brier=b_iso_ref.get("delta_brier_vs_raw"),
                benchmark_delta_ci_95=b_iso_ref.get("delta_brier_ci_95"),
            )
            if p_iso is not None
            else None
        )

        # Applicability-Domain Stratification
        if sample_rec.get("ad_scores") is not None and p_sig is not None:
            ad_report = compute_ad_stratified_diagnostics(
                y_true=y_test,
                y_prob_raw=p_raw,
                y_prob_sigmoid=p_sig,
                ad_scores=sample_rec["ad_scores"],
                endpoint=endpoint,
                training_corpus_size=int(sample_rec.get("training_corpus_size", 0)),
            )
        else:
            ad_report = EndpointADStratificationReport(
                endpoint=endpoint,
                status="UNAVAILABLE",
                status_message="Endpoint applicability-domain fingerprint corpus (`training_fingerprints.npz`) unavailable.",
                is_exploratory=True,
                stratification_method="EXPLORATORY_EMPIRICAL_TERTILES",
                domain_thresholds_validated=False,
                min_positives_support_rule=MIN_POSITIVES_FOR_STRATUM_SUPPORT,
                n_total_test=int(len(y_test)),
                n_valid_ad_scores=0,
                n_unavailable_ad_scores=int(len(y_test)),
                training_corpus_size=0,
                quantile_edges=[],
                strata=[],
                unavailable_stratum=None,
                scientific_disclaimer=AD_STRATIFICATION_DISCLAIMER,
            )

    else:
        # Fallback to read-only benchmark artifact tables if raw dataset/model is absent
        sample_status = f"BENCHMARK_ONLY ({sample_rec.get('status', 'UNAVAILABLE')})"
        n_test = int(data_summary.get("test_count", 0))
        n_pos = int(data_summary.get("test_positive_count", 0))
        n_neg = int(data_summary.get("test_negative_count", max(0, n_test - n_pos)))
        prev = round(float(n_pos / n_test), 4) if n_test > 0 else 0.0

        for m_key in ("uncalibrated", "sigmoid", "isotonic"):
            b_m = held_out_bench.get(m_key)
            if not isinstance(b_m, dict):
                continue
            curve_d = b_m.get("calibration_curve_10bin")
            is_eval_only = m_key == "isotonic" and rec_method != "isotonic"
            rel_series = (
                build_reliability_series_from_benchmark_dict(
                    curve_d,
                    n_valid=n_test,
                    n_positive=n_pos,
                    n_negative=n_neg,
                    series_name=m_key,
                    is_evaluation_only=is_eval_only,
                )
                if isinstance(curve_d, dict)
                else None
            )
            if rel_series is not None:
                reliability_series[m_key] = rel_series
                gaps_10 = [b.abs_calibration_gap for b in rel_series.bins if b.abs_calibration_gap is not None]
                mace_10 = round(float(max(gaps_10)), 4) if gaps_10 else None
                mace_10_mean = round(float(np.mean(gaps_10)), 4) if gaps_10 else None
            else:
                mace_10 = None
                mace_10_mean = None

            series_metrics[m_key] = DiagnosticMethodMetrics(
                method=m_key,
                display_label=rel_series.display_label if rel_series else m_key,
                calibration_status="EVALUATION_ONLY" if is_eval_only else cal_status,
                is_deployed_method=(m_key == rec_method),
                is_evaluation_only=is_eval_only,
                n_valid=n_test,
                n_positive=n_pos,
                n_negative=n_neg,
                prevalence=prev,
                is_single_class=(n_pos == 0 or n_neg == 0),
                single_class_reason="Single-class test split." if (n_pos == 0 or n_neg == 0) else None,
                brier_score=b_m.get("brier_score"),
                log_loss_value=b_m.get("log_loss_value"),
                ece_10bin_weighted=b_m.get("ece_10bin_weighted"),
                mace_10bin_max=mace_10,
                mace_5bin_max=None,
                mace_5bin_unweighted_mean=b_m.get("mace_5bin_unweighted"),
                roc_auc=b_m.get("roc_auc") if (n_pos > 0 and n_neg > 0) else None,
                pr_auc=b_m.get("average_precision") if (n_pos > 0 and n_neg > 0) else None,
                pr_auc_definition=PR_AUC_DEFINITION,
                ece_definition=ECE_DEFINITION,
                mace_definition=MACE_DEFINITION,
                reliability_diagram_10bin=rel_series,
                mace_10bin_unweighted_mean=mace_10_mean,
            )

        b_raw_m = held_out_bench.get("uncalibrated", {})
        b_sig_m = held_out_bench.get("sigmoid", {})
        delta_val = b_sig_m.get("delta_brier_vs_raw")
        delta_ci = b_sig_m.get("delta_brier_ci_95")
        if delta_val is not None and isinstance(delta_ci, list) and len(delta_ci) == 2:
            imp_val = round(-float(delta_val), 6)
            imp_ci = [round(-float(delta_ci[1]), 6), round(-float(delta_ci[0]), 6)]
            ci_zero = bool(imp_ci[0] <= 0.0 <= imp_ci[1])
            boot_sig = PairedBootstrapResult(
                endpoint=endpoint,
                comparator_method="sigmoid",
                status="BENCHMARK_PRECOMPUTED",
                status_message="Loaded from precomputed benchmark artifact (sample-level dataset not loaded).",
                n_samples=n_test,
                n_positive=n_pos,
                n_negative=n_neg,
                n_requested_replicates=int(ref_art.get("provenance", {}).get("bootstrap_replicates", 500)),
                n_valid_replicates=int(ref_art.get("provenance", {}).get("bootstrap_replicates", 500)),
                n_two_class_replicates=int(ref_art.get("provenance", {}).get("bootstrap_replicates", 500)),
                n_degenerate_replicates=0,
                random_seed=int(ref_art.get("provenance", {}).get("random_seed", 42)),
                raw_brier=b_raw_m.get("brier_score"),
                raw_brier_ci_95=b_raw_m.get("brier_ci_95"),
                calibrated_brier=b_sig_m.get("brier_score"),
                calibrated_brier_ci_95=b_sig_m.get("brier_ci_95"),
                brier_improvement=imp_val,
                brier_improvement_ci_95=imp_ci,
                delta_brier_vs_raw=float(delta_val),
                delta_brier_ci_95=[float(delta_ci[0]), float(delta_ci[1])],
                ci_includes_zero=ci_zero,
                reconciled_with_benchmark=True,
                interpretation_note=(
                    "Loaded from precomputed 500-replicate paired bootstrap benchmark."
                ),
            )
        else:
            boot_sig = compute_paired_brier_bootstrap(None, None, None, endpoint=endpoint)

        boot_iso = None
        ad_report = EndpointADStratificationReport(
            endpoint=endpoint,
            status="UNAVAILABLE",
            status_message=(
                f"Sample-level test reconstruction unavailable ({sample_rec.get('error', 'missing dataset/model')}); "
                "AD stratification requires verified sample-level records and `training_fingerprints.npz`."
            ),
            is_exploratory=True,
            stratification_method="EXPLORATORY_EMPIRICAL_TERTILES",
            domain_thresholds_validated=False,
            min_positives_support_rule=MIN_POSITIVES_FOR_STRATUM_SUPPORT,
            n_total_test=n_test,
            n_valid_ad_scores=0,
            n_unavailable_ad_scores=n_test,
            training_corpus_size=0,
            quantile_edges=[],
            strata=[],
            unavailable_stratum=None,
            scientific_disclaimer=AD_STRATIFICATION_DISCLAIMER,
        )

    all_reconciled = bool(reconciliation_details) and all(reconciliation_details.values())

    bundle = EndpointReliabilityDiagnosticBundle(
        endpoint=endpoint,
        display_name=ep_meta["name"],
        pathway_family=ep_meta["family"],
        selector_label=ep_meta["selector_label"],
        overall_status="AVAILABLE" if series_metrics else "UNAVAILABLE",
        status_message=None if series_metrics else "No diagnostic metrics available.",
        calibration_artifact_status=cal_status,
        benchmark_status=bench_status,
        sample_level_status=sample_status,
        reconciled_with_benchmark=all_reconciled,
        reconciliation_details=reconciliation_details,
        recommended_method=rec_method,
        selection_rationale=rationale,
        isotonic_selection_eligible=iso_eligible,
        data_summary=data_summary,
        score_distribution_comparison=score_dist,
        series_metrics=series_metrics,
        reliability_series=reliability_series,
        bootstrap_sigmoid=boot_sig,
        bootstrap_isotonic=boot_iso,
        ad_stratification=ad_report,
        scientific_disclaimer=TOX21_CALIBRATION_DISCLAIMER,
    )
    _ENDPOINT_BUNDLE_CACHE[cache_key] = bundle
    return bundle
