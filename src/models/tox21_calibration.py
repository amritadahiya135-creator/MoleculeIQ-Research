"""Tox21 12-Endpoint Probability Calibration & Reliability Benchmarking Engine (Phase B.1).

Implements leakage-safe post-hoc probability calibration (Sigmoid/Platt scaling and
Isotonic regression) for the 12 Tox21 Random Forest classifiers:
  1. Method selection uses a deterministic 5-fold 3-way disjoint split strictly within
     the 80% training partition (`base_rf_idx` [60%] ⊥ `cal_fit_idx` [20%] ⊥ `val_idx` [20%]),
     ensuring zero label or score coupling between base-estimator training, calibrator
     fitting, and validation evaluation, with a conservative Sigmoid fallback when
     per-fold minority positive support is below the isotonic stability threshold.
  2. Final calibrator parameters are fitted on 5-fold stratified out-of-fold (OOF)
     predictions across the 80% training partition (`X_train`, `y_train`).
  3. Final benchmark evaluation is executed once on the locked 20% test partition
     (`X_test`, `y_test`), which is never accessed during selection or fitting.
  4. Deployed base models, raw probabilities, binary activity flags, applicability-domain
     outputs, and composite candidate rankings remain 100% unchanged by default.
"""

import os
import json
import time
import hashlib
import logging
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import sklearn
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.models.tox21 import TOX21_ENDPOINTS
from src.config import TOX21_DATA_PATH, TOX21_ARTIFACT_DIR, resolve_path
from src.utils.integrity import compute_file_sha256

logger = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION = "1.1.0"
METHOD_SELECTION_RULE_VERSION = "1.1.0_3way_disjoint_with_sigmoid_fallback"
PROB_CLIP_EPS = 1e-15

# Prespecified eligibility thresholds for Isotonic Regression:
# - Minimum positives in any calibration-fitting subset to fit a 1D calibrator at all
MIN_POSITIVES_FOR_CAL_FIT = 10
# - Minimum total training positives and prevalence required to consider Isotonic
MIN_POSITIVES_FOR_ISOTONIC_SELECTION = 250
MIN_PREVALENCE_FOR_ISOTONIC_SELECTION = 0.05
# - Minimum positives required inside EACH disjoint 20% calibration-fitting fold (`cal_fit_idx`)
#   to prevent non-parametric step-function overfitting during 3-way method selection
MIN_CAL_FIT_POSITIVES_FOR_ISOTONIC = 200

TOX21_CALIBRATION_DISCLAIMER = (
    "Calibrated probabilities estimate empirical Tox21 in-vitro assay activity frequency "
    "under the benchmark evaluation protocol; they do not represent clinical toxicity risk, "
    "in-vivo human safety, or regulatory suitability."
)


@dataclass(frozen=True)
class CalibrationCurveData:
    """Binned reliability curve data for a single endpoint and method."""
    n_bins: int
    bin_strategy: str
    bin_edges: List[float]
    bin_centers: List[float]
    observed_frequency: List[Optional[float]]
    mean_predicted_probability: List[Optional[float]]
    bin_counts: List[int]


@dataclass(frozen=True)
class ProbabilityMethodMetrics:
    """Comprehensive probability, discrimination, and classification metrics."""
    method: str
    brier_score: float
    log_loss_value: float
    ece_10bin_weighted: float
    mace_5bin_unweighted: float
    roc_auc: float
    average_precision: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    confusion_matrix: List[List[int]]
    calibration_curve_10bin: CalibrationCurveData
    brier_ci_95: Optional[List[float]] = None
    delta_brier_vs_raw: Optional[float] = None
    delta_brier_ci_95: Optional[List[float]] = None


def _is_valid_sha256_hex(value: Any) -> bool:
    """Check whether `value` is a valid 64-character hexadecimal SHA-256 digest."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def _hash_index_array(indices: np.ndarray) -> str:
    """Compute a deterministic SHA-256 hex digest of a sorted 1D integer index array."""
    arr = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _validate_probability_array(probs: Any, context: str = "probability") -> np.ndarray:
    """
    Validate that `probs` contains only finite numeric values strictly within [0.0, 1.0].
    Rejects booleans, NaNs, infinities, and out-of-range values.
    """
    if isinstance(probs, (bool, np.bool_)):
        raise ValueError(f"Boolean input is not a valid {context}.")
    arr = np.asarray(probs)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Non-numeric dtype ({arr.dtype}) encountered in {context}.")
    p_arr = arr.astype(np.float64)
    if p_arr.size == 0:
        raise ValueError(f"Empty array encountered in {context}.")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError(f"Non-finite (NaN or inf) values encountered in {context}.")
    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        raise ValueError(f"Out-of-bounds values outside [0.0, 1.0] encountered in {context}.")
    return p_arr


def validate_sigmoid_params(params: Any) -> Tuple[float, float]:
    """
    Validate serialized Sigmoid (Platt) calibration parameters `a` and `b`.
    Enforces finite numeric values and strictly negative slope `a < 0.0` so that
    P(y=1|f) = 1 / (1 + exp(a*f + b)) is strictly monotonically increasing in `f`.
    """
    if not isinstance(params, dict):
        raise ValueError("Sigmoid parameters must be a dictionary.")
    if params.get("method") not in (None, "sigmoid"):
        raise ValueError(f"Invalid method '{params.get('method')}' in Sigmoid parameters.")
    if "a" not in params or "b" not in params:
        raise ValueError("Sigmoid parameters must contain keys 'a' and 'b'.")
    raw_a, raw_b = params["a"], params["b"]
    if isinstance(raw_a, (bool, np.bool_)) or isinstance(raw_b, (bool, np.bool_)):
        raise ValueError("Sigmoid parameters 'a' and 'b' cannot be booleans.")
    try:
        a = float(raw_a)
        b = float(raw_b)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric Sigmoid parameters: {exc}") from exc
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Non-finite (NaN or inf) Sigmoid parameters ('a', 'b').")
    if a >= 0.0:
        raise ValueError(
            f"Invalid Sigmoid slope a={a}: Platt scaling requires a < 0.0 to preserve monotonically increasing probability ordering."
        )
    return a, b


def validate_isotonic_params(params: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate serialized Isotonic regression threshold arrays `X_thresholds` and `y_thresholds`.
    Enforces non-empty equal lengths (>= 2), finiteness, [0.0, 1.0] bounds, strictly increasing
    `X_thresholds`, and monotonically non-decreasing `y_thresholds`.
    """
    if not isinstance(params, dict):
        raise ValueError("Isotonic parameters must be a dictionary.")
    if params.get("method") not in (None, "isotonic"):
        raise ValueError(f"Invalid method '{params.get('method')}' in Isotonic parameters.")
    if "X_thresholds" not in params or "y_thresholds" not in params:
        raise ValueError("Isotonic parameters must contain 'X_thresholds' and 'y_thresholds'.")

    x_raw = params["X_thresholds"]
    y_raw = params["y_thresholds"]
    if not isinstance(x_raw, (list, tuple, np.ndarray)) or not isinstance(y_raw, (list, tuple, np.ndarray)):
        raise ValueError("Isotonic threshold fields must be sequences.")

    x_arr = _validate_probability_array(x_raw, context="Isotonic X_thresholds")
    y_arr = _validate_probability_array(y_raw, context="Isotonic y_thresholds")

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("Isotonic threshold arrays must be 1-dimensional.")
    if len(x_arr) < 2 or len(x_arr) != len(y_arr):
        raise ValueError(
            f"Invalid Isotonic threshold lengths: len(X)={len(x_arr)}, len(y)={len(y_arr)} (require >= 2 and equal)."
        )
    if not np.all(np.diff(x_arr) > 0.0):
        raise ValueError("Isotonic X_thresholds must be strictly increasing.")
    if not np.all(np.diff(y_arr) >= -1e-12):
        raise ValueError("Isotonic y_thresholds must be monotonically non-decreasing.")

    return x_arr, y_arr


def extract_positive_class_probabilities(
    estimator: Any,
    X: np.ndarray,
    positive_label: int = 1,
) -> np.ndarray:
    """
    Safely extract positive-class (label=1) probability column from `estimator.predict_proba(X)`
    by explicitly inspecting `estimator.classes_`.
    """
    if not hasattr(estimator, "classes_"):
        raise ValueError("Estimator is missing 'classes_' attribute.")
    classes_list = list(estimator.classes_)
    if positive_label not in classes_list:
        raise ValueError(
            f"Positive class ({positive_label}) not found in estimator.classes_={classes_list}."
        )
    pos_idx = int(classes_list.index(positive_label))
    probs = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    if probs.ndim != 2 or pos_idx >= probs.shape[1]:
        raise ValueError("Invalid probability array shape from estimator.predict_proba(X).")
    return _validate_probability_array(probs[:, pos_idx], context="extracted positive-class probabilities")


def compute_ece_and_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, CalibrationCurveData]:
    """
    Compute sample-weighted Expected Calibration Error (ECE) and reliability curve bins.

    Binning Rule (`equal_width_closed_right_final`):
      - `n_bins` equal-width intervals partitioning [0.0, 1.0] via `np.linspace(0.0, 1.0, n_bins + 1)`.
      - Bins 0 .. (n_bins - 2) are half-open `[edge_b, edge_{b+1})`, and the final bin (n_bins - 1)
        is closed `[edge_{n_bins-1}, 1.0]` via `np.digitize(y_prob, bin_edges[1:-1], right=False)`.
      - Empty bins receive `observed_frequency=None`, `mean_predicted_probability=None`, and weight 0.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = _validate_probability_array(y_prob, context="ECE y_prob")
    n_total = len(y_true_arr)

    if n_total == 0 or len(y_prob_arr) != n_total:
        raise ValueError("Invalid or mismatched array lengths in compute_ece_and_curve.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist()

    # Assign each probability to [edge_i, edge_{i+1}), with 1.0 included in the last bin
    bin_indices = np.digitize(y_prob_arr, bin_edges[1:-1], right=False)

    obs_freq: List[Optional[float]] = []
    mean_pred: List[Optional[float]] = []
    counts: List[int] = []

    weighted_abs_diffs = 0.0
    non_empty_diffs: List[float] = []

    for b in range(n_bins):
        mask = (bin_indices == b)
        cnt = int(np.sum(mask))
        counts.append(cnt)
        if cnt > 0:
            o_bar = float(np.mean(y_true_arr[mask]))
            p_bar = float(np.mean(y_prob_arr[mask]))
            obs_freq.append(round(o_bar, 4))
            mean_pred.append(round(p_bar, 4))
            diff = abs(o_bar - p_bar)
            weighted_abs_diffs += (cnt / n_total) * diff
            non_empty_diffs.append(diff)
        else:
            obs_freq.append(None)
            mean_pred.append(None)

    ece_weighted = float(weighted_abs_diffs)
    mace_unweighted = float(np.mean(non_empty_diffs)) if non_empty_diffs else 0.0

    curve = CalibrationCurveData(
        n_bins=n_bins,
        bin_strategy="equal_width_closed_right_final",
        bin_edges=[round(float(e), 4) for e in bin_edges],
        bin_centers=[round(float(c), 4) for c in bin_centers],
        observed_frequency=obs_freq,
        mean_predicted_probability=mean_pred,
        bin_counts=counts,
    )
    return ece_weighted, mace_unweighted, curve


def evaluate_probability_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method_name: str,
    threshold: float = 0.5,
    y_prob_raw_baseline: Optional[np.ndarray] = None,
    n_bootstrap: int = 0,
    random_seed: int = 42,
) -> ProbabilityMethodMetrics:
    """
    Compute comprehensive probability, discrimination, and classification metrics.
    Paired bootstrap CIs resample individual test observations with replacement.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = _validate_probability_array(y_prob, context=f"{method_name} predictions")
    y_prob_clipped = np.clip(y_prob_arr, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    ll = float(log_loss(y_true_arr, y_prob_clipped, labels=[0, 1]))
    ece_10, _, curve_10 = compute_ece_and_curve(y_true_arr, y_prob_arr, n_bins=10)
    _, mace_5, _ = compute_ece_and_curve(y_true_arr, y_prob_arr, n_bins=5)

    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    ap = float(average_precision_score(y_true_arr, y_prob_arr))

    y_pred = (y_prob_arr >= threshold).astype(int)
    acc = float(accuracy_score(y_true_arr, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    raw_cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(raw_cm[0, 0]), int(raw_cm[0, 1]), int(raw_cm[1, 0]), int(raw_cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    cm_list = [[tn, fp], [fn, tp]]

    brier_ci = None
    delta_brier = None
    delta_brier_ci = None

    if y_prob_raw_baseline is not None:
        raw_arr = _validate_probability_array(y_prob_raw_baseline, context="raw baseline predictions")
        raw_brier = float(brier_score_loss(y_true_arr, raw_arr))
        delta_brier = round(brier - raw_brier, 6)

    if n_bootstrap > 0 and len(y_true_arr) >= 20:
        rng = np.random.default_rng(random_seed)
        n_samples = len(y_true_arr)
        boot_briers = []
        boot_deltas = []
        raw_arr_boot = (
            _validate_probability_array(y_prob_raw_baseline, context="bootstrap raw baseline")
            if y_prob_raw_baseline is not None
            else None
        )

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_samples, size=n_samples)
            b_val = float(brier_score_loss(y_true_arr[idx], y_prob_arr[idx]))
            boot_briers.append(b_val)
            if raw_arr_boot is not None:
                b_raw = float(brier_score_loss(y_true_arr[idx], raw_arr_boot[idx]))
                boot_deltas.append(b_val - b_raw)

        brier_ci = [
            round(float(np.percentile(boot_briers, 2.5)), 4),
            round(float(np.percentile(boot_briers, 97.5)), 4),
        ]
        if boot_deltas:
            delta_brier_ci = [
                round(float(np.percentile(boot_deltas, 2.5)), 6),
                round(float(np.percentile(boot_deltas, 97.5)), 6),
            ]

    return ProbabilityMethodMetrics(
        method=method_name,
        brier_score=round(brier, 4),
        log_loss_value=round(ll, 4),
        ece_10bin_weighted=round(ece_10, 4),
        mace_5bin_unweighted=round(mace_5, 4),
        roc_auc=round(roc_auc, 4),
        average_precision=round(ap, 4),
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        sensitivity=round(rec, 4),
        specificity=round(spec, 4),
        precision=round(prec, 4),
        f1=round(f1, 4),
        confusion_matrix=cm_list,
        calibration_curve_10bin=curve_10,
        brier_ci_95=brier_ci,
        delta_brier_vs_raw=delta_brier,
        delta_brier_ci_95=delta_brier_ci,
    )


def fit_sigmoid_calibrator(oof_probs: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Fit Platt sigmoid calibrator on 1D out-of-sample probabilities without class weighting.
    """
    p_arr = _validate_probability_array(oof_probs, context="Sigmoid fit probabilities")
    y_arr = np.asarray(y_train, dtype=int)
    if len(p_arr) != len(y_arr):
        raise ValueError("Length mismatch between probabilities and labels in Sigmoid fitting.")
    if len(np.unique(y_arr)) < 2:
        raise ValueError("Both binary classes (0 and 1) are required to fit Sigmoid calibrator.")

    cal = _SigmoidCalibration()
    cal.fit(p_arr, y_arr)
    params = {
        "method": "sigmoid",
        "a": float(cal.a_),
        "b": float(cal.b_),
    }
    validate_sigmoid_params(params)
    return params


def predict_with_sigmoid_params(probs: Union[float, np.ndarray], params: Dict[str, Any]) -> Union[float, np.ndarray]:
    """
    Apply serialized Sigmoid (Platt) parameters: P(y=1|f) = 1 / (1 + exp(a * f + b)).
    Validates finiteness and [0.0, 1.0] bounds on both parameters, inputs, and outputs.
    """
    a, b = validate_sigmoid_params(params)
    is_scalar = np.ndim(probs) == 0
    p_arr = _validate_probability_array(probs, context="Sigmoid input probability")

    logits = -(a * p_arr + b)
    raw_out = expit(logits)
    if not np.all(np.isfinite(raw_out)):
        raise ValueError("Sigmoid transformation produced non-finite output.")
    res = np.clip(raw_out, 0.0, 1.0)
    if not np.all(np.isfinite(res)) or np.any((res < 0.0) | (res > 1.0)):
        raise ValueError("Sigmoid output failed [0.0, 1.0] finiteness verification.")

    if is_scalar:
        return float(res.item())
    return res


def fit_isotonic_calibrator(oof_probs: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Fit monotonic Isotonic Regression calibrator on 1D out-of-sample probabilities.
    """
    p_arr = _validate_probability_array(oof_probs, context="Isotonic fit probabilities")
    y_arr = np.asarray(y_train, dtype=int)
    if len(p_arr) != len(y_arr):
        raise ValueError("Length mismatch between probabilities and labels in Isotonic fitting.")
    if len(np.unique(y_arr)) < 2:
        raise ValueError("Both binary classes (0 and 1) are required to fit Isotonic calibrator.")

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_arr, y_arr)
    params = {
        "method": "isotonic",
        "X_thresholds": [float(x) for x in iso.X_thresholds_],
        "y_thresholds": [float(y) for y in iso.y_thresholds_],
    }
    validate_isotonic_params(params)
    return params


def predict_with_isotonic_params(probs: Union[float, np.ndarray], params: Dict[str, Any]) -> Union[float, np.ndarray]:
    """
    Apply serialized Isotonic step-interpolation parameters with [0, 1] clipping.
    Validates threshold monotonicity, finiteness, and [0.0, 1.0] bounds.
    """
    x_thresh, y_thresh = validate_isotonic_params(params)
    is_scalar = np.ndim(probs) == 0
    p_arr = _validate_probability_array(probs, context="Isotonic input probability")

    raw_out = np.interp(p_arr, x_thresh, y_thresh)
    if not np.all(np.isfinite(raw_out)):
        raise ValueError("Isotonic interpolation produced non-finite output.")
    res = np.clip(raw_out, 0.0, 1.0)
    if not np.all(np.isfinite(res)) or np.any((res < 0.0) | (res > 1.0)):
        raise ValueError("Isotonic output failed [0.0, 1.0] finiteness verification.")

    if is_scalar:
        return float(res.item())
    return res


def apply_calibrator_params(probs: Union[float, np.ndarray], params: Dict[str, Any]) -> Union[float, np.ndarray]:
    """Dispatch probability calibration transformation based on serialized method parameters."""
    if not isinstance(params, dict):
        raise ValueError("Calibrator params must be a dictionary.")
    method = params.get("method", "uncalibrated")
    if method == "uncalibrated":
        is_scalar = np.ndim(probs) == 0
        p_arr = _validate_probability_array(probs, context="Uncalibrated input probability")
        return float(p_arr.item()) if is_scalar else p_arr
    elif method == "sigmoid":
        return predict_with_sigmoid_params(probs, params)
    elif method == "isotonic":
        return predict_with_isotonic_params(probs, params)
    else:
        raise ValueError(f"Unsupported calibration method: '{method}'")


def reconstruct_tox21_endpoint_split(
    df: pd.DataFrame,
    valid_mask_arr: np.ndarray,
    fps_list: List[Optional[np.ndarray]],
    endpoint: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Deterministically reconstruct the exact 80/20 stratified train/test split for `endpoint`
    matching `src/models/tox21.py` while tracking global dataframe row indices.
    """
    if endpoint not in df.columns:
        raise ValueError(f"Unknown Tox21 endpoint '{endpoint}'.")

    ep_series = df[endpoint]
    not_na_mask = ep_series.notna().values & np.asarray(valid_mask_arr, dtype=bool)
    ep_global_indices = np.where(not_na_mask)[0]

    X_ep = np.array([fps_list[i] for i in ep_global_indices], dtype=np.int8)
    y_ep = ep_series.values[not_na_mask].astype(int)
    rel_indices = np.arange(len(y_ep), dtype=np.int64)

    train_rel_idx, test_rel_idx, y_train, y_test = train_test_split(
        rel_indices,
        y_ep,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_ep,
    )

    if len(np.intersect1d(train_rel_idx, test_rel_idx)) > 0:
        raise RuntimeError(f"Train/test relative index overlap detected for '{endpoint}'!")

    train_global_idx = ep_global_indices[train_rel_idx]
    test_global_idx = ep_global_indices[test_rel_idx]

    if len(np.intersect1d(train_global_idx, test_global_idx)) > 0:
        raise RuntimeError(f"Train/test global index overlap detected for '{endpoint}'!")

    return {
        "X_ep": X_ep,
        "y_ep": y_ep,
        "X_train": X_ep[train_rel_idx],
        "X_test": X_ep[test_rel_idx],
        "y_train": y_train,
        "y_test": y_test,
        "train_rel_idx": train_rel_idx,
        "test_rel_idx": test_rel_idx,
        "train_global_idx": train_global_idx,
        "test_global_idx": test_global_idx,
    }


def validate_3way_disjoint_fold_indices(
    base_rf_idx: np.ndarray,
    cal_fit_idx: np.ndarray,
    val_idx: np.ndarray,
    n_train_samples: Optional[int] = None,
    train_global_idx: Optional[np.ndarray] = None,
    locked_test_global_idx: Optional[np.ndarray] = None,
) -> None:
    """
    Enforce pairwise index disjointness between:
      1. `base_rf_idx` (subset training the fold Random Forest)
      2. `cal_fit_idx` (subset fitting the fold calibrators)
      3. `val_idx`     (subset evaluating the fold calibrated predictions)
    and verify zero intersection with `locked_test_global_idx` if provided.
    """
    b_arr = np.asarray(base_rf_idx, dtype=np.int64)
    c_arr = np.asarray(cal_fit_idx, dtype=np.int64)
    v_arr = np.asarray(val_idx, dtype=np.int64)

    if b_arr.ndim != 1 or c_arr.ndim != 1 or v_arr.ndim != 1:
        raise ValueError("Fold index subsets must be 1D integer arrays.")
    if len(b_arr) == 0 or len(c_arr) == 0 or len(v_arr) == 0:
        raise ValueError("Fold index subsets (`base_rf_idx`, `cal_fit_idx`, `val_idx`) cannot be empty.")

    if len(np.unique(b_arr)) != len(b_arr):
        raise ValueError("Duplicate indices detected within `base_rf_idx`.")
    if len(np.unique(c_arr)) != len(c_arr):
        raise ValueError("Duplicate indices detected within `cal_fit_idx`.")
    if len(np.unique(v_arr)) != len(v_arr):
        raise ValueError("Duplicate indices detected within `val_idx`.")

    if len(np.intersect1d(b_arr, c_arr)) > 0:
        raise ValueError(
            "Fold independence violation: `base_rf_idx` overlaps with `cal_fit_idx` "
            "(calibrator fitting would use labels that trained the validation score generator)."
        )
    if len(np.intersect1d(b_arr, v_arr)) > 0:
        raise ValueError(
            "Fold independence violation: `base_rf_idx` overlaps with `val_idx` "
            "(base Random Forest trained on validation samples)."
        )
    if len(np.intersect1d(c_arr, v_arr)) > 0:
        raise ValueError(
            "Fold independence violation: `cal_fit_idx` overlaps with `val_idx` "
            "(calibrator fitted on validation samples)."
        )

    union_arr = np.concatenate([b_arr, c_arr, v_arr])
    if n_train_samples is not None:
        if np.any(union_arr < 0) or np.any(union_arr >= n_train_samples):
            raise ValueError("Fold indices out of bounds for training partition size.")

    if train_global_idx is not None and locked_test_global_idx is not None:
        tg_arr = np.asarray(train_global_idx, dtype=np.int64)
        test_g_arr = np.asarray(locked_test_global_idx, dtype=np.int64)
        used_global = tg_arr[union_arr]
        if len(np.intersect1d(used_global, test_g_arr)) > 0:
            raise ValueError(
                "Test-set leakage violation: locked test partition indices entered calibration selection/fitting!"
            )


def build_3way_disjoint_splits(
    y_train: np.ndarray,
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Construct `n_splits` deterministic 3-way mutually disjoint stratified fold assignments
    from `y_train`:
      - `val_idx`:     1 fold  (1 / n_splits of D_train)
      - `cal_fit_idx`: 1 fold  (1 / n_splits of D_train)
      - `base_rf_idx`: (n_splits - 2) folds ((n_splits - 2) / n_splits of D_train)
    """
    if n_splits < 3:
        raise ValueError("3-way disjoint splitting requires `n_splits >= 3` (recommended: 5).")

    y_arr = np.asarray(y_train, dtype=int)
    n_samples = len(y_arr)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))

    if n_pos < n_splits or n_neg < n_splits:
        raise ValueError(
            f"Single-class or insufficient class counts (pos={n_pos}, neg={n_neg}) for {n_splits}-fold stratified split."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    dummy_X = np.zeros((n_samples, 1), dtype=np.int8)
    raw_folds: List[np.ndarray] = [val_f for _, val_f in skf.split(dummy_X, y_arr)]

    splits: List[Dict[str, np.ndarray]] = []
    for k in range(n_splits):
        val_idx = raw_folds[k]
        cal_fit_idx = raw_folds[(k + 1) % n_splits]
        base_rf_idx = np.concatenate([raw_folds[(k + m) % n_splits] for m in range(2, n_splits)])

        validate_3way_disjoint_fold_indices(
            base_rf_idx=base_rf_idx,
            cal_fit_idx=cal_fit_idx,
            val_idx=val_idx,
            n_train_samples=n_samples,
        )
        splits.append(
            {
                "fold": k,
                "base_rf_idx": base_rf_idx,
                "cal_fit_idx": cal_fit_idx,
                "val_idx": val_idx,
            }
        )
    return splits


def select_calibration_method_via_3way_disjoint_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    n_estimators: int = 300,
    class_weight: str = "balanced",
    random_seed: int = 42,
    custom_splits: Optional[List[Dict[str, np.ndarray]]] = None,
    train_global_idx: Optional[np.ndarray] = None,
    locked_test_global_idx: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Leakage-free 3-way disjoint calibration method selection strictly within `(X_train, y_train)`.

    Why this protocol is scientifically valid:
      In each fold rotation k in {0 .. n_splits-1}, `D_train` is partitioned into three
      pairwise disjoint subsets (`base_rf_idx` ⊥ `cal_fit_idx` ⊥ `val_idx`):
        1. `M_base^(k)` is trained ONLY on `(X_train[base_rf_idx], y_train[base_rf_idx])`.
        2. Candidate calibrators (`C_sig^(k)`, `C_iso^(k)`) are fitted ONLY on out-of-sample
           predictions `(M_base^(k)(X_train[cal_fit_idx]), y_train[cal_fit_idx])`.
        3. Calibrated predictions `C^(k)(M_base^(k)(X_train[val_idx]))` are evaluated ONLY on
           the untouched validation subset `y_train[val_idx]`.
      Consequently:
        - `M_base^(k)` never trains on labels used to fit `C^(k)`.
        - `C^(k)` never fits on scores or labels influenced by `val_idx`.
        - Both `cal_fit_idx` and `val_idx` receive genuine out-of-sample predictions from the
          exact same estimator `M_base^(k)`.
        - Locked test samples (`locked_test_global_idx`) never enter any subset.
    """
    y_arr = np.asarray(y_train, dtype=int)
    n_train = len(y_arr)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    prevalence = float(n_pos / n_train) if n_train > 0 else 0.0

    splits = (
        custom_splits
        if custom_splits is not None
        else build_3way_disjoint_splits(y_arr, n_splits=n_splits, random_seed=random_seed)
    )

    if len(splits) == 0:
        raise ValueError("No fold splits provided for 3-way disjoint method selection.")

    X_train_f32 = np.ascontiguousarray(X_train, dtype=np.float32)
    val_raw_probs = np.full(n_train, np.nan, dtype=np.float64)
    val_sig_probs = np.full(n_train, np.nan, dtype=np.float64)
    val_iso_probs = np.full(n_train, np.nan, dtype=np.float64)

    fold_provenance: List[Dict[str, Any]] = []

    for fold_idx, split_dict in enumerate(splits):
        base_rf_idx = np.asarray(split_dict["base_rf_idx"], dtype=np.int64)
        cal_fit_idx = np.asarray(split_dict["cal_fit_idx"], dtype=np.int64)
        val_idx = np.asarray(split_dict["val_idx"], dtype=np.int64)

        # Enforce strict 3-way disjointness and test-set isolation at runtime
        validate_3way_disjoint_fold_indices(
            base_rf_idx=base_rf_idx,
            cal_fit_idx=cal_fit_idx,
            val_idx=val_idx,
            n_train_samples=n_train,
            train_global_idx=train_global_idx,
            locked_test_global_idx=locked_test_global_idx,
        )

        y_base = y_arr[base_rf_idx]
        y_cal = y_arr[cal_fit_idx]
        y_val = y_arr[val_idx]

        if len(np.unique(y_base)) < 2 or len(np.unique(y_cal)) < 2 or len(np.unique(y_val)) < 2:
            raise ValueError(f"Single-class fold encountered in 3-way split fold {fold_idx}.")

        fold_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_seed,
            n_jobs=-1,
        )
        fold_clf.fit(X_train_f32[base_rf_idx], y_base)

        p_cal_raw = extract_positive_class_probabilities(fold_clf, X_train_f32[cal_fit_idx])
        p_val_raw = extract_positive_class_probabilities(fold_clf, X_train_f32[val_idx])

        sig_params_k = fit_sigmoid_calibrator(p_cal_raw, y_cal)
        iso_params_k = fit_isotonic_calibrator(p_cal_raw, y_cal)

        val_raw_probs[val_idx] = p_val_raw
        val_sig_probs[val_idx] = predict_with_sigmoid_params(p_val_raw, sig_params_k)
        val_iso_probs[val_idx] = predict_with_isotonic_params(p_val_raw, iso_params_k)

        fold_provenance.append(
            {
                "fold": int(fold_idx),
                "base_rf_count": int(len(base_rf_idx)),
                "base_rf_positive_count": int(np.sum(y_base == 1)),
                "base_rf_negative_count": int(np.sum(y_base == 0)),
                "base_rf_indices_sha256": _hash_index_array(base_rf_idx),
                "cal_fit_count": int(len(cal_fit_idx)),
                "cal_fit_positive_count": int(np.sum(y_cal == 1)),
                "cal_fit_negative_count": int(np.sum(y_cal == 0)),
                "cal_fit_indices_sha256": _hash_index_array(cal_fit_idx),
                "val_count": int(len(val_idx)),
                "val_positive_count": int(np.sum(y_val == 1)),
                "val_negative_count": int(np.sum(y_val == 0)),
                "val_indices_sha256": _hash_index_array(val_idx),
                "pairwise_disjoint_verified": True,
            }
        )

    if np.any(np.isnan(val_raw_probs)) or np.any(np.isnan(val_sig_probs)) or np.any(np.isnan(val_iso_probs)):
        raise RuntimeError("Incomplete 3-way disjoint validation coverage: NaN values remain.")

    raw_3way_metrics = evaluate_probability_predictions(y_arr, val_raw_probs, "uncalibrated")
    sig_3way_metrics = evaluate_probability_predictions(
        y_arr, val_sig_probs, "sigmoid", y_prob_raw_baseline=val_raw_probs
    )
    iso_3way_metrics = evaluate_probability_predictions(
        y_arr, val_iso_probs, "isotonic", y_prob_raw_baseline=val_raw_probs
    )

    min_cal_fit_pos = min(f["cal_fit_positive_count"] for f in fold_provenance)
    isotonic_train_eligible = (
        n_pos >= MIN_POSITIVES_FOR_ISOTONIC_SELECTION
        and prevalence >= MIN_PREVALENCE_FOR_ISOTONIC_SELECTION
    )
    isotonic_fold_support_sufficient = min_cal_fit_pos >= MIN_CAL_FIT_POSITIVES_FOR_ISOTONIC
    isotonic_selection_eligible = bool(isotonic_train_eligible and isotonic_fold_support_sufficient)

    # Prespecified 3-way disjoint decision criteria:
    sig_improves = (
        sig_3way_metrics.brier_score <= raw_3way_metrics.brier_score - 0.0005
        and sig_3way_metrics.ece_10bin_weighted <= raw_3way_metrics.ece_10bin_weighted - 0.001
        and sig_3way_metrics.log_loss_value <= raw_3way_metrics.log_loss_value
    )
    iso_improves = (
        isotonic_selection_eligible
        and iso_3way_metrics.brier_score <= raw_3way_metrics.brier_score - 0.0005
        and iso_3way_metrics.ece_10bin_weighted <= raw_3way_metrics.ece_10bin_weighted - 0.001
        and iso_3way_metrics.log_loss_value <= raw_3way_metrics.log_loss_value
        and iso_3way_metrics.roc_auc >= sig_3way_metrics.roc_auc
        and iso_3way_metrics.average_precision >= sig_3way_metrics.average_precision
    )

    if sig_improves and iso_improves:
        if (
            iso_3way_metrics.brier_score <= sig_3way_metrics.brier_score - 0.0005
            and iso_3way_metrics.log_loss_value < sig_3way_metrics.log_loss_value
            and iso_3way_metrics.ece_10bin_weighted <= sig_3way_metrics.ece_10bin_weighted
        ):
            selected_method = "isotonic"
            rationale = (
                f"Isotonic selected via 3-way disjoint CV: sufficient per-fold minority support "
                f"(min_cal_fit_pos={min_cal_fit_pos}, total_train_pos={n_pos}, prev={prevalence:.2%}) "
                f"and superior 3-way validation Brier ({iso_3way_metrics.brier_score:.4f} vs "
                f"{sig_3way_metrics.brier_score:.4f}) and LogLoss ({iso_3way_metrics.log_loss_value:.4f} vs "
                f"{sig_3way_metrics.log_loss_value:.4f}) without ranking degradation."
            )
        else:
            selected_method = "sigmoid"
            rationale = (
                f"Sigmoid (Platt) selected via 3-way disjoint CV: reduces validation Brier from "
                f"{raw_3way_metrics.brier_score:.4f} to {sig_3way_metrics.brier_score:.4f}, LogLoss from "
                f"{raw_3way_metrics.log_loss_value:.4f} to {sig_3way_metrics.log_loss_value:.4f}, and ECE from "
                f"{raw_3way_metrics.ece_10bin_weighted:.4f} to {sig_3way_metrics.ece_10bin_weighted:.4f}; "
                f"preferred over non-parametric isotonic regression."
            )
    elif sig_improves:
        selected_method = "sigmoid"
        if not isotonic_train_eligible:
            iso_note = (
                f" Isotonic excluded due to low overall training minority support "
                f"(train_pos={n_pos} < {MIN_POSITIVES_FOR_ISOTONIC_SELECTION} or prev={prevalence:.2%} < 5.0%)."
            )
        elif not isotonic_fold_support_sufficient:
            iso_note = (
                f" Conservative Sigmoid fallback applied for method selection: per-fold disjoint "
                f"calibration-fitting subset has {min_cal_fit_pos} positives (< {MIN_CAL_FIT_POSITIVES_FOR_ISOTONIC} "
                f"required for stable non-parametric step-function selection; 3-way Iso LogLoss="
                f"{iso_3way_metrics.log_loss_value:.4f} vs Sig LogLoss={sig_3way_metrics.log_loss_value:.4f}, "
                f"3-way Iso PR-AUC={iso_3way_metrics.average_precision:.4f} vs Sig PR-AUC="
                f"{sig_3way_metrics.average_precision:.4f})."
            )
        else:
            iso_note = ""
        rationale = (
            f"Sigmoid (Platt) selected via 3-way disjoint CV: reduces validation Brier from "
            f"{raw_3way_metrics.brier_score:.4f} to {sig_3way_metrics.brier_score:.4f}, LogLoss from "
            f"{raw_3way_metrics.log_loss_value:.4f} to {sig_3way_metrics.log_loss_value:.4f}, and ECE from "
            f"{raw_3way_metrics.ece_10bin_weighted:.4f} to {sig_3way_metrics.ece_10bin_weighted:.4f}."
            + iso_note
        )
    elif iso_improves:
        selected_method = "isotonic"
        rationale = (
            f"Isotonic selected via 3-way disjoint CV: reduces validation Brier from "
            f"{raw_3way_metrics.brier_score:.4f} to {iso_3way_metrics.brier_score:.4f}."
        )
    else:
        selected_method = "uncalibrated"
        rationale = (
            "Uncalibrated retained: neither Sigmoid nor Isotonic achieved simultaneous "
            "3-way disjoint validation Brier, LogLoss, and ECE improvement."
        )

    return {
        "selected_method": selected_method,
        "selection_rationale": rationale,
        "method_selection_rule": METHOD_SELECTION_RULE_VERSION,
        "isotonic_train_eligible": bool(isotonic_train_eligible),
        "isotonic_fold_support_sufficient": bool(isotonic_fold_support_sufficient),
        "isotonic_selection_eligible": bool(isotonic_selection_eligible),
        "min_cal_fit_positives": int(min_cal_fit_pos),
        "fold_provenance": fold_provenance,
        "three_way_disjoint_cv_metrics": {
            "uncalibrated": asdict(raw_3way_metrics),
            "sigmoid": asdict(sig_3way_metrics),
            "isotonic": asdict(iso_3way_metrics),
        },
        "val_raw_probs_60pct": val_raw_probs,
    }


def generate_leakage_safe_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    n_estimators: int = 300,
    class_weight: str = "balanced",
    random_seed: int = 42,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate 5-fold stratified out-of-fold (OOF) probability predictions strictly within `X_train`
    for fitting the final endpoint calibrator parameters (`sigmoid_params`, `isotonic_params`).

    Every training observation receives a probability predicted by a RandomForestClassifier
    trained exclusively on the remaining (n_splits - 1) folds (80% of `X_train`).
    """
    y_arr = np.asarray(y_train, dtype=int)
    n_samples = len(y_arr)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))

    if n_pos < n_splits or n_neg < n_splits:
        raise ValueError(
            f"Single-class or insufficient class counts (pos={n_pos}, neg={n_neg}) for {n_splits}-fold stratified CV."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_probs = np.full(n_samples, np.nan, dtype=np.float64)
    fold_indices: List[Tuple[np.ndarray, np.ndarray]] = []
    X_train_f32 = np.ascontiguousarray(X_train, dtype=np.float32)

    for fold_idx, (fit_idx, val_idx) in enumerate(skf.split(X_train_f32, y_arr)):
        if len(np.intersect1d(fit_idx, val_idx)) > 0:
            raise RuntimeError(f"Fold leakage detected in OOF fold {fold_idx}!")

        fold_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_seed,
            n_jobs=-1,
        )
        fold_clf.fit(X_train_f32[fit_idx], y_arr[fit_idx])
        oof_probs[val_idx] = extract_positive_class_probabilities(fold_clf, X_train_f32[val_idx])
        fold_indices.append((fit_idx, val_idx))

    if np.any(np.isnan(oof_probs)):
        raise RuntimeError("Incomplete OOF probability generation: NaN values remain.")

    return oof_probs, fold_indices


def compute_score_distribution_comparison(
    probs_60pct_train: np.ndarray,
    probs_80pct_oof_train: np.ndarray,
    probs_100pct_deployed_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Quantify differences in raw probability distributions across base Random Forests trained on:
      - 60% of D_train (3-way disjoint method-selection folds)
      - 80% of D_train (5-fold OOF final calibrator-fitting folds)
      - 100% of D_train (deployed production model evaluated on locked D_test)
    """
    def _summary_stats(arr: np.ndarray) -> Dict[str, float]:
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": round(float(np.mean(a)), 4),
            "std": round(float(np.std(a)), 4),
            "median": round(float(np.median(a)), 4),
            "p10": round(float(np.percentile(a, 10)), 4),
            "p90": round(float(np.percentile(a, 90)), 4),
            "fraction_ge_0_5": round(float(np.mean(a >= 0.5)), 4),
        }

    p60 = np.asarray(probs_60pct_train, dtype=np.float64)
    p80 = np.asarray(probs_80pct_oof_train, dtype=np.float64)
    mad_60_vs_80 = float(np.mean(np.abs(p60 - p80)))
    corr_60_vs_80 = float(np.corrcoef(p60, p80)[0, 1]) if np.std(p60) > 0 and np.std(p80) > 0 else 1.0

    return {
        "base_rf_60pct_train_3way_val": _summary_stats(p60),
        "base_rf_80pct_train_5fold_oof": _summary_stats(p80),
        "base_rf_100pct_deployed_test": _summary_stats(probs_100pct_deployed_test),
        "train_60pct_vs_80pct_mean_abs_diff": round(mad_60_vs_80, 4),
        "train_60pct_vs_80pct_pearson_r": round(corr_60_vs_80, 4),
        "note": (
            "Base models in 3-way disjoint selection are trained on 60% of D_train (3/5 folds), "
            "final-fit OOF base models are trained on 80% of D_train (4/5 folds), and the deployed "
            "model is trained on 100% of D_train. While stratified folds preserve class ratios, "
            "smaller training subsets exhibit slightly higher finite-sample leaf variance."
        ),
    }


def benchmark_and_persist_tox21_calibration(
    csv_path: Optional[Union[str, Path]] = None,
    artifact_dir: Optional[Union[str, Path]] = None,
    endpoints: Optional[List[str]] = None,
    n_splits: int = 5,
    n_bootstrap: int = 500,
    random_seed: int = 42,
    reuse_existing: bool = False,
) -> Dict[str, Any]:
    """
    Execute Phase B.1 end-to-end leakage-safe calibration benchmarking across all 12 Tox21 endpoints.

    Persists per-endpoint `{safe_ep}_calibration.json` artifacts and an aggregate
    `calibration_benchmark.json` report without modifying any existing model weights or metadata.
    """
    t0 = time.time()
    if endpoints is None:
        endpoints = TOX21_ENDPOINTS

    resolved_csv = resolve_path(csv_path or TOX21_DATA_PATH)
    resolved_dir = resolve_path(artifact_dir or TOX21_ARTIFACT_DIR)
    os.makedirs(resolved_dir, exist_ok=True)

    dataset_sha256 = compute_file_sha256(resolved_csv)
    df = pd.read_csv(resolved_csv)
    total_raw_rows = len(df)

    # Pre-validate and featurize all valid structures identically to src/models/tox21.py
    valid_mask = []
    fps_list = []
    for smi in df["smiles"]:
        val_res = validate_smiles(str(smi))
        if not val_res.is_valid or not val_res.canonical_smiles:
            valid_mask.append(False)
            fps_list.append(None)
        else:
            valid_mask.append(True)
            fp_res = generate_morgan_fingerprint(
                val_res.canonical_smiles, radius=2, n_bits=2048, return_bit_info=False
            )
            fps_list.append(fp_res.numpy_vector)
    valid_mask_arr = np.array(valid_mask, dtype=bool)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    endpoint_reports: Dict[str, Any] = {}

    for ep in endpoints:
        ep_t0 = time.time()
        safe_name = ep.replace("-", "_")
        model_path = os.path.join(resolved_dir, f"{safe_name}_model.joblib")
        meta_path = os.path.join(resolved_dir, f"{safe_name}_metadata.json")
        cal_path = os.path.join(resolved_dir, f"{safe_name}_calibration.json")

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing deployed model or metadata for endpoint '{ep}'.")

        base_model_sha256 = compute_file_sha256(model_path)

        if reuse_existing and os.path.exists(cal_path):
            try:
                with open(cal_path, "r", encoding="utf-8") as f:
                    existing_payload = json.load(f)
                prov = existing_payload.get("provenance", {})
                if (
                    existing_payload.get("schema_version") == CALIBRATION_SCHEMA_VERSION
                    and prov.get("base_model_sha256") == base_model_sha256
                    and prov.get("dataset_sha256") == dataset_sha256
                ):
                    endpoint_reports[ep] = existing_payload
                    logger.info(f"Reused verified calibration artifact for '{ep}'.")
                    continue
            except Exception:
                pass

        base_model = joblib.load(model_path)
        if hasattr(base_model, "n_jobs"):
            base_model.n_jobs = 1
        with open(meta_path, "r", encoding="utf-8") as f:
            base_meta = json.load(f)

        classes_list = list(base_model.classes_)
        if 1 not in classes_list:
            raise ValueError(f"Positive class 1 missing in model '{ep}'.")
        pos_idx = int(classes_list.index(1))

        # Stage 1: Reconstruct exact endpoint 80/20 partition and verify index disjointness
        split_info = reconstruct_tox21_endpoint_split(
            df=df,
            valid_mask_arr=valid_mask_arr,
            fps_list=fps_list,
            endpoint=ep,
            test_size=0.2,
            random_seed=random_seed,
        )
        X_ep = split_info["X_ep"]
        y_ep = split_info["y_ep"]
        X_train = split_info["X_train"]
        X_test = split_info["X_test"]
        y_train = split_info["y_train"]
        y_test = split_info["y_test"]
        train_global_idx = split_info["train_global_idx"]
        test_global_idx = split_info["test_global_idx"]

        n_est = int(base_meta.get("n_estimators", 300))
        c_weight = str(base_meta.get("class_weight", "balanced"))

        # Stage 2: Leakage-free 3-way disjoint method selection strictly on (X_train, y_train)
        selection_info = select_calibration_method_via_3way_disjoint_cv(
            X_train=X_train,
            y_train=y_train,
            n_splits=n_splits,
            n_estimators=n_est,
            class_weight=c_weight,
            random_seed=random_seed,
            train_global_idx=train_global_idx,
            locked_test_global_idx=test_global_idx,
        )

        # Stage 3: Generate 5-fold stratified OOF predictions on X_train for final calibrator fitting
        oof_probs, oof_fold_splits = generate_leakage_safe_oof_predictions(
            X_train=X_train,
            y_train=y_train,
            n_splits=n_splits,
            n_estimators=n_est,
            class_weight=c_weight,
            random_seed=random_seed,
        )
        for fit_rel, val_rel in oof_fold_splits:
            if len(np.intersect1d(train_global_idx[fit_rel], test_global_idx)) > 0:
                raise RuntimeError(f"OOF fit fold overlapped with test set for '{ep}'!")
            if len(np.intersect1d(train_global_idx[val_rel], test_global_idx)) > 0:
                raise RuntimeError(f"OOF val fold overlapped with test set for '{ep}'!")

        # Fit final Sigmoid and Isotonic calibrators on full 5-fold OOF predictions (zero test labels)
        sigmoid_params = fit_sigmoid_calibrator(oof_probs, y_train)
        isotonic_params = fit_isotonic_calibrator(oof_probs, y_train)

        # Stage 4: Single-pass evaluation on locked 20% held-out test partition
        y_test_raw = extract_positive_class_probabilities(base_model, X_test)
        y_test_sig = predict_with_sigmoid_params(y_test_raw, sigmoid_params)
        y_test_iso = predict_with_isotonic_params(y_test_raw, isotonic_params)

        test_raw_metrics = evaluate_probability_predictions(
            y_test, y_test_raw, "uncalibrated", n_bootstrap=n_bootstrap, random_seed=random_seed
        )
        test_sig_metrics = evaluate_probability_predictions(
            y_test,
            y_test_sig,
            "sigmoid",
            y_prob_raw_baseline=y_test_raw,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        test_iso_metrics = evaluate_probability_predictions(
            y_test,
            y_test_iso,
            "isotonic",
            y_prob_raw_baseline=y_test_raw,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        score_dist_comparison = compute_score_distribution_comparison(
            probs_60pct_train=selection_info["val_raw_probs_60pct"],
            probs_80pct_oof_train=oof_probs,
            probs_100pct_deployed_test=y_test_raw,
        )

        rec_method = selection_info["selected_method"]
        n_usable = int(len(y_ep))
        n_pos_total = int(np.sum(y_ep == 1))
        n_neg_total = int(np.sum(y_ep == 0))
        n_train_pos = int(np.sum(y_train == 1))
        n_test_pos = int(np.sum(y_test == 1))

        # Check whether 95% paired bootstrap CI for Delta Brier excludes zero
        sig_ci = test_sig_metrics.delta_brier_ci_95
        sig_brier_statistically_supported = bool(sig_ci is not None and sig_ci[1] < 0.0)

        artifact_payload = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "endpoint": ep,
            "created_at": now_iso,
            "provenance": {
                "base_model_filename": f"{safe_name}_model.joblib",
                "base_model_sha256": base_model_sha256,
                "dataset_path": "data/raw/tox21.csv",
                "dataset_sha256": dataset_sha256,
                "split_strategy": "stratified_random_80_20",
                "method_selection_protocol": f"{n_splits}_fold_3way_disjoint_base60_cal20_val20",
                "calibration_protocol": f"{n_splits}_fold_stratified_out_of_fold_posthoc",
                "random_seed": random_seed,
                "positive_class_label": 1,
                "positive_class_index": pos_idx,
                "train_global_indices_sha256": _hash_index_array(train_global_idx),
                "test_global_indices_sha256": _hash_index_array(test_global_idx),
                "train_test_disjoint_verified": True,
                "bootstrap_replicates": int(n_bootstrap),
                "bootstrap_resampling_unit": "individual_test_compound_paired_with_replacement",
                "ece_binning_rule": "10_equal_width_bins_0_to_1_closed_right_final",
                "library_versions": {
                    "scikit_learn": sklearn.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
            },
            "data_summary": {
                "total_raw_records": total_raw_rows,
                "missing_or_invalid_excluded": int(total_raw_rows - n_usable),
                "usable_samples": n_usable,
                "positive_count": n_pos_total,
                "negative_count": n_neg_total,
                "prevalence": round(float(n_pos_total / n_usable), 4),
                "train_count": int(len(y_train)),
                "train_positive_count": n_train_pos,
                "train_negative_count": int(len(y_train) - n_train_pos),
                "calibration_fit_count": int(len(y_train)),
                "calibration_fit_positive_count": n_train_pos,
                "calibration_fit_negative_count": int(len(y_train) - n_train_pos),
                "min_3way_cal_fit_positive_count": selection_info["min_cal_fit_positives"],
                "test_count": int(len(y_test)),
                "test_positive_count": n_test_pos,
                "test_negative_count": int(len(y_test) - n_test_pos),
            },
            "selection_governance": {
                "recommended_method": rec_method,
                "method_selection_rule": selection_info["method_selection_rule"],
                "selection_rationale": selection_info["selection_rationale"],
                "isotonic_train_eligible": selection_info["isotonic_train_eligible"],
                "isotonic_fold_support_sufficient": selection_info["isotonic_fold_support_sufficient"],
                "isotonic_selection_eligible": selection_info["isotonic_selection_eligible"],
                "test_brier_improvement_point_estimate": bool(
                    test_sig_metrics.delta_brier_vs_raw is not None
                    and test_sig_metrics.delta_brier_vs_raw < 0.0
                ),
                "test_brier_improvement_95ci_excludes_zero": sig_brier_statistically_supported,
                "default_production_behavior": "raw_uncalibrated_preserved",
                "opt_in_calibrated_available": True,
                "scientific_disclaimer": TOX21_CALIBRATION_DISCLAIMER,
            },
            "calibrator_parameters": {
                "sigmoid": sigmoid_params,
                "isotonic": isotonic_params,
            },
            "method_selection_fold_provenance": selection_info["fold_provenance"],
            "score_distribution_comparison": score_dist_comparison,
            "training_3way_disjoint_cv_metrics": selection_info["three_way_disjoint_cv_metrics"],
            "training_oof_cv_metrics": selection_info["three_way_disjoint_cv_metrics"],
            "held_out_test_metrics": {
                "uncalibrated": asdict(test_raw_metrics),
                "sigmoid": asdict(test_sig_metrics),
                "isotonic": asdict(test_iso_metrics),
            },
            "runtime_seconds": round(time.time() - ep_t0, 2),
        }

        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(artifact_payload, f, indent=2)

        endpoint_reports[ep] = artifact_payload
        logger.info(
            f"Calibrated '{ep}' ({artifact_payload['runtime_seconds']}s): "
            f"rec={rec_method} | 3-Way Val Brier Raw={selection_info['three_way_disjoint_cv_metrics']['uncalibrated']['brier_score']:.4f}, "
            f"Sig={selection_info['three_way_disjoint_cv_metrics']['sigmoid']['brier_score']:.4f}, "
            f"Iso={selection_info['three_way_disjoint_cv_metrics']['isotonic']['brier_score']:.4f} | "
            f"Test Brier Raw={test_raw_metrics.brier_score:.4f}, "
            f"Sig={test_sig_metrics.brier_score:.4f}, Iso={test_iso_metrics.brier_score:.4f}"
        )

    # Compute macro-average and sample-weighted average summaries across all 12 endpoints
    methods = ["uncalibrated", "sigmoid", "isotonic"]
    macro_summary: Dict[str, Any] = {}
    weighted_summary: Dict[str, Any] = {}
    total_test_samples = sum(r["data_summary"]["test_count"] for r in endpoint_reports.values())

    for m in methods:
        briers = [r["held_out_test_metrics"][m]["brier_score"] for r in endpoint_reports.values()]
        lls = [r["held_out_test_metrics"][m]["log_loss_value"] for r in endpoint_reports.values()]
        eces = [r["held_out_test_metrics"][m]["ece_10bin_weighted"] for r in endpoint_reports.values()]
        aucs = [r["held_out_test_metrics"][m]["roc_auc"] for r in endpoint_reports.values()]
        aps = [r["held_out_test_metrics"][m]["average_precision"] for r in endpoint_reports.values()]
        weights = [r["data_summary"]["test_count"] / total_test_samples for r in endpoint_reports.values()]

        macro_summary[m] = {
            "brier_score": round(float(np.mean(briers)), 4),
            "log_loss": round(float(np.mean(lls)), 4),
            "ece_10bin_weighted": round(float(np.mean(eces)), 4),
            "roc_auc": round(float(np.mean(aucs)), 4),
            "average_precision": round(float(np.mean(aps)), 4),
        }
        weighted_summary[m] = {
            "brier_score": round(float(np.sum(np.array(briers) * weights)), 4),
            "log_loss": round(float(np.sum(np.array(lls) * weights)), 4),
            "ece_10bin_weighted": round(float(np.sum(np.array(eces) * weights)), 4),
            "roc_auc": round(float(np.sum(np.array(aucs) * weights)), 4),
            "average_precision": round(float(np.sum(np.array(aps) * weights)), 4),
        }

    benchmark_summary = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_at": now_iso,
        "total_runtime_seconds": round(time.time() - t0, 2),
        "method_selection_protocol": f"{n_splits}_fold_3way_disjoint_base60_cal20_val20",
        "protocol": f"{n_splits}_fold_stratified_out_of_fold_posthoc",
        "historical_note": (
            "Supersedes exploratory schema 1.0.0 precomputed-OOF selection run; "
            "method selection in schema 1.1.0 uses strictly disjoint 3-way (base_rf / cal_fit / val) folds."
        ),
        "macro_average_test_metrics": macro_summary,
        "sample_weighted_test_metrics": weighted_summary,
        "endpoints": endpoint_reports,
    }

    summary_path = os.path.join(resolved_dir, "calibration_benchmark.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    return benchmark_summary


class Tox21CalibrationService:
    """
    Thread-safe runtime loader and evaluator for Tox21 post-hoc probability calibration.

    Enforces:
      - Exact supported `schema_version` (`CALIBRATION_SCHEMA_VERSION == "1.1.0"`)
      - Endpoint identity match
      - Valid 64-hex `dataset_sha256` and `base_model_sha256` provenance
      - Cryptographic SHA-256 binding to deployed `{safe_ep}_model.joblib`
      - Non-empty, finite, monotonic calibration parameters and finite outputs in `[0.0, 1.0]`
    Fails safely without suppressing valid raw predictions.
    """

    def __init__(self, artifact_dir: Optional[Union[str, Path]] = None):
        self.artifact_dir = resolve_path(artifact_dir or TOX21_ARTIFACT_DIR)
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._model_hashes: Dict[str, str] = {}

    def _get_model_sha256(self, endpoint: str) -> Optional[str]:
        if endpoint in self._model_hashes:
            return self._model_hashes[endpoint]
        safe_name = endpoint.replace("-", "_")
        model_path = os.path.join(self.artifact_dir, f"{safe_name}_model.joblib")
        if not os.path.exists(model_path):
            return None
        try:
            digest = compute_file_sha256(model_path)
            self._model_hashes[endpoint] = digest
            return digest
        except Exception:
            return None

    def load_endpoint_calibration(self, endpoint: str) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
        """
        Load and strictly validate calibration artifact for `endpoint`.

        Returns
        -------
        artifact : Optional[Dict[str, Any]]
            Validated calibration dictionary, or None if unavailable/invalid.
        status : str
            "AVAILABLE", "UNAVAILABLE", "UNSUPPORTED_SCHEMA", "CORRUPT_ARTIFACT", or "MODEL_HASH_MISMATCH".
        warning : Optional[str]
            Diagnostic message when calibration cannot be loaded or applied.
        """
        if endpoint in self._artifacts:
            return self._artifacts[endpoint], "AVAILABLE", None

        safe_name = endpoint.replace("-", "_")
        cal_path = os.path.join(self.artifact_dir, f"{safe_name}_calibration.json")
        if not os.path.exists(cal_path):
            return None, "UNAVAILABLE", f"Calibration artifact not found for '{endpoint}'."

        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"Corrupt calibration artifact JSON for '{endpoint}': {exc}")
            return None, "CORRUPT_ARTIFACT", f"Corrupt calibration artifact for '{endpoint}': {exc}"

        if not isinstance(data, dict):
            return None, "CORRUPT_ARTIFACT", f"Calibration artifact root must be a JSON object for '{endpoint}'."

        # 1. Exact schema version check
        schema_ver = data.get("schema_version")
        if schema_ver != CALIBRATION_SCHEMA_VERSION:
            return (
                None,
                "UNSUPPORTED_SCHEMA",
                f"Unsupported calibration schema version '{schema_ver}' for '{endpoint}' "
                f"(expected '{CALIBRATION_SCHEMA_VERSION}').",
            )

        # 2. Required top-level sections
        required_keys = {
            "schema_version",
            "endpoint",
            "provenance",
            "data_summary",
            "selection_governance",
            "calibrator_parameters",
        }
        if not required_keys.issubset(set(data.keys())):
            missing = sorted(required_keys - set(data.keys()))
            return (
                None,
                "CORRUPT_ARTIFACT",
                f"Missing required top-level calibration keys {missing} for '{endpoint}'.",
            )

        # 3. Endpoint identity check
        if data.get("endpoint") != endpoint:
            return None, "CORRUPT_ARTIFACT", f"Endpoint mismatch in calibration artifact for '{endpoint}'."

        # 4. Provenance & cryptographic hash validation
        prov = data.get("provenance")
        required_prov_keys = {
            "base_model_filename",
            "base_model_sha256",
            "dataset_path",
            "dataset_sha256",
            "split_strategy",
            "method_selection_protocol",
            "calibration_protocol",
            "random_seed",
            "library_versions",
        }
        if not isinstance(prov, dict) or not required_prov_keys.issubset(set(prov.keys())):
            return None, "CORRUPT_ARTIFACT", f"Incomplete provenance metadata for '{endpoint}'."

        expected_model_hash = prov.get("base_model_sha256")
        dataset_hash = prov.get("dataset_sha256")
        if not _is_valid_sha256_hex(expected_model_hash) or not _is_valid_sha256_hex(dataset_hash):
            return None, "CORRUPT_ARTIFACT", f"Malformed SHA-256 hash format in provenance for '{endpoint}'."

        actual_model_hash = self._get_model_sha256(endpoint)
        if not actual_model_hash or expected_model_hash.lower() != actual_model_hash.lower():
            return (
                None,
                "MODEL_HASH_MISMATCH",
                f"Calibration artifact SHA-256 mismatch with deployed base model for '{endpoint}'.",
            )

        # 5. Data summary & sample count validation
        ds = data.get("data_summary")
        required_ds_keys = {
            "usable_samples",
            "train_count",
            "train_positive_count",
            "train_negative_count",
            "test_count",
        }
        if not isinstance(ds, dict) or not required_ds_keys.issubset(set(ds.keys())):
            return None, "CORRUPT_ARTIFACT", f"Incomplete data_summary in calibration artifact for '{endpoint}'."
        try:
            if (
                int(ds["train_count"]) <= 0
                or int(ds["train_positive_count"]) <= 0
                or int(ds["train_negative_count"]) <= 0
                or int(ds["train_positive_count"]) + int(ds["train_negative_count"]) != int(ds["train_count"])
            ):
                return None, "CORRUPT_ARTIFACT", f"Invalid training class counts in data_summary for '{endpoint}'."
        except (TypeError, ValueError):
            return None, "CORRUPT_ARTIFACT", f"Non-integer counts in data_summary for '{endpoint}'."

        # 6. Selection governance validation
        gov = data.get("selection_governance")
        if (
            not isinstance(gov, dict)
            or gov.get("recommended_method") not in {"sigmoid", "isotonic", "uncalibrated"}
            or not gov.get("method_selection_rule")
        ):
            return None, "CORRUPT_ARTIFACT", f"Invalid selection_governance in calibration artifact for '{endpoint}'."

        # 7. Calibrator parameter finiteness & monotonicity validation
        cal_params = data.get("calibrator_parameters")
        if not isinstance(cal_params, dict):
            return None, "CORRUPT_ARTIFACT", f"Invalid calibrator_parameters in calibration artifact for '{endpoint}'."

        try:
            if "sigmoid" in cal_params:
                validate_sigmoid_params(cal_params["sigmoid"])
            if "isotonic" in cal_params:
                validate_isotonic_params(cal_params["isotonic"])
            rec_m = gov["recommended_method"]
            if rec_m in {"sigmoid", "isotonic"} and rec_m not in cal_params:
                return (
                    None,
                    "CORRUPT_ARTIFACT",
                    f"Recommended method '{rec_m}' missing from calibrator_parameters for '{endpoint}'.",
                )
        except ValueError as exc:
            return (
                None,
                "CORRUPT_ARTIFACT",
                f"Invalid calibrator parameters in artifact for '{endpoint}': {exc}",
            )

        self._artifacts[endpoint] = data
        return data, "AVAILABLE", None

    def calibrate_probability(
        self,
        endpoint: str,
        raw_probability: float,
        method: str = "recommended",
    ) -> Dict[str, Any]:
        """
        Compute calibrated probability for a given endpoint and raw probability.
        Never emits NaN, infinity, or out-of-range values as an available prediction.
        """
        if isinstance(raw_probability, (bool, np.bool_)):
            return {
                "calibrated_probability": None,
                "calibration_method": None,
                "calibration_status": "INVALID_INPUT",
                "calibration_version": None,
                "calibration_warning": "Raw probability cannot be a boolean.",
            }
        try:
            raw_val = float(raw_probability)
        except (TypeError, ValueError):
            return {
                "calibrated_probability": None,
                "calibration_method": None,
                "calibration_status": "INVALID_INPUT",
                "calibration_version": None,
                "calibration_warning": "Raw probability must be a finite float within [0.0, 1.0].",
            }

        if not np.isfinite(raw_val) or raw_val < 0.0 or raw_val > 1.0:
            return {
                "calibrated_probability": None,
                "calibration_method": None,
                "calibration_status": "INVALID_INPUT",
                "calibration_version": None,
                "calibration_warning": "Raw probability must be finite within [0.0, 1.0].",
            }

        art, status, err_msg = self.load_endpoint_calibration(endpoint)
        if art is None:
            return {
                "calibrated_probability": None,
                "calibration_method": None,
                "calibration_status": status,
                "calibration_version": None,
                "calibration_warning": err_msg,
            }

        gov = art["selection_governance"]
        target_method = gov["recommended_method"] if method == "recommended" else method
        version = art.get("schema_version", CALIBRATION_SCHEMA_VERSION)

        if target_method == "uncalibrated":
            return {
                "calibrated_probability": round(raw_val, 4),
                "calibration_method": "uncalibrated",
                "calibration_status": "AVAILABLE",
                "calibration_version": version,
                "calibration_warning": gov.get("scientific_disclaimer", TOX21_CALIBRATION_DISCLAIMER),
            }

        cal_params = art.get("calibrator_parameters", {}).get(target_method)
        if not cal_params:
            return {
                "calibrated_probability": None,
                "calibration_method": target_method,
                "calibration_status": "UNAVAILABLE",
                "calibration_version": version,
                "calibration_warning": f"Parameters for method '{target_method}' not found.",
            }

        try:
            cal_prob = float(apply_calibrator_params(raw_val, cal_params))
            if not np.isfinite(cal_prob) or cal_prob < 0.0 or cal_prob > 1.0:
                raise ValueError(f"Calibrated output {cal_prob} is non-finite or outside [0.0, 1.0].")
            return {
                "calibrated_probability": round(cal_prob, 4),
                "calibration_method": target_method,
                "calibration_status": "AVAILABLE",
                "calibration_version": version,
                "calibration_warning": gov.get("scientific_disclaimer", TOX21_CALIBRATION_DISCLAIMER),
            }
        except Exception as exc:
            return {
                "calibrated_probability": None,
                "calibration_method": target_method,
                "calibration_status": "ERROR",
                "calibration_version": version,
                "calibration_warning": f"Calibration transformation failed: {exc}",
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = benchmark_and_persist_tox21_calibration(reuse_existing=False)
    print(f"Completed Tox21 calibration benchmark in {summary['total_runtime_seconds']}s.")
    print("Macro-average held-out test metrics:", json.dumps(summary["macro_average_test_metrics"], indent=2))
