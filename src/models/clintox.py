"""ClinTox dataset ingestion, deduplication, calibration, training, and evaluation (Module M04)."""

import os
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Tuple, Optional, List, Union
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
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
from src.chemistry.scaffolds import scaffold_train_test_split
from src.config import CLINTOX_DATA_PATH, CLINTOX_ARTIFACT_DIR, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class ClinToxEvaluationMetrics:
    """Evaluation metrics on held-out test dataset."""
    sample_count: int
    train_count: int
    test_count: int
    test_positive_count: int
    test_negative_count: int
    roc_auc: float
    pr_auc: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list  # [[TN, FP], [FN, TP]]
    brier_score: float = 0.0
    log_loss: float = 0.0
    expected_calibration_error: float = 0.0
    calibrated: bool = False
    calibration_method: Optional[str] = None
    duplicate_policy: str = "conservative_toxic"
    conflicts_resolved_count: int = 0


@dataclass
class ClinToxTrainingResult:
    """Artifact bundle returned after training ClinTox classifier."""
    model: Any
    metrics: ClinToxEvaluationMetrics
    metadata: Dict[str, Any]
    artifact_path: str
    metadata_path: str


def audit_clintox_duplicates(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Audit ClinTox dataset for duplicate canonical SMILES and conflicting target labels.

    Parameters
    ----------
    csv_path : Optional[str]
        Path to raw ClinTox CSV. Defaults to CLINTOX_DATA_PATH.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with duplicate groups, conflicting labels, and row index mappings.
    """
    resolved_path = resolve_path(csv_path or CLINTOX_DATA_PATH)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"ClinTox CSV dataset not found at '{resolved_path}'.")

    df = pd.read_csv(resolved_path)
    if "CT_TOX" not in df.columns or "smiles" not in df.columns:
        raise ValueError(f"ClinTox CSV must contain 'smiles' and 'CT_TOX' columns. Found: {df.columns.tolist()}")

    canon_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid_rows = []

    for idx, row in df.iterrows():
        smi = str(row["smiles"])
        val_res = validate_smiles(smi)
        if not val_res.is_valid:
            invalid_rows.append({"index": idx, "smiles": smi, "error": val_res.error_message})
            continue

        fda = row.get("FDA_APPROVED", 0)
        try:
            fda_val = int(fda) if pd.notna(fda) else 0
        except (ValueError, TypeError):
            fda_val = 0

        target = row["CT_TOX"]
        try:
            target_val = int(target) if pd.notna(target) else None
        except (ValueError, TypeError):
            target_val = None

        if target_val is None or target_val not in (0, 1):
            continue

        canon_map[val_res.canonical_smiles].append({
            "index": idx,
            "raw_smiles": smi,
            "FDA_APPROVED": fda_val,
            "CT_TOX": target_val,
        })

    duplicate_groups = {}
    conflicting_groups = {}
    consistent_groups = {}

    for c_smi, entries in canon_map.items():
        if len(entries) > 1:
            duplicate_groups[c_smi] = entries
            labels = set(e["CT_TOX"] for e in entries)
            if len(labels) > 1:
                conflicting_groups[c_smi] = entries
            else:
                consistent_groups[c_smi] = entries

    report = {
        "total_raw_rows": len(df),
        "invalid_smiles_count": len(invalid_rows),
        "unique_canonical_smiles": len(canon_map),
        "duplicate_groups_count": len(duplicate_groups),
        "conflicting_groups_count": len(conflicting_groups),
        "consistent_duplicate_groups_count": len(consistent_groups),
        "conflicting_groups": conflicting_groups,
        "invalid_rows": invalid_rows,
    }
    return report


def prepare_clintox_dataset(
    csv_path: str = "data/raw/clintox.csv",
    radius: int = 2,
    n_bits: int = 2048,
    duplicate_policy: str = "conservative_toxic",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Ingest, validate, deduplicate, and featurize the ClinTox benchmark dataset.

    Parameters
    ----------
    csv_path : str
        Path to raw ClinTox CSV.
    radius : int
        Morgan fingerprint circular radius.
    n_bits : int
        Number of fingerprint bits.
    duplicate_policy : str, default="conservative_toxic"
        Policy for resolving conflicting labels across duplicate canonical SMILES:
        - "conservative_toxic": Precautionary screening; assign CT_TOX=1 if any observation failed.
        - "drop_conflicts": Exclude all contradictory groups to maintain strict label purity.
        - "majority_vote": Round average of conflicting labels.

    Returns
    -------
    X : np.ndarray
        Shape (N_valid, n_bits), binary Morgan fingerprints.
    y : np.ndarray
        Shape (N_valid,), binary CT_TOX target labels.
    df_clean : pd.DataFrame
        Sanitized dataframe with validation and deduplication metadata.
    """
    resolved_path = resolve_path(csv_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"ClinTox CSV dataset not found at '{resolved_path}'.")

    df = pd.read_csv(resolved_path)
    if "CT_TOX" not in df.columns or "smiles" not in df.columns:
        raise ValueError(f"ClinTox CSV must contain 'smiles' and 'CT_TOX' columns. Found: {df.columns.tolist()}")

    supported_policies = ("conservative_toxic", "drop_conflicts", "majority_vote")
    if duplicate_policy not in supported_policies:
        raise ValueError(
            f"Unsupported duplicate_policy '{duplicate_policy}'. Choose from {supported_policies}."
        )

    # 1. Group records by canonical SMILES
    canon_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid_count = 0

    for idx, row in df.iterrows():
        smi = str(row["smiles"])
        val_res = validate_smiles(smi)
        if not val_res.is_valid:
            logger.warning(f"Row {idx}: skipping invalid SMILES '{smi}': {val_res.error_message}")
            invalid_count += 1
            continue

        fda = row.get("FDA_APPROVED", 0)
        try:
            fda_val = int(fda) if pd.notna(fda) else 0
        except (ValueError, TypeError):
            fda_val = 0

        target = row["CT_TOX"]
        if pd.isna(target):
            continue

        try:
            target_int = int(target)
            if target_int not in (0, 1):
                continue
        except (ValueError, TypeError):
            continue

        canon_map[val_res.canonical_smiles].append({
            "source_idx": idx,
            "raw_smiles": smi,
            "FDA_APPROVED": fda_val,
            "CT_TOX": target_int,
        })

    # 2. Deterministic deduplication & conflict resolution
    cleaned_rows = []
    fps = []
    labels = []
    conflicts_resolved_count = 0
    duplicate_groups_count = 0

    for c_smi, entries in canon_map.items():
        is_dup = len(entries) > 1
        if is_dup:
            duplicate_groups_count += 1

        tox_labels = [e["CT_TOX"] for e in entries]
        unique_tox = set(tox_labels)
        is_conflict = len(unique_tox) > 1

        if is_conflict:
            conflicts_resolved_count += 1
            if duplicate_policy == "drop_conflicts":
                logger.info(f"Deduplication: dropped conflicting group '{c_smi}' with labels {tox_labels}")
                continue
            elif duplicate_policy == "conservative_toxic":
                # Precautionary toxic screening: trial failure anywhere flags risk
                resolved_tox = 1
            elif duplicate_policy == "majority_vote":
                resolved_tox = int(round(np.mean(tox_labels)))
            else:
                resolved_tox = 1
        else:
            resolved_tox = tox_labels[0]

        # Consolidated FDA approval flag: approved in any source record
        fda_approved_max = int(max(e["FDA_APPROVED"] for e in entries))
        source_indices = [e["source_idx"] for e in entries]
        raw_smiles_list = [e["raw_smiles"] for e in entries]

        # Generate Morgan fingerprint for unique canonical structure
        fp_res = generate_morgan_fingerprint(
            c_smi,
            radius=radius,
            n_bits=n_bits,
            return_bit_info=False,
        )

        fps.append(fp_res.numpy_vector)
        labels.append(resolved_tox)
        cleaned_rows.append({
            "smiles": c_smi,
            "canonical_smiles": c_smi,
            "CT_TOX": resolved_tox,
            "FDA_APPROVED": fda_approved_max,
            "source_indices": source_indices,
            "raw_smiles": raw_smiles_list,
            "is_duplicate": is_dup,
            "is_conflict": is_conflict,
        })

    X = np.array(fps, dtype=np.int8)
    y = np.array(labels, dtype=np.int32)
    df_clean = pd.DataFrame(cleaned_rows)

    report = {
        "total_raw": len(df),
        "invalid_count": invalid_count,
        "unique_canonical": len(canon_map),
        "duplicate_groups_count": duplicate_groups_count,
        "conflicts_resolved_count": conflicts_resolved_count,
        "duplicate_policy": duplicate_policy,
        "retained_compounds": len(y),
        "retained_positives": int(np.sum(y == 1)),
        "retained_negatives": int(np.sum(y == 0)),
    }
    df_clean.attrs["duplicate_report"] = report

    logger.info(
        f"ClinTox prepared (policy={duplicate_policy}): {len(y)} unique compounds "
        f"({np.sum(y == 1)} toxic, {np.sum(y == 0)} non-toxic). Resolved {conflicts_resolved_count} conflicts."
    )
    return X, y, df_clean


def train_clintox_model(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_estimators: int = 300,
    class_weight: str = "balanced",
    split_ratio: float = 0.2,
    random_seed: int = 42,
    radius: int = 2,
    n_bits: int = 2048,
    split_type: str = "random",
    duplicate_policy: str = "conservative_toxic",
    calibration_method: Optional[str] = "sigmoid",
) -> ClinToxTrainingResult:
    """
    Train, calibrate, evaluate, and serialize the ClinTox classifier.

    Parameters
    ----------
    data_path : Optional[str]
        Path to raw ClinTox CSV.
    output_dir : Optional[str]
        Directory to save model artifacts.
    n_estimators : int
        Number of trees in Random Forest.
    class_weight : str
        Class weight setting (e.g. 'balanced').
    split_ratio : float
        Fraction reserved for test split.
    random_seed : int
        Random seed for reproducibility.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Morgan fingerprint bit length.
    split_type : str, default="random"
        Splitting regime: "random" (stratified random split) or "scaffold" (Bemis-Murcko).
    duplicate_policy : str, default="conservative_toxic"
        Duplicate conflict resolution policy: "conservative_toxic", "drop_conflicts", or "majority_vote".
    calibration_method : Optional[str], default="sigmoid"
        Probability calibration method: "sigmoid" (Platt scaling), "isotonic", or None.
    """
    resolved_data = resolve_path(data_path or CLINTOX_DATA_PATH)
    resolved_output = resolve_path(output_dir or CLINTOX_ARTIFACT_DIR)
    os.makedirs(resolved_output, exist_ok=True)

    X, y, df_clean = prepare_clintox_dataset(
        csv_path=resolved_data,
        radius=radius,
        n_bits=n_bits,
        duplicate_policy=duplicate_policy,
    )

    if split_type == "scaffold":
        train_idx, test_idx = scaffold_train_test_split(
            df_clean["canonical_smiles"].tolist(),
            test_size=split_ratio,
            random_state=random_seed,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        df_train = df_clean.iloc[train_idx].copy()
    else:
        # Default stratified random split
        idx_arr = np.arange(len(y))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            idx_arr,
            test_size=split_ratio,
            random_state=random_seed,
            stratify=y,
        )
        df_train = df_clean.iloc[idx_train].copy()

    # Train base Random Forest classifier
    base_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_seed,
        n_jobs=-1,
    )
    base_clf.fit(X_train, y_train)

    # Positive class index in base model
    positive_class_idx = int(np.where(base_clf.classes_ == 1)[0][0])

    # Probability calibration (fitted strictly on training partition only)
    if calibration_method is not None and calibration_method in ("sigmoid", "isotonic"):
        cal_clf = CalibratedClassifierCV(
            estimator=base_clf,
            method=calibration_method,
            cv=5,
        )
        cal_clf.fit(X_train, y_train)
        serving_model = cal_clf
        is_calibrated = True
    else:
        serving_model = base_clf
        is_calibrated = False

    # Predict on held-out test split using the serving model
    y_test_probs = serving_model.predict_proba(X_test)[:, positive_class_idx]
    y_test_preds = (y_test_probs >= 0.5).astype(int)

    # Compute comprehensive evaluation metrics
    roc_auc = float(roc_auc_score(y_test, y_test_probs))
    pr_auc = float(average_precision_score(y_test, y_test_probs))
    acc = float(accuracy_score(y_test, y_test_preds))
    bal_acc = float(balanced_accuracy_score(y_test, y_test_preds))
    prec = float(precision_score(y_test, y_test_preds, zero_division=0))
    rec = float(recall_score(y_test, y_test_preds, zero_division=0))
    f1 = float(f1_score(y_test, y_test_preds, zero_division=0))
    cm = confusion_matrix(y_test, y_test_preds).tolist()
    brier = float(brier_score_loss(y_test, y_test_probs))
    ll = float(log_loss(y_test, y_test_probs))

    # Calibration error computation
    prob_true, prob_pred = calibration_curve(y_test, y_test_probs, n_bins=5)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    dup_report = df_clean.attrs.get("duplicate_report", {})
    conflicts_resolved = dup_report.get("conflicts_resolved_count", 0)

    metrics = ClinToxEvaluationMetrics(
        sample_count=len(y),
        train_count=len(y_train),
        test_count=len(y_test),
        test_positive_count=int(np.sum(y_test == 1)),
        test_negative_count=int(np.sum(y_test == 0)),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        confusion_matrix=cm,
        brier_score=round(brier, 4),
        log_loss=round(ll, 4),
        expected_calibration_error=round(ece, 4),
        calibrated=is_calibrated,
        calibration_method=calibration_method if is_calibrated else None,
        duplicate_policy=duplicate_policy,
        conflicts_resolved_count=conflicts_resolved,
    )

    metadata = {
        "model_id": "clintox_rf_calibrated" if is_calibrated else "clintox_rf_baseline",
        "target": "CT_TOX",
        "algorithm": "RandomForestClassifier",
        "calibrated": is_calibrated,
        "calibration_method": calibration_method if is_calibrated else None,
        "duplicate_policy": duplicate_policy,
        "n_estimators": n_estimators,
        "class_weight": class_weight,
        "split_ratio": split_ratio,
        "split_type": split_type,
        "random_seed": random_seed,
        "features": {
            "type": "morgan",
            "radius": radius,
            "n_bits": n_bits,
        },
        "classes": [0, 1],
        "positive_class_index": positive_class_idx,
        "metrics": asdict(metrics),
        "duplicate_report": dup_report,
        "calibration_curve": {
            "prob_pred": np.round(prob_pred, 4).tolist(),
            "prob_true": np.round(prob_true, 4).tolist(),
        },
        "status": "APPROVED_FOR_DEMO",
    }

    # Save artifacts:
    # 1. Serving model (calibrated if enabled)
    model_path = os.path.join(resolved_output, "model.joblib")
    joblib.dump(serving_model, model_path)

    # 2. Base uncalibrated tree ensemble (for TreeSHAP explainability)
    uncal_model_path = os.path.join(resolved_output, "uncalibrated_model.joblib")
    joblib.dump(base_clf, uncal_model_path)

    # 3. Training fingerprints and canonical smiles (for Applicability Domain verification)
    train_fps_path = os.path.join(resolved_output, "training_fingerprints.npz")
    np.savez_compressed(
        train_fps_path,
        fingerprints=X_train,
        smiles=df_train["canonical_smiles"].values,
    )

    # 4. Comprehensive metadata
    meta_path = os.path.join(resolved_output, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"ClinTox model saved to {model_path}. "
        f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Brier: {brier:.4f}, ECE: {ece:.4f}"
    )

    return ClinToxTrainingResult(
        model=serving_model,
        metrics=metrics,
        metadata=metadata,
        artifact_path=model_path,
        metadata_path=meta_path,
    )


def evaluate_clintox_dual_benchmarks(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    duplicate_policy: str = "conservative_toxic",
    calibration_method: Optional[str] = "sigmoid",
) -> Dict[str, Any]:
    """
    Execute and record dual benchmarks: Stratified Random Split vs. Bemis-Murcko Scaffold Split.
    """
    resolved_output = resolve_path(output_dir or CLINTOX_ARTIFACT_DIR)

    # 1. Random split baseline
    res_random = train_clintox_model(
        data_path=data_path,
        output_dir=resolved_output,
        split_type="random",
        duplicate_policy=duplicate_policy,
        calibration_method=calibration_method,
    )

    # 2. Scaffold split benchmark
    res_scaffold = train_clintox_model(
        data_path=data_path,
        output_dir=resolved_output,
        split_type="scaffold",
        duplicate_policy=duplicate_policy,
        calibration_method=calibration_method,
    )

    # 3. Re-save primary production model as random split baseline
    res_final = train_clintox_model(
        data_path=data_path,
        output_dir=resolved_output,
        split_type="random",
        duplicate_policy=duplicate_policy,
        calibration_method=calibration_method,
    )

    dual_comparison = {
        "dataset": "ClinTox",
        "target": "CT_TOX",
        "duplicate_policy": duplicate_policy,
        "calibration_method": calibration_method,
        "random_split": asdict(res_random.metrics),
        "scaffold_split": asdict(res_scaffold.metrics),
        "methodological_notice": (
            "Random split evaluates chemical interpolation within known scaffolds; "
            "Bemis-Murcko scaffold split evaluates out-of-scaffold generalization to novel chemotypes."
        ),
    }

    benchmark_path = os.path.join(resolved_output, "scaffold_benchmark.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(dual_comparison, f, indent=2)

    return dual_comparison
