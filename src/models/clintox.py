"""ClinTox dataset ingestion, training pipeline, and evaluation (Module M04)."""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
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
)

from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class ClinToxEvaluationMetrics:
    """Evaluation metrics on held-out stratified test dataset."""
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


@dataclass
class ClinToxTrainingResult:
    """Artifact bundle returned after training ClinTox classifier."""
    model: RandomForestClassifier
    metrics: ClinToxEvaluationMetrics
    metadata: Dict[str, Any]
    artifact_path: str
    metadata_path: str


def prepare_clintox_dataset(
    csv_path: str = "data/raw/clintox.csv",
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Ingest, validate, and featurize the ClinTox benchmark dataset.

    Parameters
    ----------
    csv_path : str
        Path to raw ClinTox CSV.
    radius : int
        Morgan fingerprint circular radius.
    n_bits : int
        Number of fingerprint bits.

    Returns
    -------
    X : np.ndarray
        Shape (N_valid, n_bits), binary Morgan fingerprints.
    y : np.ndarray
        Shape (N_valid,), binary CT_TOX target labels.
    df_clean : pd.DataFrame
        Sanitized dataframe with validation metadata.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ClinTox CSV dataset not found at '{csv_path}'.")

    df = pd.read_csv(csv_path)
    if "CT_TOX" not in df.columns or "smiles" not in df.columns:
        raise ValueError(f"ClinTox CSV must contain 'smiles' and 'CT_TOX' columns. Found: {df.columns.tolist()}")

    valid_indices = []
    fps = []
    labels = []
    canonical_smiles_list = []

    for idx, row in df.iterrows():
        smi = str(row["smiles"])
        val_res = validate_smiles(smi)
        if not val_res.is_valid:
            logger.warning(f"Row {idx}: skipping invalid SMILES '{smi}': {val_res.error_message}")
            continue

        target = row["CT_TOX"]
        if pd.isna(target):
            continue

        try:
            target_int = int(target)
            if target_int not in (0, 1):
                continue
        except (ValueError, TypeError):
            continue

        fp_res = generate_morgan_fingerprint(
            val_res.canonical_smiles,
            radius=radius,
            n_bits=n_bits,
            return_bit_info=False,
        )

        valid_indices.append(idx)
        fps.append(fp_res.numpy_vector)
        labels.append(target_int)
        canonical_smiles_list.append(val_res.canonical_smiles)

    X = np.array(fps, dtype=np.int8)
    y = np.array(labels, dtype=np.int32)
    df_clean = df.loc[valid_indices].copy()
    df_clean["canonical_smiles"] = canonical_smiles_list

    logger.info(f"ClinTox prepared: {len(y)} valid compounds ({np.sum(y == 1)} toxic, {np.sum(y == 0)} non-toxic).")
    return X, y, df_clean


def train_clintox_model(
    data_path: str = "data/raw/clintox.csv",
    output_dir: str = "artifacts/clintox",
    n_estimators: int = 300,
    class_weight: str = "balanced",
    split_ratio: float = 0.2,
    random_seed: int = 42,
    radius: int = 2,
    n_bits: int = 2048,
) -> ClinToxTrainingResult:
    """
    Train and evaluate the baseline ClinTox Random Forest classifier.

    Follows the exact PRD/SRD baseline specifications:
      - Estimators: 300
      - Class weight: 'balanced'
      - Split: 80/20 Stratified Random Split, random_state=42
      - Features: Morgan 2048-bit radius=2
    """
    os.makedirs(output_dir, exist_ok=True)
    X, y, df_clean = prepare_clintox_dataset(csv_path=data_path, radius=radius, n_bits=n_bits)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_ratio,
        random_state=random_seed,
        stratify=y,
    )

    # Train Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Explicit positive class index lookup
    positive_class_idx = int(np.where(clf.classes_ == 1)[0][0])

    # Predict on held-out test split
    y_test_probs = clf.predict_proba(X_test)[:, positive_class_idx]
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
    )

    metadata = {
        "model_id": "clintox_rf_baseline",
        "target": "CT_TOX",
        "algorithm": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "class_weight": class_weight,
        "split_ratio": split_ratio,
        "random_seed": random_seed,
        "features": {
            "type": "morgan",
            "radius": radius,
            "n_bits": n_bits,
        },
        "classes": clf.classes_.tolist(),
        "positive_class_index": positive_class_idx,
        "metrics": asdict(metrics),
        "status": "APPROVED_FOR_DEMO",
    }

    model_path = os.path.join(output_dir, "model.joblib")
    meta_path = os.path.join(output_dir, "metadata.json")

    joblib.dump(clf, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"ClinTox model saved to {model_path}. ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    return ClinToxTrainingResult(
        model=clf,
        metrics=metrics,
        metadata=metadata,
        artifact_path=model_path,
        metadata_path=meta_path,
    )
