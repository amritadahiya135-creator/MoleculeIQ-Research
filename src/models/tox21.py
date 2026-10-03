"""Tox21 12-endpoint multi-model training and evaluation engine (Module M05)."""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
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
from src.chemistry.scaffolds import scaffold_train_test_split
from src.config import TOX21_DATA_PATH, TOX21_ARTIFACT_DIR, resolve_path

logger = logging.getLogger(__name__)

TOX21_ENDPOINTS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


@dataclass
class Tox21EndpointMetrics:
    """Evaluation metrics for an individual Tox21 endpoint."""
    endpoint: str
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
    confusion_matrix: list


def train_tox21_all_endpoints(
    csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    endpoints: Optional[List[str]] = None,
    n_estimators: int = 300,
    class_weight: str = "balanced",
    split_ratio: float = 0.2,
    random_seed: int = 42,
    radius: int = 2,
    n_bits: int = 2048,
    split_type: str = "random",
) -> Dict[str, Tox21EndpointMetrics]:
    """
    Train 12 independent binary Random Forest classifiers for all specified Tox21 endpoints.

    Parameters
    ----------
    split_type : str, default="random"
        Splitting methodology: "random" (stratified random split) or "scaffold" (Bemis-Murcko scaffold split).
    """
    if endpoints is None:
        endpoints = TOX21_ENDPOINTS

    resolved_csv = resolve_path(csv_path or TOX21_DATA_PATH)
    resolved_output = resolve_path(output_dir or TOX21_ARTIFACT_DIR)
    os.makedirs(resolved_output, exist_ok=True)
    df = pd.read_csv(resolved_csv)

    logger.info(f"Ingesting Tox21 dataset from '{resolved_csv}' with {len(df)} total records...")

    # Step 1: Pre-validate and generate fingerprints for all valid SMILES
    valid_mask = []
    fps_list = []
    canonical_smiles_list = []

    for idx, smi in enumerate(df["smiles"]):
        val_res = validate_smiles(str(smi))
        if not val_res.is_valid or not val_res.canonical_smiles:
            valid_mask.append(False)
            fps_list.append(None)
            canonical_smiles_list.append(None)
            continue

        try:
            fp_res = generate_morgan_fingerprint(
                val_res.canonical_smiles,
                radius=radius,
                n_bits=n_bits,
                return_bit_info=False,
            )
            valid_mask.append(True)
            fps_list.append(fp_res.numpy_vector)
            canonical_smiles_list.append(val_res.canonical_smiles)
        except Exception:
            valid_mask.append(False)
            fps_list.append(None)
            canonical_smiles_list.append(None)

    valid_mask = np.array(valid_mask, dtype=bool)
    logger.info(f"Validated {np.sum(valid_mask)} / {len(df)} molecules in Tox21.")

    all_metrics: Dict[str, Tox21EndpointMetrics] = {}
    summary: Dict[str, Any] = {
        "dataset": "Tox21",
        "split_type": split_type,
        "endpoints_trained": [],
        "metrics_summary": {},
    }

    # Step 2: Train independent classifier for each endpoint
    for endpoint in endpoints:
        if endpoint not in df.columns:
            logger.warning(f"Endpoint '{endpoint}' not in dataframe columns. Skipping.")
            continue

        # Extract target series where label is NOT NaN and molecule is valid
        ep_series = df[endpoint]
        not_na_mask = ep_series.notna().values & valid_mask

        n_samples = np.sum(not_na_mask)
        if n_samples < 50:
            logger.warning(f"Insufficient samples ({n_samples}) for endpoint '{endpoint}'. Skipping.")
            continue

        ep_indices = np.where(not_na_mask)[0]
        X_ep = np.array([fps_list[i] for i in ep_indices], dtype=np.int8)
        y_ep = ep_series.values[not_na_mask].astype(int)
        ep_smiles = [canonical_smiles_list[i] for i in ep_indices]

        n_pos = int(np.sum(y_ep == 1))
        n_neg = int(np.sum(y_ep == 0))

        if n_pos < 5 or n_neg < 5:
            logger.warning(f"Extreme class sparsity for '{endpoint}' (pos={n_pos}, neg={n_neg}). Skipping.")
            continue

        if split_type == "scaffold":
            train_idx, test_idx = scaffold_train_test_split(
                ep_smiles,
                test_size=split_ratio,
                random_state=random_seed,
            )
            X_train, X_test = X_ep[train_idx], X_ep[test_idx]
            y_train, y_test = y_ep[train_idx], y_ep[test_idx]
        else:
            # Stratified train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_ep,
                y_ep,
                test_size=split_ratio,
                random_state=random_seed,
                stratify=y_ep,
            )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # Check explicit positive class index
        pos_idx = int(np.where(clf.classes_ == 1)[0][0])
        y_test_probs = clf.predict_proba(X_test)[:, pos_idx]
        y_test_preds = (y_test_probs >= 0.5).astype(int)

        # Compute metrics
        roc_auc = float(roc_auc_score(y_test, y_test_probs))
        pr_auc = float(average_precision_score(y_test, y_test_probs))
        acc = float(accuracy_score(y_test, y_test_preds))
        bal_acc = float(balanced_accuracy_score(y_test, y_test_preds))
        prec = float(precision_score(y_test, y_test_preds, zero_division=0))
        rec = float(recall_score(y_test, y_test_preds, zero_division=0))
        f1 = float(f1_score(y_test, y_test_preds, zero_division=0))
        raw_cm = confusion_matrix(y_test, y_test_preds)
        cm = [[int(cell) for cell in row] for row in raw_cm]

        metrics = Tox21EndpointMetrics(
            endpoint=endpoint,
            sample_count=int(n_samples),
            train_count=int(len(y_train)),
            test_count=int(len(y_test)),
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

        all_metrics[endpoint] = metrics

        # Persist individual endpoint model and metadata
        safe_ep_name = endpoint.replace("-", "_")
        model_filename = f"{safe_ep_name}_model.joblib"
        meta_filename = f"{safe_ep_name}_metadata.json"

        model_path = os.path.join(resolved_output, model_filename)
        meta_path = os.path.join(resolved_output, meta_filename)

        metadata = {
            "endpoint": endpoint,
            "algorithm": "RandomForestClassifier",
            "n_estimators": int(n_estimators),
            "class_weight": class_weight,
            "split_ratio": float(split_ratio),
            "split_type": split_type,
            "random_seed": int(random_seed),
            "classes": [int(c) for c in clf.classes_],
            "positive_class_index": int(pos_idx),
            "metrics": asdict(metrics),
            "status": "APPROVED_FOR_DEMO",
        }

        joblib.dump(clf, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        summary["endpoints_trained"].append(endpoint)
        summary["metrics_summary"][endpoint] = asdict(metrics)
        logger.info(f"Tox21 endpoint '{endpoint}' ({split_type} split) saved. ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    # Save aggregate summary
    summary_path = os.path.join(resolved_output, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return all_metrics


def evaluate_tox21_dual_benchmarks(
    csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    endpoints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute and record dual benchmarks across selected Tox21 endpoints.
    """
    if endpoints is None:
        # Default representative subset covering nuclear receptors and stress response
        endpoints = ["NR-AR-LBD", "NR-AhR", "SR-MMP", "SR-p53"]

    resolved_output = resolve_path(output_dir or TOX21_ARTIFACT_DIR)

    # 1. Evaluate on scaffold split
    scaffold_metrics = train_tox21_all_endpoints(
        csv_path=csv_path,
        output_dir=resolved_output,
        endpoints=endpoints,
        split_type="scaffold",
    )

    # 2. Re-evaluate and persist random split baseline (ensuring primary models remain on baseline)
    random_metrics = train_tox21_all_endpoints(
        csv_path=csv_path,
        output_dir=resolved_output,
        endpoints=endpoints,
        split_type="random",
    )

    comparison = {
        "dataset": "Tox21",
        "endpoints_evaluated": endpoints,
        "random_split": {ep: asdict(m) for ep, m in random_metrics.items()},
        "scaffold_split": {ep: asdict(m) for ep, m in scaffold_metrics.items()},
        "methodological_notice": (
            "Random split evaluates chemical interpolation within known scaffolds; "
            "Bemis-Murcko scaffold split evaluates out-of-scaffold generalization to novel chemotypes."
        ),
    }

    benchmark_path = os.path.join(resolved_output, "scaffold_benchmark.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return comparison
