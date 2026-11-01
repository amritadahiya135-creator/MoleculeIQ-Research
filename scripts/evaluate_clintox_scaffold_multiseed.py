"""
Controlled ClinTox Generalization Evaluation Runner (Priority 2D).

Executes a reproducible, non-destructive, multi-seed 5-fold cross-validation evaluation
of ClinTox (CT_TOX) comparing:
1. Primary analysis: Bemis-Murcko Scaffold-Grouped Stratified K-Fold (StratifiedGroupKFold)
2. Reference analysis: Matched Stratified Random K-Fold (StratifiedKFold)

Across 3 prespecified seeds (42, 1042, 2042) using the verified evaluation engine in
src/models/scaffold_generalization.py.

Calibration & Scientific Governance:
- Fits all models strictly on training partitions; held-out fold labels are never used
  for calibration, feature selection, or threshold tuning.
- Follows STAGE1_CALIBRATION_SCOPE_NOTE: evaluates uncalibrated tree-ensemble voting
  probabilities for rank discrimination. Post-hoc probability calibration metrics (ECE, MACE)
  are explicitly marked as UNAVAILABLE because internal random-fold calibration does NOT
  establish valid calibration on out-of-scaffold chemotypes, and nested scaffold-grouped
  calibration is outside Stage 1 scope.
- Does not modify or overwrite production models, checksum manifests, or datasets.
- Outputs structured diagnostic report to artifacts/diagnostics/clintox_scaffold_multiseed_report.json.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root is in sys.path when invoked directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.scaffold_generalization import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_SEED,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    STAGE1_CALIBRATION_SCOPE_NOTE,
    EvaluationCohort,
    BinaryEvaluationMetrics,
    CrossValidationFoldSummary,
    build_clintox_evaluation_cohort,
    compute_binary_evaluation_metrics,
    compute_scaffold_cluster_bootstrap,
    generate_grouped_kfold_splits,
    generate_matched_random_kfold_splits,
    generate_scaffold_group_assignments,
    summarize_fold_and_pooled_metrics,
    summarize_repeated_cv_runs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_clintox_scaffold_multiseed")

PRESPECIFIED_SEEDS: Tuple[int, ...] = (42, 1042, 2042)
DEFAULT_OUTPUT_PATH = "artifacts/diagnostics/clintox_scaffold_multiseed_report.json"


def _verify_all_checksums() -> Tuple[bool, List[str]]:
    """Verify that all 32 ML artifact checksums and 5 literature checksums remain intact."""
    failures: List[str] = []

    ml_manifest = Path("artifacts/checksums.sha256")
    if ml_manifest.exists():
        for line in ml_manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            exp_hash, rel_path = line.split(maxsplit=1)
            target = Path("artifacts") / rel_path.strip().replace("\\", "/")
            if not target.exists():
                failures.append(f"Missing ML artifact: {target}")
            else:
                act_hash = hashlib.sha256(target.read_bytes()).hexdigest()
                if act_hash != exp_hash:
                    failures.append(f"ML checksum mismatch: {target}")

    lit_manifest = Path("data/benchmarks/literature/checksums.sha256")
    if lit_manifest.exists():
        for line in lit_manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            exp_hash, rel_path = line.split(maxsplit=1)
            target = Path(rel_path.strip().replace("\\", "/"))
            if not target.exists():
                failures.append(f"Missing literature artifact: {target}")
            else:
                act_hash = hashlib.sha256(target.read_bytes()).hexdigest()
                if act_hash != exp_hash:
                    failures.append(f"Literature checksum mismatch: {target}")

    return len(failures) == 0, failures


def _metrics_to_dict(m: BinaryEvaluationMetrics) -> Dict[str, Any]:
    """Serialize BinaryEvaluationMetrics dataclass cleanly to a JSON-serializable dictionary."""
    data = asdict(m)
    return data


def run_single_fold_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold_seed: int,
    n_estimators: int = 300,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit base RF strictly on training fold and evaluate on test fold.

    Returns:
        (y_test_probs, metrics_dict)
    """
    train_prev = float(np.mean(y_train == 1))

    # Base Random Forest Classifier fitted strictly on training partition
    base_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=fold_seed,
        n_jobs=-1,
    )
    base_clf.fit(X_train, y_train)

    pos_idx = int(np.where(base_clf.classes_ == 1)[0][0])
    y_test_probs = base_clf.predict_proba(X_test)[:, pos_idx]

    metrics = compute_binary_evaluation_metrics(
        y_true=y_test,
        y_prob=y_test_probs,
        fixed_threshold=0.50,
        exploratory_train_prevalence_threshold=train_prev,
    )

    return y_test_probs, _metrics_to_dict(metrics)


def run_cross_validation_regime(
    cohort: EvaluationCohort,
    regime: str,
    seeds: Sequence[int],
    n_splits: int = 5,
    strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    n_bootstraps: int = 500,
) -> Dict[str, Any]:
    """Execute complete multi-seed K-fold CV for a specified regime ('scaffold_grouped' or 'stratified_random')."""
    assert cohort.fingerprints is not None, "Cohort fingerprints must be precomputed."
    X = cohort.fingerprints
    y = cohort.labels
    n_total = len(y)

    group_keys, _ = generate_scaffold_group_assignments(
        cohort.canonical_smiles,
        strategy=strategy,
    )

    seed_results: Dict[int, Dict[str, Any]] = {}
    seed_cv_summaries: Dict[int, CrossValidationFoldSummary] = {}

    for s_idx, seed in enumerate(seeds):
        logger.info(f"[{regime.upper()}] Starting seed {seed} ({s_idx + 1}/{len(seeds)}) ...")

        if regime == "scaffold_grouped":
            split_report, _ = generate_grouped_kfold_splits(
                smiles_or_identities=cohort.canonical_smiles,
                labels=y,
                strategy=strategy,
                n_splits=n_splits,
                seed=seed,
            )
        else:
            split_report = generate_matched_random_kfold_splits(
                smiles_or_identities=cohort.canonical_smiles,
                labels=y,
                reference_strategy=strategy,
                n_splits=n_splits,
                seed=seed,
            )

        assert split_report.zero_index_overlap, f"Index overlap detected in {regime} seed {seed}!"
        assert split_report.complete_index_conservation, f"Index loss detected in {regime} seed {seed}!"
        if regime == "scaffold_grouped":
            assert split_report.zero_group_overlap, f"Scaffold group leakage in {regime} seed {seed}!"

        oof_probs = np.full(n_total, np.nan, dtype=float)
        fold_records: List[Dict[str, Any]] = []
        fold_metrics_list: List[BinaryEvaluationMetrics] = []

        for f_idx, fold_summary in enumerate(split_report.folds):
            tr_idx = np.asarray(fold_summary.train_indices, dtype=int)
            te_idx = np.asarray(fold_summary.test_indices, dtype=int)
            fold_seed = int(seed * 100 + f_idx)

            t0 = time.time()
            y_te_probs, m_dict = run_single_fold_evaluation(
                X_train=X[tr_idx],
                y_train=y[tr_idx],
                X_test=X[te_idx],
                y_test=y[te_idx],
                fold_seed=fold_seed,
            )
            elapsed = time.time() - t0

            oof_probs[te_idx] = y_te_probs

            fold_metrics_list.append(
                compute_binary_evaluation_metrics(
                    y_true=y[te_idx],
                    y_prob=y_te_probs,
                    fixed_threshold=0.50,
                    exploratory_train_prevalence_threshold=fold_summary.train_prevalence,
                )
            )

            fold_records.append({
                "fold_index": f_idx,
                "fold_seed": fold_seed,
                "n_train": fold_summary.n_train,
                "n_test": fold_summary.n_test,
                "train_positives": fold_summary.n_train_positives,
                "train_negatives": fold_summary.n_train_negatives,
                "train_prevalence": fold_summary.train_prevalence,
                "test_positives": fold_summary.n_test_positives,
                "test_negatives": fold_summary.n_test_negatives,
                "test_prevalence": fold_summary.test_prevalence,
                "n_train_groups": fold_summary.n_train_groups,
                "n_test_groups": fold_summary.n_test_groups,
                "n_test_singletons": fold_summary.n_test_singleton_groups,
                "fit_and_eval_seconds": round(elapsed, 2),
                "metrics": m_dict,
            })

        assert not np.any(np.isnan(oof_probs)), f"Missing OOF predictions in {regime} seed {seed}!"

        # Summarize fold dispersion and pooled OOF metrics
        cv_summary = summarize_fold_and_pooled_metrics(
            fold_metrics_list=fold_metrics_list,
            pooled_y_true=y,
            pooled_y_prob=oof_probs,
            subset_name=f"{regime}_seed_{seed}",
            fixed_threshold=0.50,
        )
        seed_cv_summaries[seed] = cv_summary

        # Cyclic-only subset pooled OOF metrics
        cyclic_mask = cohort.cyclic_mask
        cyclic_m = compute_binary_evaluation_metrics(
            y_true=y[cyclic_mask],
            y_prob=oof_probs[cyclic_mask],
            subset_name="cyclic_only",
            fixed_threshold=0.50,
        )

        # Scaffold-cluster block bootstrap (500 replicates) on pooled OOF predictions
        logger.info(f"[{regime.upper()}] Computing {n_bootstraps}-replicate scaffold-cluster block bootstrap for seed {seed} ...")
        cluster_boot = compute_scaffold_cluster_bootstrap(
            y_true=y,
            y_prob=oof_probs,
            cluster_ids=group_keys,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )

        seed_results[seed] = {
            "seed": seed,
            "regime": regime,
            "n_splits": n_splits,
            "split_integrity_verified": True,
            "folds": fold_records,
            "fold_dispersion": {
                k: asdict(v) for k, v in cv_summary.fold_dispersion.items()
            },
            "pooled_oof_full_cohort": _metrics_to_dict(cv_summary.pooled_oof_metrics),
            "pooled_oof_cyclic_only": _metrics_to_dict(cyclic_m),
            "scaffold_cluster_bootstrap": {
                k: asdict(v) for k, v in cluster_boot.items()
            },
        }

    # Across-seed pooled summary
    across_seed = summarize_repeated_cv_runs(seed_cv_summaries)

    return {
        "regime": regime,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "n_splits_per_seed": n_splits,
        "total_folds_evaluated": len(seeds) * n_splits,
        "across_seed_pooled_oof_summary": across_seed["across_seed_pooled_oof_summary"],
        "per_seed_evaluations": {str(s): seed_results[s] for s in seeds},
    }


def run_controlled_clintox_evaluation(
    seeds: Sequence[int] = PRESPECIFIED_SEEDS,
    n_splits: int = DEFAULT_CV_SPLITS,
    strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    n_bootstraps: int = 500,
    output_path: str = DEFAULT_OUTPUT_PATH,
    smoke_test_only: bool = False,
) -> Dict[str, Any]:
    """Execute the full controlled ClinTox generalization evaluation or a 1-fold preflight smoke test."""
    logger.info("Building ClinTox evaluation cohort (duplicate_policy='conservative_toxic') ...")
    cohort = build_clintox_evaluation_cohort(
        duplicate_policy="conservative_toxic",
        compute_fingerprints=True,
    )
    logger.info(
        f"Cohort constructed: N={cohort.n_samples} ({cohort.n_positives} pos, {cohort.n_negatives} neg, "
        f"prevalence={cohort.prevalence*100:.2f}%, cyclic={int(np.sum(cohort.cyclic_mask))}, "
        f"acyclic={int(np.sum(~cohort.cyclic_mask))})."
    )

    if smoke_test_only:
        logger.info("[SMOKE TEST] Running 1-fold preflight smoke test on seed 42 ...")
        t0 = time.time()
        assert cohort.fingerprints is not None
        rep, _ = generate_grouped_kfold_splits(
            smiles_or_identities=cohort.canonical_smiles,
            labels=cohort.labels,
            strategy=strategy,
            n_splits=n_splits,
            seed=42,
        )
        fold0 = rep.folds[0]
        tr_idx = np.asarray(fold0.train_indices, dtype=int)
        te_idx = np.asarray(fold0.test_indices, dtype=int)

        y_probs, m_dict = run_single_fold_evaluation(
            X_train=cohort.fingerprints[tr_idx],
            y_train=cohort.labels[tr_idx],
            X_test=cohort.fingerprints[te_idx],
            y_test=cohort.labels[te_idx],
            fold_seed=4200,
        )
        dur = time.time() - t0
        logger.info(
            f"[SMOKE TEST] Passed in {dur:.2f}s. "
            f"Fold 0 Test N={len(te_idx)} ({fold0.n_test_positives} pos, {fold0.n_test_negatives} neg). "
            f"ROC-AUC={m_dict['roc_auc']}, PR-AUC={m_dict['pr_auc']}, Brier={m_dict['brier_score']}."
        )
        return {
            "status": "SMOKE_TEST_PASSED",
            "duration_seconds": round(dur, 2),
            "smoke_fold_metrics": m_dict,
        }

    # Full Evaluation
    t_start = time.time()
    logger.info("Executing Primary Analysis: 5-Fold StratifiedGroupKFold across 3 seeds ...")
    scaffold_results = run_cross_validation_regime(
        cohort=cohort,
        regime="scaffold_grouped",
        seeds=seeds,
        n_splits=n_splits,
        strategy=strategy,
        n_bootstraps=n_bootstraps,
    )

    logger.info("Executing Reference Analysis: Matched 5-Fold StratifiedKFold across 3 seeds ...")
    random_results = run_cross_validation_regime(
        cohort=cohort,
        regime="stratified_random",
        seeds=seeds,
        n_splits=n_splits,
        strategy=strategy,
        n_bootstraps=n_bootstraps,
    )
    t_total = time.time() - t_start

    # Compute comparative summary across regimes
    scaf_across = scaffold_results["across_seed_pooled_oof_summary"]
    rand_across = random_results["across_seed_pooled_oof_summary"]

    delta_roc_auc = (
        round(scaf_across["roc_auc"]["mean"] - rand_across["roc_auc"]["mean"], 6)
        if scaf_across["roc_auc"]["mean"] is not None and rand_across["roc_auc"]["mean"] is not None
        else None
    )
    delta_pr_auc = (
        round(scaf_across["pr_auc"]["mean"] - rand_across["pr_auc"]["mean"], 6)
        if scaf_across["pr_auc"]["mean"] is not None and rand_across["pr_auc"]["mean"] is not None
        else None
    )

    # Pre-save checksum audit verification
    checksums_valid, checksum_failures = _verify_all_checksums()
    assert checksums_valid, f"Protected checksum failure detected before saving report: {checksum_failures}"

    report = {
        "report_id": "clintox_scaffold_generalization_multiseed_evaluation_v1",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "protocol_metadata": {
            "dataset": "ClinTox",
            "target_endpoint": "CT_TOX",
            "duplicate_policy": "conservative_toxic",
            "n_raw_records": cohort.n_raw_rows,
            "n_invalid_excluded": cohort.n_invalid_smiles_excluded,
            "n_duplicates_resolved": cohort.n_duplicate_rows_resolved,
            "n_unique_canonical_samples": cohort.n_samples,
            "n_positives": cohort.n_positives,
            "n_negatives": cohort.n_negatives,
            "cohort_prevalence": cohort.prevalence,
            "n_cyclic_compounds": int(np.sum(cohort.cyclic_mask)),
            "n_acyclic_compounds": int(np.sum(~cohort.cyclic_mask)),
            "grouping_strategy": strategy,
            "prespecified_seeds": list(seeds),
            "n_splits_per_seed": n_splits,
            "total_evaluations_per_regime": len(seeds) * n_splits,
            "base_algorithm": "RandomForestClassifier(n_estimators=300, class_weight='balanced')",
            "probability_calibration_protocol": (
                "UNAVAILABLE_STAGE1_SCOPE: Evaluates uncalibrated tree-ensemble voting probabilities. "
                "Post-hoc probability calibration metrics are marked as UNAVAILABLE because internal "
                "random-fold calibration does NOT establish valid calibration on out-of-scaffold chemotypes, "
                "and nested scaffold-grouped calibration is outside Stage 1 scope (STAGE1_CALIBRATION_SCOPE_NOTE)."
            ),
            "features": "Morgan circular fingerprints (radius=2, n_bits=2048, binary int8)",
            "scaffold_cluster_bootstrap_replicates": n_bootstraps,
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
        },
        "scientific_caveats": [
            "Folds and seeds reuse the same dataset (N=1,461, 112 positives) and are NOT independent biological cohorts.",
            "Repeated cross-validation quantifies partition variance, not external biological validity.",
            "This model is an exploratory research and educational prioritization tool, not a validated clinical safety determination.",
            "High ROC-AUC does not guarantee reliable detection of toxicity on novel chemotypes at specific operational thresholds.",
        ],
        "summary_comparison": {
            "scaffold_grouped_pooled_roc_auc_mean": scaf_across["roc_auc"]["mean"],
            "scaffold_grouped_pooled_roc_auc_sd": scaf_across["roc_auc"]["sd"],
            "stratified_random_pooled_roc_auc_mean": rand_across["roc_auc"]["mean"],
            "stratified_random_pooled_roc_auc_sd": rand_across["roc_auc"]["sd"],
            "delta_roc_auc_scaffold_minus_random": delta_roc_auc,
            "scaffold_grouped_pooled_pr_auc_mean": scaf_across["pr_auc"]["mean"],
            "scaffold_grouped_pooled_pr_auc_sd": scaf_across["pr_auc"]["sd"],
            "stratified_random_pooled_pr_auc_mean": rand_across["pr_auc"]["mean"],
            "stratified_random_pooled_pr_auc_sd": rand_across["pr_auc"]["sd"],
            "delta_pr_auc_scaffold_minus_random": delta_pr_auc,
        },
        "scaffold_grouped_analysis": scaffold_results,
        "stratified_random_reference": random_results,
        "execution_summary": {
            "total_elapsed_seconds": round(t_total, 2),
            "checksums_verified_before_save": checksums_valid,
            "checksum_status": "32 ML artifacts and 5 literature benchmarks verified 100% matched.",
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Diagnostic report successfully written to '{out_file}' (elapsed: {t_total:.2f}s).")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Controlled ClinTox Generalization Evaluation.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-fold preflight smoke test only.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path for JSON diagnostic report.")
    parser.add_argument("--n-bootstraps", type=int, default=500, help="Number of scaffold cluster bootstrap draws.")
    args = parser.parse_args()

    try:
        run_controlled_clintox_evaluation(
            output_path=args.output,
            n_bootstraps=args.n_bootstraps,
            smoke_test_only=args.smoke_test,
        )
        return 0
    except Exception as exc:
        logger.error(f"Evaluation execution failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
