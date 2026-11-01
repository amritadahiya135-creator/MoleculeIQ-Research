"""
Unit tests for Controlled ClinTox Scaffold Generalization Multi-Seed Report (Priority 2D).

Verifies that:
1. artifacts/diagnostics/clintox_scaffold_multiseed_report.json adheres to schema requirements.
2. Complete sample conservation and zero scaffold-group overlap are achieved across all folds.
3. Both primary (scaffold-grouped) and reference (stratified random) regimes report valid metrics.
4. Scaffold cluster bootstrap intervals and invalid replicate statistics are recorded.
5. All 31 ML checksums and 5 literature benchmark checksums remain intact.
"""

import hashlib
import json
from pathlib import Path
import pytest

REPORT_PATH = Path("artifacts/diagnostics/clintox_scaffold_multiseed_report.json")


def test_multiseed_report_schema_and_metadata():
    """Verify the multi-seed diagnostic report exists and contains required metadata."""
    assert REPORT_PATH.exists(), f"Report file must exist at {REPORT_PATH}."
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("status") == "COMPLETED"
    meta = data.get("protocol_metadata", {})
    assert meta.get("dataset") == "ClinTox"
    assert meta.get("target_endpoint") == "CT_TOX"
    assert meta.get("duplicate_policy") == "conservative_toxic"
    assert meta.get("n_unique_canonical_samples") == 1461
    assert meta.get("n_positives") == 112
    assert meta.get("n_negatives") == 1349
    assert meta.get("prespecified_seeds") == [42, 1042, 2042]
    assert meta.get("n_splits_per_seed") == 5

    # Check scientific caveats
    caveats = data.get("scientific_caveats", [])
    assert len(caveats) >= 3
    caveat_text = " ".join(caveats).lower()
    assert "not independent biological cohorts" in caveat_text
    assert "exploratory research" in caveat_text


def test_both_regimes_present_with_five_folds_per_seed():
    """Verify that both scaffold_grouped and stratified_random regimes have 3 seeds and 5 folds each."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for regime_key in ("scaffold_grouped_analysis", "stratified_random_reference"):
        regime_data = data.get(regime_key, {})
        assert regime_data.get("n_seeds") == 3
        assert regime_data.get("n_splits_per_seed") == 5
        assert regime_data.get("total_folds_evaluated") == 15

        per_seed = regime_data.get("per_seed_evaluations", {})
        for seed_str in ("42", "1042", "2042"):
            assert seed_str in per_seed
            seed_eval = per_seed[seed_str]
            assert seed_eval.get("split_integrity_verified") is True
            folds = seed_eval.get("folds", [])
            assert len(folds) == 5

            total_test_samples = sum(f["n_test"] for f in folds)
            assert total_test_samples == 1461, "All 1,461 samples must be evaluated across the 5 folds."

            total_test_positives = sum(f["test_positives"] for f in folds)
            assert total_test_positives == 112, "All 112 positive samples must be evaluated across the 5 folds."

            # Verify metrics validity for each fold
            for fold in folds:
                m = fold.get("metrics", {})
                roc = m.get("roc_auc")
                pr = m.get("pr_auc")
                assert roc is not None and 0.5 <= roc <= 1.0
                assert pr is not None and 0.0 < pr <= 1.0


def test_scaffold_generalization_gap_and_metrics_consistency():
    """Verify that summary comparisons and bootstrap intervals are mathematically consistent."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    comp = data.get("summary_comparison", {})
    scaf_roc = comp.get("scaffold_grouped_pooled_roc_auc_mean")
    rand_roc = comp.get("stratified_random_pooled_roc_auc_mean")
    delta_roc = comp.get("delta_roc_auc_scaffold_minus_random")

    assert scaf_roc is not None and 0.70 <= scaf_roc <= 0.90
    assert rand_roc is not None and 0.75 <= rand_roc <= 0.95
    assert delta_roc is not None
    # Scaffold out-of-distribution ranking should be harder than random in-distribution
    assert delta_roc <= 0.05, "Scaffold ROC-AUC should generally reflect generalization penalty."

    # Check bootstrap intervals in scaffold analysis
    scaf_per_seed = data["scaffold_grouped_analysis"]["per_seed_evaluations"]
    for seed_str in ("42", "1042", "2042"):
        boot = scaf_per_seed[seed_str].get("scaffold_cluster_bootstrap", {})
        roc_boot = boot.get("roc_auc", {})
        assert roc_boot.get("n_requested_replicates") == 500
        assert roc_boot.get("n_valid_replicates", 0) > 400


def test_production_checksums_and_model_weights_unmodified():
    """Verify that all 31 ML checksums and 5 literature checksums remain bit-for-bit identical."""
    ml_manifest = Path("artifacts/checksums.sha256")
    assert ml_manifest.exists()
    for line in ml_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exp_hash, rel_path = line.split(maxsplit=1)
        target = Path("artifacts") / rel_path.strip().replace("\\", "/")
        assert target.exists(), f"Protected ML artifact missing: {target}"
        act_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        assert act_hash == expected_hash if (expected_hash := exp_hash) else True
        assert act_hash == exp_hash, f"Checksum mismatch in {target}"

    lit_manifest = Path("data/benchmarks/literature/checksums.sha256")
    assert lit_manifest.exists()
    for line in lit_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exp_hash, rel_path = line.split(maxsplit=1)
        target = Path(rel_path.strip().replace("\\", "/"))
        assert target.exists(), f"Protected literature benchmark missing: {target}"
        act_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        assert act_hash == exp_hash, f"Checksum mismatch in {target}"
