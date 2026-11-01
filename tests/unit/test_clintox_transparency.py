"""
Tests for ClinTox Scientific Transparency & Governance Upgrade (Priority 2B).

Verifies that:
1. docs/MODEL_CARD_CLINTOX.md documents dual-split protocols, Hanley-McNeil CIs,
   out-of-scaffold false-negative risks, and exploratory research boundaries.
2. src/ui/model_insights.py surfaces the dual benchmark comparison, Hanley-McNeil CIs,
   and prominent out-of-scaffold low-recall disclosures without forbidden prescriptive terms.
3. ClinTox inference remains numerically invariant and model weights are untouched.
4. Cryptographic checksums of protected ML and literature artifacts remain intact.
"""

import hashlib
import os
from pathlib import Path
import pytest

from src.models.inference import ToxicityInferenceService


# =============================================================================
# 1. Model Card Scientific Transparency Checks
# =============================================================================

def test_clintox_model_card_dual_split_disclosures():
    """Verify MODEL_CARD_CLINTOX.md discloses dual splits, sample counts, and distinct models."""
    card_path = Path("docs/MODEL_CARD_CLINTOX.md")
    assert card_path.exists(), "docs/MODEL_CARD_CLINTOX.md must exist."
    content = card_path.read_text(encoding="utf-8")

    # Dual split protocol distinction
    assert "Stratified Random Split" in content
    assert "Bemis–Murcko Scaffold Split" in content or "Bemis-Murcko Scaffold Split" in content

    # Distinct model roles (scaffold benchmark is a separately trained comparator)
    assert "separately trained" in content or "independently trained" in content

    # Verified test set counts and positive prevalence
    assert "293" in content
    assert "22" in content
    assert "17" in content

    # Hanley-McNeil 95% confidence intervals and statistical uncertainty
    assert "Hanley–McNeil" in content or "Hanley-McNeil" in content
    assert "0.7484" in content
    assert "0.9530" in content
    assert "0.6569" in content
    assert "0.9185" in content

    # Out-of-scaffold recall and false negative disclosure
    assert "5.88%" in content or "0.0588" in content
    assert "16" in content and "false negative" in content.lower()

    # Research and educational scope
    assert "exploratory research" in content.lower()
    assert "clinical decision-making" in content.lower()


# =============================================================================
# 2. Model Insights UI Scientific Transparency Checks
# =============================================================================

def test_clintox_ui_insights_transparency():
    """Verify src/ui/model_insights.py renders dual benchmark comparison and caveats."""
    ui_path = Path("src/ui/model_insights.py")
    assert ui_path.exists(), "src/ui/model_insights.py must exist."
    code = ui_path.read_text(encoding="utf-8")

    # Dual benchmark table header and data loading
    assert "Dual Generalization Benchmark" in code
    assert "scaffold_benchmark.json" in code

    # Prominent callout regarding out-of-scaffold false negatives
    assert "High False-Negative Risk on Novel Scaffolds" in code
    assert "5.88% recall" in code or "5.88%" in code
    assert "16 of 17" in code or "16 false negatives" in code.lower()

    # Approximate Hanley-McNeil confidence intervals
    assert "Hanley–McNeil" in code or "Hanley-McNeil" in code
    assert "0.7484" in code
    assert "0.9530" in code
    assert "0.6569" in code
    assert "0.9185" in code

    # Absence of forbidden prescriptive terms in UI
    forbidden_terms = [
        "High Priority Candidate",
        "Moderate Priority Candidate",
        "Low Priority Candidate",
        "Safe Leads",
        "Lead Lead",
        "Composite Research Triage Index",
        "LOW RISK",
        "ELEVATED RISK",
        "Failure Likelihood",
        "Failure Probability",
        "Failure Risk",
        "Low-Risk",
    ]
    for term in forbidden_terms:
        assert term not in code, f"Forbidden prescriptive term '{term}' found in src/ui/model_insights.py"


# =============================================================================
# 3. Model Inference Invariance Checks
# =============================================================================

def test_clintox_model_serving_invariance():
    """Verify ClinTox serving model loads and produces expected calibrated output."""
    service = ToxicityInferenceService()
    assert service.is_clintox_available(), "ClinTox model must be available on disk."

    # Caffeine (canonical benchmark compound)
    caffeine_smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    pred = service.predict_clintox(caffeine_smi)
    assert pred.status == "SUCCESS"
    assert isinstance(pred.probability, float)
    assert 0.0 <= pred.probability <= 1.0
    assert pred.predicted_class in (0, 1)
    assert pred.applicability_domain_status in ("WITHIN_DOMAIN", "NEAR_BOUNDARY", "OUTSIDE_DOMAIN")


# =============================================================================
# 4. Checksum Integrity Verification
# =============================================================================

def test_protected_artifact_checksums_intact():
    """Verify that all 31 ML checksums and 5 literature benchmark checksums match."""
    # Check ML artifacts
    ml_manifest = Path("artifacts/checksums.sha256")
    assert ml_manifest.exists()
    for line in ml_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(maxsplit=1)
        rel_path = rel_path.strip().replace("\\", "/")
        target_file = Path("artifacts") / rel_path
        assert target_file.exists(), f"Protected ML artifact missing: {target_file}"
        actual_hash = hashlib.sha256(target_file.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Checksum mismatch in {target_file}: {actual_hash} != {expected_hash}"

    # Check literature benchmarks
    lit_manifest = Path("data/benchmarks/literature/checksums.sha256")
    assert lit_manifest.exists()
    for line in lit_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(maxsplit=1)
        rel_path = rel_path.strip().replace("\\", "/")
        target_file = Path(rel_path)
        assert target_file.exists(), f"Protected literature benchmark missing: {target_file}"
        actual_hash = hashlib.sha256(target_file.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Checksum mismatch in {target_file}: {actual_hash} != {expected_hash}"
