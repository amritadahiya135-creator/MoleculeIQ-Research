"""Unit tests for deployment readiness, model hydration, and unavailable-state handling."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.download_release_artifacts import (
    verify_all_local_artifacts,
    ensure_artifacts_hydrated,
    load_checksum_manifest,
    compute_sha256,
    CHECKSUMS_FILE,
    REPO_ROOT,
)
from src.chemistry.validation import MoleculeValidationResult
from src.chemistry.descriptors import calculate_descriptors
from src.services.contracts import MoleculeAnalysisResult
from src.models.inference import ClinToxPrediction, Tox21MultiPrediction, Tox21EndpointPrediction
from src.ranking.scoring import PrioritizationScore
from src.similarity.index import SimilaritySearchResult, SimilarityMatch
from src.ui.components import (
    render_triage_editorial_grid,
    render_executive_interpretation,
    render_risk_gauge,
)
from src.services.orchestration import ScreeningOrchestrator


def test_packages_txt_presence_and_contents():
    """Verify packages.txt exists at repository root with required Linux libraries."""
    pkg_file = REPO_ROOT / "packages.txt"
    assert pkg_file.exists(), "packages.txt must exist at repository root"
    content = pkg_file.read_text(encoding="utf-8").strip().splitlines()
    packages = [line.strip() for line in content if line.strip() and not line.startswith("#")]
    assert "libxrender1" in packages
    assert "libxext6" in packages
    assert "libsm6" in packages


def test_streamlit_config_modernization():
    """Verify .streamlit/config.toml contains toolbarMode=viewer and no deprecated enableCORS."""
    cfg_file = REPO_ROOT / ".streamlit" / "config.toml"
    assert cfg_file.exists()
    content = cfg_file.read_text(encoding="utf-8")
    assert "enableCORS" not in content, "enableCORS is deprecated and must not be present"
    assert "toolbarMode = \"viewer\"" in content or "toolbarMode=\"viewer\"" in content
    assert "headless = true" in content


def test_case_a_all_model_artifacts_available():
    """Test Case A: All local model artifacts are present and cryptographically verified."""
    passed, missing, mismatched = verify_all_local_artifacts()
    assert passed is True, f"Expected all artifacts present. Missing: {missing}, Mismatched: {mismatched}"
    assert len(missing) == 0
    assert len(mismatched) == 0

    success, msg = ensure_artifacts_hydrated()
    assert success is True
    assert "All model artifacts verified locally" in msg


def test_case_b_missing_artifacts_trigger_download_and_verify(tmp_path):
    """Test Case B: Missing artifacts trigger download and post-extraction verification."""
    # Create a dummy file and compute its sha
    dummy_file = tmp_path / "model.joblib"
    dummy_file.write_bytes(b"dummy_model_bytes_12345")
    dummy_hash = compute_sha256(dummy_file)

    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{dummy_hash}  test_model/model.joblib\n", encoding="utf-8")

    # Target destination
    target_artifacts = tmp_path / "artifacts"
    target_model = target_artifacts / "test_model" / "model.joblib"

    # Initially missing
    assert not target_model.exists()

    with patch("scripts.download_release_artifacts.CHECKSUMS_FILE", manifest), \
         patch("scripts.download_release_artifacts.REPO_ROOT", tmp_path):
        
        # Verify reports missing
        passed, missing, mismatched = verify_all_local_artifacts(manifest_path=manifest)
        assert passed is False
        assert len(missing) == 1

        # Simulate download_and_extract_bundle writing the target file
        def fake_download_and_extract(repo, tag):
            target_model.parent.mkdir(parents=True, exist_ok=True)
            target_model.write_bytes(b"dummy_model_bytes_12345")
            return True

        with patch("scripts.download_release_artifacts.download_and_extract_bundle", side_effect=fake_download_and_extract):
            ok, msg = ensure_artifacts_hydrated(manifest_path=manifest)
            assert ok is True
            assert "downloaded and verified successfully" in msg


def test_failed_download_never_reports_success(tmp_path):
    """Verify that a network or extraction error returns failure and never claims success."""
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text("abcdef123456  missing_dir/model.joblib\n", encoding="utf-8")

    with patch("scripts.download_release_artifacts.CHECKSUMS_FILE", manifest), \
         patch("scripts.download_release_artifacts.REPO_ROOT", tmp_path), \
         patch("scripts.download_release_artifacts.download_and_extract_bundle", return_value=False):
        
        ok, msg = ensure_artifacts_hydrated(manifest_path=manifest)
        assert ok is False
        assert "Failed to download or extract" in msg


def test_clintox_successful_prediction_display():
    """Verify that a SUCCESS prediction renders probability, risk badge, and priority score."""
    clintox_res = ClinToxPrediction(
        probability=0.742,
        predicted_class=1,
        is_toxic=True,
        model_version="clintox_rf_calibrated",
        task_definition="Clinical trial failure due to toxicity (CT_TOX)",
        status="SUCCESS",
        applicability_domain_status="WITHIN_DOMAIN",
        applicability_score=0.65,
    )
    prio_res = PrioritizationScore(
        safety_score=25.8,
        druglikeness_score=100.0,
        similarity_score=62.0,
        overall_score=55.3,
        weight_safety=0.5,
        weight_druglikeness=0.3,
        weight_similarity=0.2,
        is_complete=True,
        status="SUCCESS",
    )
    analysis = MoleculeAnalysisResult(
        validation=MoleculeValidationResult(original_smiles="CC(=O)Oc1ccccc1C(=O)O", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O", is_valid=True),
        descriptors=calculate_descriptors("CC(=O)Oc1ccccc1C(=O)O"),
        clintox=clintox_res,
        tox21=None,
        similarity=SimilaritySearchResult(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            matches=[SimilarityMatch(reference_id="CHEMBL25", name="Aspirin", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O", similarity=1.0)],
            best_similarity=1.0,
            total_references=1396,
            status="SUCCESS",
        ),
        prioritization=prio_res,
        explanation=None,
        analysis_timestamp="2026-11-11T00:00:00Z",
        is_valid=True,
        status="SUCCESS",
    )

    with patch("streamlit.markdown") as mock_markdown:
        render_triage_editorial_grid(analysis)
        rendered_html = "".join(str(call.args[0]) for call in mock_markdown.call_args_list)
        assert "74.2%" in rendered_html
        assert "Higher Predicted ClinTox Probability" in rendered_html
        assert "55.3" in rendered_html  # Priority score
        assert "0.0%" not in rendered_html

    with patch("streamlit.markdown") as mock_markdown:
        render_executive_interpretation(analysis)
        rendered_html = "".join(str(call.args[0]) for call in mock_markdown.call_args_list)
        assert "Higher Predicted ClinTox Probability" in rendered_html
        assert "74.2%" in rendered_html


def test_clintox_unavailable_display_prevents_false_negative():
    """Verify that an UNAVAILABLE status displays N/A and NEVER displays 0.0% or implies safety."""
    clintox_unavail = ClinToxPrediction(
        probability=0.0,
        predicted_class=0,
        is_toxic=False,
        model_version="unavailable",
        task_definition="Clinical trial failure due to toxicity (CT_TOX)",
        status="UNAVAILABLE",
        applicability_domain_status="UNAVAILABLE",
        error_message="Model artifact missing",
    )
    # When ClinTox is missing, prioritization is incomplete
    prio_incomplete = PrioritizationScore(
        safety_score=0.0,
        druglikeness_score=100.0,
        similarity_score=62.0,
        overall_score=0.0,
        weight_safety=0.5,
        weight_druglikeness=0.3,
        weight_similarity=0.2,
        is_complete=False,
        status="INCOMPLETE_DATA",
    )
    analysis = MoleculeAnalysisResult(
        validation=MoleculeValidationResult(original_smiles="CC(=O)Oc1ccccc1C(=O)O", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O", is_valid=True),
        descriptors=calculate_descriptors("CC(=O)Oc1ccccc1C(=O)O"),
        clintox=clintox_unavail,
        tox21=None,
        similarity=SimilaritySearchResult(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            matches=[SimilarityMatch(reference_id="CHEMBL25", name="Aspirin", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O", similarity=1.0)],
            best_similarity=1.0,
            total_references=1396,
            status="SUCCESS",
        ),
        prioritization=prio_incomplete,
        explanation=None,
        analysis_timestamp="2026-11-11T00:00:00Z",
        is_valid=True,
        status="SUCCESS",
    )

    with patch("streamlit.markdown") as mock_markdown:
        render_triage_editorial_grid(analysis)
        rendered_html = "".join(str(call.args[0]) for call in mock_markdown.call_args_list)
        # CRITICAL ASSERTIONS:
        assert "Model Unavailable" in rendered_html
        assert "N/A" in rendered_html
        assert "0.0%" not in rendered_html, "Must not display 0.0% when model is unavailable!"
        assert "Lower Predicted ClinTox Probability" not in rendered_html, "Must not claim lower probability when unavailable!"

    with patch("streamlit.markdown") as mock_markdown:
        render_executive_interpretation(analysis)
        rendered_html = "".join(str(call.args[0]) for call in mock_markdown.call_args_list)
        assert "ClinTox Model Unavailable" in rendered_html
        assert "0.0%" not in rendered_html
        assert "Lower Predicted ClinTox Probability" not in rendered_html

    with patch("streamlit.markdown") as mock_markdown:
        render_risk_gauge(clintox_unavail.probability, clintox_unavail.is_toxic, status=clintox_unavail.status)
        rendered_html = "".join(str(call.args[0]) for call in mock_markdown.call_args_list)
        assert "MODEL UNAVAILABLE" in rendered_html


def test_tox21_unavailable_state_display():
    """Verify that Tox21 unavailable state reports models unavailable instead of 0 active."""
    analysis = MoleculeAnalysisResult(
        validation=MoleculeValidationResult(original_smiles="CC(=O)Oc1ccccc1C(=O)O", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O", is_valid=True),
        descriptors=None,
        clintox=None,
        tox21=None,  # Missing Tox21
        similarity=None,
        prioritization=None,
        explanation=None,
        analysis_timestamp="2026-11-11T00:00:00Z",
        is_valid=True,
        status="SUCCESS",
    )
    with patch("streamlit.markdown") as mock_markdown:
        render_executive_interpretation(analysis)
        rendered_html = "".join(str(call.args[0]) for call in mock_markdown.call_args_list)
        assert "Tox21 assay models are currently unavailable." in rendered_html
        assert "0 of 12" not in rendered_html


def test_orchestrator_end_to_end_local_execution():
    """Verify that the screening orchestrator runs end-to-end on a test compound."""
    orchestrator = ScreeningOrchestrator()
    res = orchestrator.analyze_single_molecule("CC(=O)Oc1ccccc1C(=O)O", compute_explanation=False)
    assert res.is_valid is True
    assert res.descriptors is not None
    assert res.clintox is not None
    assert res.clintox.status == "SUCCESS"
    assert 0.0 <= res.clintox.probability <= 1.0
    assert res.tox21 is not None
    assert res.tox21.status == "SUCCESS"
    assert len(res.tox21.endpoints) == 12
    assert res.similarity is not None
    assert res.similarity.best_similarity > 0.99
    assert res.prioritization is not None
    assert res.prioritization.is_complete is True
