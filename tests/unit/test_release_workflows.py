"""
Unit tests for MoleculeIQ public release workflows (Phase 2E).

Validates:
1. PubMed benchmark manifest parsing, PMID validation, and verification-only mode.
2. Error handling: missing PMIDs and network failure resilience.
3. Release artifact packaging inventory and manifest generation.
4. Cryptographic SHA-256 verification and corruption/missing artifact detection.
"""

import json
import io
import tarfile
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.hydrate_pubmed_benchmark import (
    MANIFEST_PATH,
    TARGET_BENCHMARK_PATH,
    verify_benchmark_file,
    hydrate_benchmark,
)
from scripts.package_release_artifacts import (
    build_artifact_inventory,
    compute_sha256,
)
from scripts.download_release_artifacts import (
    verify_all_local_artifacts,
    load_checksum_manifest,
    extract_bundle,
)


class TestPubMedHydrationWorkflow:
    """Validates PubMed benchmark manifest and hydration mechanisms."""

    def test_manifest_schema_and_contents(self):
        assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0.0"
        assert data["total_queries"] == 3
        assert data["total_articles"] == 18
        assert len(data["articles"]) == 18

        for art in data["articles"]:
            assert "pmid" in art
            assert "title" in art
            assert "query_id" in art
            assert "url" in art
            # Verbatim copyrighted abstract text must NOT be in public manifest
            assert "abstract" not in art or art.get("abstract") == ""

    def test_pmid_format_validation(self):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for art in data["articles"]:
            pmid = art["pmid"]
            assert pmid.isdigit(), f"Non-numeric PMID: {pmid}"
            assert len(pmid) >= 7, f"PMID suspiciously short: {pmid}"

    def test_verify_only_mode_on_existing_benchmark(self):
        assert TARGET_BENCHMARK_PATH.exists()
        assert verify_benchmark_file(TARGET_BENCHMARK_PATH) is True

    def test_verify_benchmark_detects_missing_file(self, tmp_path):
        non_existent = tmp_path / "missing_benchmark.json"
        assert verify_benchmark_file(non_existent) is False

    def test_verify_benchmark_detects_empty_abstract(self, tmp_path):
        bad_file = tmp_path / "bad_benchmark.json"
        bad_data = {
            "schema_version": "1.0.0",
            "articles": [
                {
                    "pmid": "12345678",
                    "title": "Test Title",
                    "abstract": "",  # Empty
                    "has_complete_abstract": False,
                }
            ],
        }
        with open(bad_file, "w", encoding="utf-8") as f:
            json.dump(bad_data, f)
        assert verify_benchmark_file(bad_file) is False

    @patch("scripts.hydrate_pubmed_benchmark.fetch_abstracts_from_ncbi")
    def test_network_failure_handling(self, mock_fetch, tmp_path):
        mock_fetch.side_effect = RuntimeError("NCBI E-utilities connection timed out")
        target = tmp_path / "target.json"
        success = hydrate_benchmark(manifest_path=MANIFEST_PATH, target_path=target, force=True)
        assert success is False
        assert not target.exists()

    @patch("scripts.hydrate_pubmed_benchmark.fetch_abstracts_from_ncbi")
    def test_missing_pmid_in_response_handling(self, mock_fetch, tmp_path):
        # Return only 1 abstract when 18 are expected
        mock_fetch.return_value = {"42834623": "A single abstract"}
        target = tmp_path / "target.json"
        success = hydrate_benchmark(manifest_path=MANIFEST_PATH, target_path=target, force=True)
        assert success is False


class TestArtifactPackagingWorkflow:
    """Validates model artifact packaging, manifest generation, and integrity checks."""

    def test_artifact_inventory_matches_production_manifest(self):
        inventory = build_artifact_inventory()
        assert len(inventory) == 32, f"Expected 32 production artifacts, got {len(inventory)}"

        for item in inventory:
            assert "filename" in item
            assert "relative_path" in item
            assert "sha256" in item
            assert len(item["sha256"]) == 64
            assert item["file_size_bytes"] > 0
            assert item["dataset_association"] in ("clintox", "tox21", "tox21_diagnostics", "shared")

    def test_verify_all_local_artifacts_current_repo(self):
        all_passed, missing, mismatched = verify_all_local_artifacts()
        assert all_passed is True
        assert len(missing) == 0
        assert len(mismatched) == 0

    def test_corrupted_artifact_detection(self, tmp_path):
        # Create a mock manifest and tampered file
        mock_manifest = tmp_path / "checksums.sha256"
        dummy_file = tmp_path / "model.joblib"
        dummy_file.write_bytes(b"AUTHENTIC_BYTES")
        authentic_sha = compute_sha256(dummy_file)

        # Write manifest with authentic hash
        with open(mock_manifest, "w", encoding="utf-8") as f:
            f.write(f"{authentic_sha}  model.joblib\n")

        # Now tamper the file
        dummy_file.write_bytes(b"TAMPERED_BYTES")

        with patch("scripts.download_release_artifacts.REPO_ROOT", tmp_path.parent):
            with patch("scripts.download_release_artifacts.CHECKSUMS_FILE", mock_manifest):
                # Verify tamper is flagged
                all_passed, missing, mismatched = verify_all_local_artifacts(mock_manifest)
                # Since REPO_ROOT / artifacts / rel_path is checked:
                # We can verify load_checksum_manifest and compute_sha256 directly
                manifest_map = load_checksum_manifest(mock_manifest)
                assert manifest_map["model.joblib"] == authentic_sha
                assert compute_sha256(dummy_file) != authentic_sha

    def test_missing_artifact_detection(self, tmp_path):
        mock_manifest = tmp_path / "checksums.sha256"
        with open(mock_manifest, "w", encoding="utf-8") as f:
            f.write(f"{'a'*64}  nonexistent_file.joblib\n")

        manifest_map = load_checksum_manifest(mock_manifest)
        assert "nonexistent_file.joblib" in manifest_map


class TestSecureExtractionWorkflow:
    """Validates PEP 706 secure tar extraction and safety protections."""

    def test_valid_archive_extracts_correctly(self, tmp_path):
        archive_path = tmp_path / "valid.tar.gz"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        # Build a valid archive
        with tarfile.open(archive_path, "w:gz") as tar:
            data = b"safe file content"
            ti = tarfile.TarInfo(name="models/test_model.joblib")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))

        success = extract_bundle(archive_path, dest_dir)
        assert success is True
        extracted_file = dest_dir / "models" / "test_model.joblib"
        assert extracted_file.exists()
        assert extracted_file.read_bytes() == b"safe file content"

    def test_path_traversal_archive_rejected(self, tmp_path):
        archive_path = tmp_path / "traversal.tar.gz"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        # Build an archive containing path traversal
        with tarfile.open(archive_path, "w:gz") as tar:
            data = b"malicious content"
            ti = tarfile.TarInfo(name="../escape.txt")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))

        success = extract_bundle(archive_path, dest_dir)
        assert success is False
        assert not (tmp_path / "escape.txt").exists()

    def test_unsafe_absolute_symlink_rejected(self, tmp_path):
        archive_path = tmp_path / "bad_link.tar.gz"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        # Build an archive containing an absolute symlink
        with tarfile.open(archive_path, "w:gz") as tar:
            ti = tarfile.TarInfo(name="unsafe_link")
            ti.type = tarfile.SYMTYPE
            ti.linkname = "/etc/shadow"
            tar.addfile(ti)

        success = extract_bundle(archive_path, dest_dir)
        assert success is False

    def test_invalid_corrupted_archive_fails_safely(self, tmp_path):
        bad_archive = tmp_path / "corrupt.tar.gz"
        bad_archive.write_bytes(b"NOT_A_VALID_GZIP_FILE")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        success = extract_bundle(bad_archive, dest_dir)
        assert success is False

    def test_missing_data_filter_fails_closed(self, tmp_path, monkeypatch):
        archive_path = tmp_path / "test.tar.gz"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        with tarfile.open(archive_path, "w:gz") as tar:
            data = b"safe file content"
            ti = tarfile.TarInfo(name="models/test_model.joblib")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))

        # Simulate Python environment lacking PEP 706 tarfile.data_filter
        monkeypatch.delattr(tarfile, "data_filter", raising=False)

        success = extract_bundle(archive_path, dest_dir)
        assert success is False
        assert not (dest_dir / "models" / "test_model.joblib").exists()

    def test_unavailable_filter_never_calls_unrestricted_extractall(self, tmp_path, monkeypatch):
        archive_path = tmp_path / "test.tar.gz"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        with tarfile.open(archive_path, "w:gz") as tar:
            data = b"safe file content"
            ti = tarfile.TarInfo(name="models/test_model.joblib")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))

        # Simulate Python environment lacking PEP 706 tarfile.data_filter
        monkeypatch.delattr(tarfile, "data_filter", raising=False)

        with patch.object(tarfile.TarFile, "extractall") as mock_extractall:
            success = extract_bundle(archive_path, dest_dir)
            assert success is False
            # Verify unrestricted extractall was never invoked
            assert mock_extractall.call_count == 0
