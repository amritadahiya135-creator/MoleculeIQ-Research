"""Unit tests for artifact cryptographic integrity and checksum verification."""

import tempfile
from pathlib import Path
import pytest

from src.config import ARTIFACTS_DIR
from src.utils.integrity import (
    compute_file_sha256,
    generate_checksums_manifest,
    verify_artifact_checksums,
)


class TestArtifactIntegrity:
    """Verifies SHA-256 cryptographic hashes and tampering detection."""

    def test_compute_file_sha256_known_string(self, tmp_path):
        test_file = tmp_path / "hello.txt"
        test_file.write_text("MoleculeIQ Scientific Validation\n", encoding="utf-8")
        digest = compute_file_sha256(test_file)
        assert len(digest) == 64
        # Deterministic
        assert digest == compute_file_sha256(test_file)

    def test_repository_artifacts_verify_against_manifest(self):
        manifest_path = ARTIFACTS_DIR / "checksums.sha256"
        assert manifest_path.exists(), "artifacts/checksums.sha256 must exist."

        result = verify_artifact_checksums(manifest_path, ARTIFACTS_DIR)
        assert result.is_valid is True
        assert result.verified_count >= 30
        assert len(result.failed_checksums) == 0
        assert len(result.missing_files) == 0

    def test_tamper_detection(self, tmp_path):
        dummy_dir = tmp_path / "artifacts"
        dummy_dir.mkdir()
        file1 = dummy_dir / "model.joblib"
        file1.write_bytes(b"MODEL_WEIGHTS_ORIGINAL")

        manifest = generate_checksums_manifest(dummy_dir, dummy_dir / "checksums.sha256")
        res1 = verify_artifact_checksums(manifest, dummy_dir)
        assert res1.is_valid is True
        assert res1.verified_count == 1

        # Tamper with file
        file1.write_bytes(b"MODEL_WEIGHTS_TAMPERED")
        res2 = verify_artifact_checksums(manifest, dummy_dir)
        assert res2.is_valid is False
        assert len(res2.failed_checksums) == 1

    def test_missing_file_detection(self, tmp_path):
        dummy_dir = tmp_path / "artifacts"
        dummy_dir.mkdir()
        file1 = dummy_dir / "model.joblib"
        file1.write_bytes(b"TEST")

        manifest = generate_checksums_manifest(dummy_dir, dummy_dir / "checksums.sha256")
        file1.unlink()  # delete file

        res = verify_artifact_checksums(manifest, dummy_dir)
        assert res.is_valid is False
        assert "model.joblib" in res.missing_files
