"""Model and data artifact integrity verification (SHA-256 cryptographic digests)."""

import os
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChecksumVerificationResult:
    """Immutable record of artifact checksum verification outcome."""
    is_valid: bool
    verified_count: int
    failed_checksums: List[str]
    missing_files: List[str]
    error_message: Optional[str] = None


def compute_file_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Compute the SHA-256 hexadecimal hash digest of a file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the target file.
    chunk_size : int, default 65536
        Chunk size in bytes for streaming digest calculation.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 hash.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Target file for hashing not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_checksums_manifest(
    artifacts_dir: Union[str, Path] = ARTIFACTS_DIR,
    output_path: Optional[Union[str, Path]] = None,
    extensions: Tuple[str, ...] = (".joblib", ".json", ".npz", ".csv"),
) -> Path:
    """
    Generate SHA-256 manifest for all relevant artifact files in the directory.

    Parameters
    ----------
    artifacts_dir : Union[str, Path]
        Root directory containing artifacts.
    output_path : Optional[Union[str, Path]]
        Destination path for manifest file. Defaults to `artifacts_dir / 'checksums.sha256'`.
    extensions : Tuple[str, ...]
        File extensions to scan and include.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    root_dir = Path(artifacts_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Artifacts directory does not exist: {root_dir}")

    dest = Path(output_path) if output_path else root_dir / "checksums.sha256"

    manifest_lines: List[str] = []
    # Collect all matching files sorted for reproducibility
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix in extensions and path.name != "checksums.sha256":
            rel_path = path.relative_to(root_dir).as_posix()
            digest = compute_file_sha256(path)
            manifest_lines.append(f"{digest}  {rel_path}\n")

    with open(dest, "w", encoding="utf-8") as f:
        f.writelines(manifest_lines)

    logger.info(f"Generated SHA-256 manifest with {len(manifest_lines)} entries at {dest}")
    return dest


def verify_artifact_checksums(
    manifest_path: Optional[Union[str, Path]] = None,
    artifacts_dir: Union[str, Path] = ARTIFACTS_DIR,
) -> ChecksumVerificationResult:
    """
    Verify artifact integrity against recorded SHA-256 manifest entries.

    Parameters
    ----------
    manifest_path : Optional[Union[str, Path]]
        Path to checksums.sha256 manifest. Defaults to `artifacts_dir / 'checksums.sha256'`.
    artifacts_dir : Union[str, Path]
        Directory against which relative paths in manifest are resolved.

    Returns
    -------
    ChecksumVerificationResult
        Structured verification outcome with passed/failed counts and error lists.
    """
    root_dir = Path(artifacts_dir).resolve()
    manifest = Path(manifest_path) if manifest_path else root_dir / "checksums.sha256"

    if not manifest.exists():
        return ChecksumVerificationResult(
            is_valid=False,
            verified_count=0,
            failed_checksums=[],
            missing_files=[str(manifest)],
            error_message=f"Checksums manifest not found: {manifest}",
        )

    verified_count = 0
    failed_checksums: List[str] = []
    missing_files: List[str] = []

    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            expected_hash, rel_path = parts[0], parts[1].strip()
            # Normalize path separator
            file_path = root_dir / Path(rel_path)

            if not file_path.exists():
                missing_files.append(rel_path)
                continue

            actual_hash = compute_file_sha256(file_path)
            if actual_hash.lower() != expected_hash.lower():
                failed_checksums.append(
                    f"{rel_path} (expected: {expected_hash[:8]}..., actual: {actual_hash[:8]}...)"
                )
            else:
                verified_count += 1

    is_valid = (len(failed_checksums) == 0 and len(missing_files) == 0 and verified_count > 0)
    error_msg = None
    if not is_valid:
        error_msg = (
            f"Artifact integrity check failed: {len(failed_checksums)} hash mismatch(es), "
            f"{len(missing_files)} missing file(s)."
        )

    return ChecksumVerificationResult(
        is_valid=is_valid,
        verified_count=verified_count,
        failed_checksums=failed_checksums,
        missing_files=missing_files,
        error_message=error_msg,
    )
