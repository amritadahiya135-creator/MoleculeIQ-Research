#!/usr/bin/env python3
"""
MoleculeIQ — Release Artifact Downloader & Integrity Verifier (Phase 2E).

Automates the retrieval and cryptographic verification of trained machine learning
models and fingerprint arrays from versioned GitHub Release assets.

This workflow ensures that:
1. Model weights (~405 MB total) do not bloat ordinary Git clone history.
2. Git LFS free bandwidth quotas (1 GB/month) are not exhausted by public clones.
3. Every downloaded artifact is cryptographically verified against SHA-256 checksums.

Usage:
    python scripts/download_release_artifacts.py [--verify-only] [--tag v5.0.0] [--repo amritadahiya135-creator/MoleculeIQ-Research]
"""

import os
import sys
import hashlib
import logging
import argparse
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKSUMS_FILE = REPO_ROOT / "artifacts" / "checksums.sha256"
DEFAULT_TAG = "v5.0.0"
DEFAULT_REPO = "amritadahiya135-creator/MoleculeIQ-Research"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_release_artifacts")


def load_checksum_manifest(manifest_path: Path = CHECKSUMS_FILE) -> Dict[str, str]:
    """Load expected SHA-256 hashes from artifacts/checksums.sha256."""
    if not manifest_path.exists():
        logger.error(f"Checksum manifest not found: {manifest_path}")
        return {}
    checksums = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                sha, rel_path = parts[0].strip(), parts[1].strip().lstrip("*")
                # Normalize path separators
                norm_path = str(Path(rel_path))
                checksums[norm_path] = sha
    return checksums


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_all_local_artifacts(manifest_path: Path = CHECKSUMS_FILE) -> Tuple[bool, List[str], List[str]]:
    """
    Check if all artifacts declared in the manifest exist locally and match SHA-256.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (all_passed, missing_files, mismatched_files)
    """
    checksums = load_checksum_manifest(manifest_path)
    if not checksums:
        return False, ["artifacts/checksums.sha256"], []

    missing = []
    mismatched = []

    for rel_path, expected_hash in checksums.items():
        # Checksums manifest paths are relative to the artifacts directory
        full_path = REPO_ROOT / "artifacts" / rel_path
        if not full_path.exists():
            missing.append(rel_path)
            continue
        actual_hash = compute_sha256(full_path)
        if actual_hash.lower() != expected_hash.lower():
            mismatched.append(rel_path)

    all_passed = (len(missing) == 0 and len(mismatched) == 0)
    return all_passed, missing, mismatched


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from: {url}")
    logger.info(f"Destination: {dest_path}")
    try:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, (block_num * block_size / total_size) * 100.0)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({block_num * block_size // 1024 // 1024} MB / {total_size // 1024 // 1024} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
        sys.stdout.write("\n")
        logger.info("Download completed successfully.")
        return True
    except Exception as exc:
        sys.stdout.write("\n")
        logger.error(f"Download failed: {exc}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_bundle(archive_path: Path, dest_dir: Path = REPO_ROOT) -> bool:
    """
    Extract a release bundle securely using PEP 706 filter='data'.

    Fails closed if the runtime Python environment does not support PEP 706
    secure tar extraction filters (tarfile.data_filter).

    Parameters
    ----------
    archive_path : Path
        Path to the tar.gz archive to extract.
    dest_dir : Path
        Target destination directory.

    Returns
    -------
    bool
        True if extraction succeeded, False otherwise.
    """
    if not hasattr(tarfile, "data_filter"):
        logger.error(
            "Secure archive extraction requires PEP 706 tarfile filter support "
            "(Python 3.12+ or security-patched 3.10/3.11). 'tarfile.data_filter' "
            "is not available in the current Python runtime. Unrestricted extraction "
            "is refused to prevent path traversal and arbitrary file overwrite attacks."
        )
        return False

    logger.info(f"Extracting '{archive_path.name}' into '{dest_dir}'...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Enforce PEP 706 data filter (rejects path traversal, unsafe symlinks/devices)
            tar.extractall(path=dest_dir, filter="data")
        logger.info("Extraction completed successfully.")
        return True
    except Exception as exc:
        logger.error(f"Failed to extract bundle: {exc}")
        return False


def download_and_extract_bundle(repo: str, tag: str) -> bool:
    """Download consolidated model archive from GitHub Release assets."""
    bundle_name = f"moleculeiq-models-{tag}.tar.gz"
    url = f"https://github.com/{repo}/releases/download/{tag}/{bundle_name}"
    tmp_dest = REPO_ROOT / "artifacts" / bundle_name

    logger.info(f"Attempting to download release asset bundle '{bundle_name}' from GitHub Releases...")
    if not download_file(url, tmp_dest):
        return False

    success = extract_bundle(tmp_dest, REPO_ROOT)
    if tmp_dest.exists():
        tmp_dest.unlink()
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Verify or download MoleculeIQ model release assets."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing local artifacts against artifacts/checksums.sha256.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Extract directly from a local archive path instead of downloading from GitHub.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=DEFAULT_TAG,
        help=f"Release tag on GitHub (default: {DEFAULT_TAG}).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help=f"Target GitHub repository (default: {DEFAULT_REPO}).",
    )
    args = parser.parse_args()

    logger.info("Checking local artifact integrity against artifacts/checksums.sha256...")
    passed, missing, mismatched = verify_all_local_artifacts()

    if passed:
        logger.info(f"VERIFICATION PASSED: All {len(load_checksum_manifest())} model and diagnostic artifacts are present and cryptographically verified.")
        sys.exit(0)

    if args.verify_only:
        logger.error(f"Verification FAILED: {len(missing)} missing files, {len(mismatched)} checksum mismatches.")
        if missing:
            logger.error(f"Missing: {missing[:5]}...")
        if mismatched:
            logger.error(f"Mismatched: {mismatched[:5]}...")
        sys.exit(1)

    if args.archive:
        if not args.archive.exists():
            logger.error(f"Specified archive does not exist: {args.archive}")
            sys.exit(1)
        logger.info(f"Extracting local archive '{args.archive}' into repository root...")
        if not extract_bundle(args.archive, REPO_ROOT):
            logger.error("Local archive extraction failed.")
            sys.exit(1)
    else:
        logger.warning(f"Local artifact check found {len(missing)} missing and {len(mismatched)} mismatched files.")
        logger.info(f"Initiating release asset download from {args.repo} (tag: {args.tag})...")

        success = download_and_extract_bundle(args.repo, args.tag)
        if not success:
            logger.error(
                "Download from GitHub Releases failed or release asset is not yet published.\n"
                "If working in a local development environment, ensure artifacts were copied from the verified external backup."
            )
            sys.exit(1)

    passed_after, missing_after, mismatched_after = verify_all_local_artifacts()
    if passed_after:
        logger.info("Post-download verification PASSED: All artifacts cryptographically verified.")
        sys.exit(0)
    else:
        logger.error(f"Post-download verification FAILED: {len(missing_after)} missing, {len(mismatched_after)} mismatched.")
        sys.exit(1)


if __name__ == "__main__":
    main()
