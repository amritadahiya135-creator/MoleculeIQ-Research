#!/usr/bin/env python3
"""
MoleculeIQ — Model Release Asset Packaging Utility (Phase 2E).

Bundles the verified trained model artifacts and fingerprint matrices into
a compressed release asset (moleculeiq-models-v5.0.0.tar.gz) suitable for
attachment to a GitHub Release.

Usage:
    python scripts/package_release_artifacts.py [--tag v5.0.0] [--output dist]
"""

import os
import sys
import json
import tarfile
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAG = "v5.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("package_release_artifacts")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def build_artifact_inventory(artifacts_dir: Path = REPO_ROOT / "artifacts") -> List[Dict[str, Any]]:
    """Build detailed metadata inventory for all 32 production artifacts."""
    checksums_file = artifacts_dir / "checksums.sha256"
    if not checksums_file.exists():
        logger.error(f"Checksums manifest not found: {checksums_file}")
        return []

    inventory = []
    with open(checksums_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            expected_hash, rel_path = parts[0], parts[1].strip()
            file_path = artifacts_dir / Path(rel_path)
            if not file_path.exists():
                logger.warning(f"Artifact in manifest does not exist: {file_path}")
                continue

            size_bytes = file_path.stat().st_size
            actual_hash = compute_sha256(file_path)

            # Association
            if "clintox" in rel_path:
                assoc = "clintox"
            elif "tox21" in rel_path:
                assoc = "tox21"
            elif "diagnostics" in rel_path:
                assoc = "tox21_diagnostics"
            else:
                assoc = "shared"

            # Runtime requirement
            # Models, calibration params, fingerprints, and UI diagnostics are runtime required
            runtime_req = not (rel_path.endswith("metrics.json") or rel_path.endswith("calibration_benchmark.json"))

            inventory.append({
                "filename": file_path.name,
                "relative_path": rel_path.replace("\\", "/"),
                "file_size_bytes": size_bytes,
                "file_size_mb": round(size_bytes / (1024 * 1024), 3),
                "sha256": actual_hash,
                "dataset_association": assoc,
                "required_at_runtime": runtime_req,
            })

    return inventory


def tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Filter out build caches, bytecode, and log files from the release archive."""
    parts = Path(tarinfo.name).parts
    if "__pycache__" in parts:
        return None
    name_lower = tarinfo.name.lower()
    if name_lower.endswith((".pyc", ".pyo", ".pyd", ".log", ".tmp", ".bak", ".swp")):
        return None
    basename = Path(tarinfo.name).name
    if basename.startswith(".") and basename not in [".gitkeep", ".gitattributes"]:
        return None
    return tarinfo


def package_models(tag: str = DEFAULT_TAG, output_dir: Path = REPO_ROOT / "dist") -> Path:
    """Bundle artifacts into a compressed release tarball and generate manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"moleculeiq-models-{tag}.tar.gz"
    bundle_path = output_dir / bundle_name

    logger.info(f"Creating release bundle: {bundle_path}")

    # Build detailed artifact inventory
    inventory = build_artifact_inventory(REPO_ROOT / "artifacts")
    manifest_data = {
        "release_tag": tag,
        "bundle_name": bundle_name,
        "total_artifacts": len(inventory),
        "total_size_bytes": sum(item["file_size_bytes"] for item in inventory),
        "total_size_mb": round(sum(item["file_size_bytes"] for item in inventory) / (1024 * 1024), 2),
        "artifacts": inventory,
    }

    manifest_path = output_dir / "release_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Generated release manifest with {len(inventory)} artifacts at {manifest_path}")

    files_to_pack = [
        REPO_ROOT / "artifacts" / "clintox",
        REPO_ROOT / "artifacts" / "tox21",
        REPO_ROOT / "artifacts" / "diagnostics",
        REPO_ROOT / "artifacts" / "checksums.sha256",
    ]

    with tarfile.open(bundle_path, "w:gz") as tar:
        for p in files_to_pack:
            if not p.exists():
                logger.warning(f"Path does not exist (skipping): {p}")
                continue
            arcname = p.relative_to(REPO_ROOT)
            logger.info(f"  Adding {arcname}...")
            tar.add(str(p), arcname=str(arcname), filter=tar_filter)
        # Add the release manifest itself
        tar.add(str(manifest_path), arcname="artifacts/release_manifest.json", filter=tar_filter)

    sha = compute_sha256(bundle_path)
    size_mb = bundle_path.stat().st_size / (1024 * 1024)

    logger.info(f"Successfully packaged '{bundle_name}' ({size_mb:.2f} MB).")
    logger.info(f"SHA-256: {sha}")

    sha_file = output_dir / f"{bundle_name}.sha256"
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write(f"{sha}  {bundle_name}\n")
    logger.info(f"Saved hash manifest to {sha_file}")

    return bundle_path


def main():
    parser = argparse.ArgumentParser(description="Package MoleculeIQ model artifacts for GitHub Release.")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG, help=f"Release tag (default: {DEFAULT_TAG}).")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "dist", help="Output directory (default: dist/).")
    args = parser.parse_args()

    package_models(tag=args.tag, output_dir=args.output)


if __name__ == "__main__":
    main()
