"""Centralized application path configuration and environment resolution."""

import os
from pathlib import Path

# Project root directory anchored dynamically to codebase root location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core data directories with environment variable override capabilities
DATA_DIR = Path(os.getenv("MOLECULEIQ_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = Path(os.getenv("MOLECULEIQ_RAW_DATA_DIR", DATA_DIR / "raw"))
REFERENCE_DATA_DIR = Path(os.getenv("MOLECULEIQ_REFERENCE_DATA_DIR", DATA_DIR / "reference"))

# Model artifact directories
ARTIFACTS_DIR = Path(os.getenv("MOLECULEIQ_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
CLINTOX_ARTIFACT_DIR = Path(os.getenv("MOLECULEIQ_CLINTOX_ARTIFACT_DIR", ARTIFACTS_DIR / "clintox"))
TOX21_ARTIFACT_DIR = Path(os.getenv("MOLECULEIQ_TOX21_ARTIFACT_DIR", ARTIFACTS_DIR / "tox21"))

# Standard file paths
REFERENCE_CACHE_PATH = Path(os.getenv("MOLECULEIQ_REFERENCE_CACHE_PATH", REFERENCE_DATA_DIR / "approved_drugs.csv"))
CLINTOX_DATA_PATH = Path(os.getenv("MOLECULEIQ_CLINTOX_DATA_PATH", RAW_DATA_DIR / "clintox.csv"))
TOX21_DATA_PATH = Path(os.getenv("MOLECULEIQ_TOX21_DATA_PATH", RAW_DATA_DIR / "tox21.csv"))


def resolve_path(path_str_or_obj: os.PathLike) -> str:
    """Resolve a path to an absolute path string anchored to PROJECT_ROOT if relative."""
    p = Path(path_str_or_obj)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)
