"""Tox21 Assay-Specific Applicability Domain Evaluation (Phase A.2).

Provides research-grade chemical applicability domain assessment for each of the
12 Tox21 assay endpoints, strictly utilizing endpoint-specific training partitions
derived with zero test-set leakage and isolated label-missingness handling.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.models.tox21 import TOX21_ENDPOINTS
from src.config import TOX21_DATA_PATH, TOX21_ARTIFACT_DIR, resolve_path

logger = logging.getLogger(__name__)

# Constants and Defaults
TOX21_AD_DEFAULT_RADIUS = 2
TOX21_AD_DEFAULT_NBITS = 2048

# Domain Status Codes
STATUS_UNVALIDATED = "UNVALIDATED"
STATUS_UNAVAILABLE = "UNAVAILABLE"

# Scientific Disclaimer
TOX21_AD_DISCLAIMER = (
    "Tox21 assay-specific applicability thresholds are unvalidated; "
    "structural similarity does not guarantee in-vitro assay inactivity or biological safety."
)


@dataclass(frozen=True)
class Tox21EndpointADResult:
    """Applicability domain assessment result for a single Tox21 assay endpoint."""
    endpoint: str
    applicability_score: Optional[float]
    applicability_domain_status: str
    applicability_warning: Optional[str]
    training_sample_count: int


def build_tox21_reference_corpora(
    csv_path: Optional[Union[str, Path]] = None,
    output_npz_path: Optional[Union[str, Path]] = None,
    endpoints: Optional[List[str]] = None,
    split_ratio: float = 0.2,
    random_seed: int = 42,
    radius: int = TOX21_AD_DEFAULT_RADIUS,
    n_bits: int = TOX21_AD_DEFAULT_NBITS,
) -> Dict[str, Any]:
    """
    Deterministically construct endpoint-specific training reference corpora from raw Tox21 data.

    Enforces zero leakage by strictly excluding test partition compounds and
    compounds with missing (NaN) labels for each respective endpoint.

    Parameters
    ----------
    csv_path : Optional[Union[str, Path]]
        Path to raw tox21.csv file. Defaults to TOX21_DATA_PATH.
    output_npz_path : Optional[Union[str, Path]]
        Path to write compressed npz archive. Defaults to TOX21_ARTIFACT_DIR / 'training_fingerprints.npz'.
    endpoints : Optional[List[str]]
        Endpoints to process. Defaults to TOX21_ENDPOINTS.
    split_ratio : float
        Test split ratio matching model training (0.2).
    random_seed : int
        Random seed matching model training (42).
    radius : int
        Morgan fingerprint radius (2).
    n_bits : int
        Morgan fingerprint bit length (2048).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing arrays for fingerprints and SMILES per endpoint.
    """
    if endpoints is None:
        endpoints = TOX21_ENDPOINTS

    resolved_csv = resolve_path(csv_path or TOX21_DATA_PATH)
    if not os.path.exists(resolved_csv):
        raise FileNotFoundError(f"Tox21 raw dataset not found at '{resolved_csv}'.")

    logger.info(f"Building Tox21 reference corpora from '{resolved_csv}'...")
    df = pd.read_csv(resolved_csv)

    # Step 1: Pre-validate and generate Morgan fingerprints for all valid SMILES
    valid_mask = []
    fps_list = []
    canonical_smiles_list = []

    for smi in df["smiles"]:
        val_res = validate_smiles(str(smi))
        if not val_res.is_valid or not val_res.canonical_smiles:
            valid_mask.append(False)
            fps_list.append(None)
            canonical_smiles_list.append(None)
        else:
            valid_mask.append(True)
            fp_res = generate_morgan_fingerprint(
                val_res.canonical_smiles,
                radius=radius,
                n_bits=n_bits,
                return_bit_info=False,
            )
            fps_list.append(fp_res.numpy_vector)
            canonical_smiles_list.append(val_res.canonical_smiles)

    valid_mask = np.array(valid_mask, dtype=bool)

    archive_dict: Dict[str, Any] = {}

    for ep in endpoints:
        if ep not in df.columns:
            logger.warning(f"Endpoint '{ep}' missing from columns. Skipping.")
            continue

        ep_series = df[ep]
        not_na_mask = ep_series.notna().values & valid_mask
        n_samples = int(np.sum(not_na_mask))

        if n_samples < 50:
            logger.warning(f"Insufficient samples ({n_samples}) for endpoint '{ep}'. Skipping.")
            continue

        ep_indices = np.where(not_na_mask)[0]
        X_ep = np.array([fps_list[i] for i in ep_indices], dtype=np.int8)
        y_ep = ep_series.values[not_na_mask].astype(int)

        # Replicate exact stratified train_test_split used during model training
        idx_all = np.arange(len(ep_indices))
        train_idx, _ = train_test_split(
            idx_all,
            test_size=split_ratio,
            random_state=random_seed,
            stratify=y_ep,
        )

        X_train = X_ep[train_idx]
        train_orig_indices = ep_indices[train_idx]
        smiles_train = np.array([canonical_smiles_list[i] for i in train_orig_indices], dtype=object)

        safe_name = ep.replace("-", "_")
        archive_dict[f"{safe_name}_fingerprints"] = X_train
        archive_dict[f"{safe_name}_smiles"] = smiles_train

    if output_npz_path is not None or TOX21_ARTIFACT_DIR is not None:
        target_path = resolve_path(output_npz_path or os.path.join(TOX21_ARTIFACT_DIR, "training_fingerprints.npz"))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        np.savez_compressed(target_path, **archive_dict)
        logger.info(f"Persisted Tox21 reference corpora ({len(archive_dict)} arrays) to '{target_path}'.")

    return archive_dict


class Tox21ApplicabilityEvaluator:
    """
    Thread-safe, lazy-loading evaluator for Tox21 12-endpoint applicability domain.

    Computes maximum Tanimoto similarity against endpoint-specific training corpora.
    Categorical thresholds are intentionally kept as UNVALIDATED to prevent false
    safety assurances.
    """

    def __init__(
        self,
        artifact_dir: Optional[Union[str, Path]] = None,
        raw_data_path: Optional[Union[str, Path]] = None,
    ):
        self.artifact_dir = resolve_path(artifact_dir or TOX21_ARTIFACT_DIR)
        self.raw_data_path = resolve_path(raw_data_path or TOX21_DATA_PATH)
        self._fps_by_endpoint: Optional[Dict[str, np.ndarray]] = None
        self._sums_by_endpoint: Optional[Dict[str, np.ndarray]] = None
        self._smiles_by_endpoint: Optional[Dict[str, np.ndarray]] = None

    def _ensure_corpora_loaded(self) -> bool:
        """Lazy-load reference corpora from disk npz or build deterministically if absent."""
        if self._fps_by_endpoint is not None:
            return True

        npz_path = os.path.join(self.artifact_dir, "training_fingerprints.npz")
        data = None

        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                logger.debug(f"Loaded Tox21 reference corpora from '{npz_path}'.")
            except Exception as exc:
                logger.warning(f"Failed loading Tox21 reference npz at '{npz_path}': {exc}")
                data = None

        if data is None and os.path.exists(self.raw_data_path):
            try:
                logger.info("Generating Tox21 reference corpora dynamically from raw data...")
                data = build_tox21_reference_corpora(
                    csv_path=self.raw_data_path,
                    output_npz_path=npz_path,
                )
            except Exception as exc:
                logger.error(f"Failed building Tox21 reference corpora: {exc}")
                return False

        if data is None:
            return False

        fps_dict: Dict[str, np.ndarray] = {}
        sums_dict: Dict[str, np.ndarray] = {}
        smiles_dict: Dict[str, np.ndarray] = {}

        for ep in TOX21_ENDPOINTS:
            safe = ep.replace("-", "_")
            fp_key = f"{safe}_fingerprints"
            smi_key = f"{safe}_smiles"

            if fp_key in data:
                fps = data[fp_key]
                fps_dict[ep] = fps
                sums_dict[ep] = np.sum(fps, axis=1)
                if smi_key in data:
                    smiles_dict[ep] = data[smi_key]

        self._fps_by_endpoint = fps_dict
        self._sums_by_endpoint = sums_dict
        self._smiles_by_endpoint = smiles_dict
        return len(fps_dict) > 0

    def is_available(self, endpoint: Optional[str] = None) -> bool:
        """Check if reference corpus is available for an endpoint or all endpoints."""
        if not self._ensure_corpora_loaded():
            return False
        if endpoint is not None:
            return endpoint in (self._fps_by_endpoint or {})
        return bool(self._fps_by_endpoint and len(self._fps_by_endpoint) > 0)

    def get_training_count(self, endpoint: str) -> int:
        """Return the number of training compounds in the reference corpus for endpoint."""
        if not self._ensure_corpora_loaded() or not self._fps_by_endpoint:
            return 0
        arr = self._fps_by_endpoint.get(endpoint)
        return int(arr.shape[0]) if arr is not None else 0

    def get_training_smiles(self, endpoint: str) -> List[str]:
        """Return the list of canonical training SMILES for an endpoint."""
        if not self._ensure_corpora_loaded() or not self._smiles_by_endpoint:
            return []
        arr = self._smiles_by_endpoint.get(endpoint)
        return list(arr) if arr is not None else []

    def evaluate_endpoint(
        self,
        endpoint: str,
        query_fp_vec: np.ndarray,
    ) -> Tox21EndpointADResult:
        """
        Evaluate applicability domain for a single endpoint given a 2048-bit Morgan fingerprint.

        Parameters
        ----------
        endpoint : str
            Tox21 endpoint name (e.g. 'NR-AR').
        query_fp_vec : np.ndarray
            1D array of length 2048 (bit or count vector).

        Returns
        -------
        Tox21EndpointADResult
            Structured result containing maximum Tanimoto score, unvalidated status, and warning.
        """
        if not self._ensure_corpora_loaded() or not self._fps_by_endpoint or endpoint not in self._fps_by_endpoint:
            return Tox21EndpointADResult(
                endpoint=endpoint,
                applicability_score=None,
                applicability_domain_status=STATUS_UNAVAILABLE,
                applicability_warning=f"Reference training corpus unavailable for endpoint '{endpoint}'.",
                training_sample_count=0,
            )

        ref_fps = self._fps_by_endpoint[endpoint]
        if ref_fps.shape[0] == 0:
            return Tox21EndpointADResult(
                endpoint=endpoint,
                applicability_score=None,
                applicability_domain_status=STATUS_UNAVAILABLE,
                applicability_warning=f"Empty reference corpus for endpoint '{endpoint}'.",
                training_sample_count=0,
            )

        ref_sums = self._sums_by_endpoint[endpoint]  # type: ignore
        q_sum = float(np.sum(query_fp_vec))

        if q_sum == 0:
            # Query has no bits set
            return Tox21EndpointADResult(
                endpoint=endpoint,
                applicability_score=0.0,
                applicability_domain_status=STATUS_UNVALIDATED,
                applicability_warning=TOX21_AD_DISCLAIMER,
                training_sample_count=int(ref_fps.shape[0]),
            )

        intersection = np.dot(ref_fps, query_fp_vec)
        union = ref_sums + q_sum - intersection
        sims = np.where(union > 0, intersection / union, 0.0)
        max_tanimoto = float(np.max(sims))

        return Tox21EndpointADResult(
            endpoint=endpoint,
            applicability_score=round(max_tanimoto, 4),
            applicability_domain_status=STATUS_UNVALIDATED,
            applicability_warning=TOX21_AD_DISCLAIMER,
            training_sample_count=int(ref_fps.shape[0]),
        )

    def evaluate_all(self, smiles_or_fp: Any) -> Dict[str, Tox21EndpointADResult]:
        """
        Evaluate applicability domain across all 12 Tox21 endpoints.

        Parameters
        ----------
        smiles_or_fp : str or np.ndarray
            Input SMILES string or 2048-bit Morgan fingerprint vector.

        Returns
        -------
        Dict[str, Tox21EndpointADResult]
            Endpoint-keyed applicability domain results.
        """
        if isinstance(smiles_or_fp, str):
            val_res = validate_smiles(smiles_or_fp)
            if not val_res.is_valid or not val_res.canonical_smiles:
                return {
                    ep: Tox21EndpointADResult(
                        endpoint=ep,
                        applicability_score=None,
                        applicability_domain_status=STATUS_UNAVAILABLE,
                        applicability_warning=f"Invalid molecular structure: {val_res.error_message or 'validation failed'}",
                        training_sample_count=self.get_training_count(ep),
                    )
                    for ep in TOX21_ENDPOINTS
                }
            try:
                fp_res = generate_morgan_fingerprint(
                    val_res.canonical_smiles,
                    radius=TOX21_AD_DEFAULT_RADIUS,
                    n_bits=TOX21_AD_DEFAULT_NBITS,
                )
                query_vec = fp_res.numpy_vector
            except Exception as exc:
                return {
                    ep: Tox21EndpointADResult(
                        endpoint=ep,
                        applicability_score=None,
                        applicability_domain_status=STATUS_UNAVAILABLE,
                        applicability_warning=f"Fingerprint generation failed: {exc}",
                        training_sample_count=self.get_training_count(ep),
                    )
                    for ep in TOX21_ENDPOINTS
                }
        elif isinstance(smiles_or_fp, np.ndarray):
            query_vec = smiles_or_fp.ravel()
            if len(query_vec) != TOX21_AD_DEFAULT_NBITS:
                return {
                    ep: Tox21EndpointADResult(
                        endpoint=ep,
                        applicability_score=None,
                        applicability_domain_status=STATUS_UNAVAILABLE,
                        applicability_warning=f"Fingerprint vector length ({len(query_vec)}) must match {TOX21_AD_DEFAULT_NBITS}.",
                        training_sample_count=self.get_training_count(ep),
                    )
                    for ep in TOX21_ENDPOINTS
                }
        else:
            return {
                ep: Tox21EndpointADResult(
                    endpoint=ep,
                    applicability_score=None,
                    applicability_domain_status=STATUS_UNAVAILABLE,
                    applicability_warning="Input must be SMILES string or numpy array.",
                    training_sample_count=self.get_training_count(ep),
                )
                for ep in TOX21_ENDPOINTS
            }

        return {
            ep: self.evaluate_endpoint(ep, query_vec)
            for ep in TOX21_ENDPOINTS
        }
