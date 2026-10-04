"""Unified model inference and caching service (Modules M04, M05)."""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.config import CLINTOX_ARTIFACT_DIR, TOX21_ARTIFACT_DIR, resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClinToxPrediction:
    """Inference result for ClinTox clinical trial toxicity prediction."""
    probability: float
    predicted_class: int
    is_toxic: bool
    model_version: str
    task_definition: str
    status: str
    applicability_domain_status: str = "WITHIN_DOMAIN"
    applicability_score: Optional[float] = None
    applicability_warning: Optional[str] = None
    error_message: Optional[str] = None
    uncalibrated_probability: Optional[float] = None


@dataclass(frozen=True)
class Tox21EndpointPrediction:
    """Inference result for a single Tox21 assay endpoint."""
    endpoint: str
    probability: float
    predicted_class: int
    is_active: bool
    status: str
    error_message: Optional[str] = None


@dataclass(frozen=True)
class Tox21MultiPrediction:
    """Aggregated prediction results across all supported Tox21 endpoints."""
    endpoints: Dict[str, Tox21EndpointPrediction]
    total_endpoints: int
    active_endpoints_count: int
    status: str


class ToxicityInferenceService:
    """
    Thread-safe, lazy-loading model inference manager for ClinTox and Tox21 classifiers.
    """

    def __init__(
        self,
        clintox_artifact_dir: Optional[str] = None,
        tox21_artifact_dir: Optional[str] = None,
    ):
        self.clintox_artifact_dir = clintox_artifact_dir if clintox_artifact_dir is not None else str(CLINTOX_ARTIFACT_DIR)
        self.tox21_artifact_dir = tox21_artifact_dir if tox21_artifact_dir is not None else str(TOX21_ARTIFACT_DIR)
        self._clintox_model: Optional[Any] = None
        self._uncalibrated_model: Optional[Any] = None
        self._training_fps: Optional[np.ndarray] = None
        self._clintox_meta: Optional[Dict[str, Any]] = None
        self._tox21_models: Dict[str, RandomForestClassifier] = {}
        self._tox21_metas: Dict[str, Dict[str, Any]] = {}

    def is_clintox_available(self) -> bool:
        """Check if ClinTox model artifact exists on disk."""
        model_path = os.path.join(self.clintox_artifact_dir, "model.joblib")
        return os.path.exists(model_path)

    def is_tox21_available(self) -> bool:
        """Check if any Tox21 model artifacts exist on disk."""
        if not os.path.exists(self.tox21_artifact_dir):
            return False
        return any(f.endswith("_model.joblib") for f in os.listdir(self.tox21_artifact_dir))

    def _load_clintox(self) -> Tuple[Any, Dict[str, Any]]:
        if self._clintox_model is None:
            model_path = os.path.join(self.clintox_artifact_dir, "model.joblib")
            meta_path = os.path.join(self.clintox_artifact_dir, "metadata.json")

            if not os.path.exists(model_path) or not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"ClinTox model artifacts missing in '{self.clintox_artifact_dir}'."
                )

            self._clintox_model = joblib.load(model_path)
            if hasattr(self._clintox_model, "calibrated_classifiers_"):
                for cc in self._clintox_model.calibrated_classifiers_:
                    if hasattr(cc, "estimator") and hasattr(cc.estimator, "n_jobs"):
                        cc.estimator.n_jobs = 1

            with open(meta_path, "r", encoding="utf-8") as f:
                self._clintox_meta = json.load(f)

            uncal_path = os.path.join(self.clintox_artifact_dir, "uncalibrated_model.joblib")
            if os.path.exists(uncal_path):
                try:
                    self._uncalibrated_model = joblib.load(uncal_path)
                    if hasattr(self._uncalibrated_model, "n_jobs"):
                        self._uncalibrated_model.n_jobs = 1
                except Exception as exc:
                    logger.warning(f"Failed loading uncalibrated ClinTox model: {exc}")

            fps_path = os.path.join(self.clintox_artifact_dir, "training_fingerprints.npz")
            if os.path.exists(fps_path):
                try:
                    self._training_fps = np.load(fps_path)["fingerprints"]
                    self._train_sums = np.sum(self._training_fps, axis=1)
                except Exception as exc:
                    logger.warning(f"Failed loading ClinTox training fingerprints: {exc}")

        return self._clintox_model, self._clintox_meta  # type: ignore

    def _load_tox21_endpoint(self, endpoint: str) -> Optional[Tuple[RandomForestClassifier, Dict[str, Any]]]:
        if endpoint in self._tox21_models:
            return self._tox21_models[endpoint], self._tox21_metas[endpoint]

        safe_name = endpoint.replace("-", "_")
        model_path = os.path.join(self.tox21_artifact_dir, f"{safe_name}_model.joblib")
        meta_path = os.path.join(self.tox21_artifact_dir, f"{safe_name}_metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return None

        try:
            model = joblib.load(model_path)
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._tox21_models[endpoint] = model
            self._tox21_metas[endpoint] = meta
            return model, meta
        except Exception as exc:
            logger.error(f"Failed to load Tox21 model for '{endpoint}': {exc}")
            return None

    def assess_applicability_domain(
        self, query_fp_vec: np.ndarray
    ) -> Tuple[str, Optional[float], Optional[str]]:
        """
        Assess chemical applicability domain via Tanimoto distance to training set library.

        Parameters
        ----------
        query_fp_vec : np.ndarray
            1D Morgan fingerprint array of query molecule (2048 bits).

        Returns
        -------
        status : str
            "WITHIN_DOMAIN", "NEAR_BOUNDARY", "OUTSIDE_DOMAIN", or "UNAVAILABLE".
        score : Optional[float]
            Maximum Tanimoto similarity to any training set compound.
        warning : Optional[str]
            Contextual advisory if query molecule is near or outside domain boundaries.
        """
        if self._training_fps is None:
            fps_path = os.path.join(self.clintox_artifact_dir, "training_fingerprints.npz")
            if os.path.exists(fps_path):
                try:
                    self._training_fps = np.load(fps_path)["fingerprints"]
                except Exception:
                    return "UNAVAILABLE", None, None
            else:
                return "UNAVAILABLE", None, None

        if getattr(self, "_train_sums", None) is None:
            self._train_sums = np.sum(self._training_fps, axis=1)

        train_sums = self._train_sums
        intersection = np.dot(self._training_fps, query_fp_vec)
        query_sum = np.sum(query_fp_vec)
        union = train_sums + query_sum - intersection
        sims = np.where(union > 0, intersection / union, 0.0)
        max_tanimoto = float(np.max(sims))

        if max_tanimoto >= 0.40:
            return "WITHIN_DOMAIN", round(max_tanimoto, 4), None
        elif max_tanimoto >= 0.25:
            warning = (
                f"Near applicability domain boundary (max Tanimoto to training set: {max_tanimoto:.2f}). "
                "Few close structural analogues in training set; prediction carries higher uncertainty."
            )
            return "NEAR_BOUNDARY", round(max_tanimoto, 4), warning
        else:
            warning = (
                f"Outside applicability domain (max Tanimoto to training set: {max_tanimoto:.2f} < 0.25). "
                "Molecule possesses a novel chemotype or alien scaffold unrepresented in training data; "
                "prediction is an extrapolation."
            )
            return "OUTSIDE_DOMAIN", round(max_tanimoto, 4), warning

    def predict_clintox(self, smiles_or_fp: Any) -> ClinToxPrediction:
        """
        Execute ClinTox clinical trial toxicity probability prediction.

        Parameters
        ----------
        smiles_or_fp : str or np.ndarray
            Input SMILES string or precomputed 2048-bit Morgan fingerprint array.
        """
        try:
            model, meta = self._load_clintox()
        except Exception as exc:
            return ClinToxPrediction(
                probability=0.0,
                predicted_class=0,
                is_toxic=False,
                model_version="unavailable",
                task_definition="Clinical trial failure due to toxicity (CT_TOX)",
                status="UNAVAILABLE",
                applicability_domain_status="UNAVAILABLE",
                error_message=str(exc),
            )

        if isinstance(smiles_or_fp, str):
            val_res = validate_smiles(smiles_or_fp)
            if not val_res.is_valid:
                return ClinToxPrediction(
                    probability=0.0,
                    predicted_class=0,
                    is_toxic=False,
                    model_version=meta.get("model_id", "v1"),
                    task_definition="Clinical trial failure due to toxicity (CT_TOX)",
                    status="INVALID_INPUT",
                    applicability_domain_status="UNAVAILABLE",
                    error_message=val_res.error_message,
                )
            fp_res = generate_morgan_fingerprint(val_res.canonical_smiles, radius=2, n_bits=2048)
            X = fp_res.numpy_vector.reshape(1, -1)
        elif isinstance(smiles_or_fp, np.ndarray):
            X = smiles_or_fp.reshape(1, -1)
        else:
            return ClinToxPrediction(
                probability=0.0,
                predicted_class=0,
                is_toxic=False,
                model_version=meta.get("model_id", "v1"),
                task_definition="Clinical trial failure due to toxicity (CT_TOX)",
                status="INVALID_INPUT",
                applicability_domain_status="UNAVAILABLE",
                error_message="Input must be SMILES or numpy array",
            )

        # Dynamic positive class probability index lookup
        pos_idx = meta.get("positive_class_index", 1)
        probs = model.predict_proba(X)[0]
        prob_toxic = float(probs[pos_idx])
        pred_class = int(prob_toxic >= 0.5)

        # Uncalibrated probability lookup if base tree model exists
        uncal_prob = None
        if self._uncalibrated_model is not None:
            try:
                uncal_probs = self._uncalibrated_model.predict_proba(X)[0]
                uncal_prob = round(float(uncal_probs[pos_idx]), 4)
            except Exception:
                uncal_prob = None

        # Dynamic applicability domain evaluation
        ad_status, ad_score, ad_warning = self.assess_applicability_domain(X[0])

        return ClinToxPrediction(
            probability=round(prob_toxic, 4),
            predicted_class=pred_class,
            is_toxic=(pred_class == 1),
            model_version=meta.get("model_id", "clintox_rf_calibrated"),
            task_definition="Clinical trial failure due to toxicity (CT_TOX)",
            status="SUCCESS",
            applicability_domain_status=ad_status,
            applicability_score=ad_score,
            applicability_warning=ad_warning,
            uncalibrated_probability=uncal_prob,
        )

    def predict_tox21(self, smiles_or_fp: Any) -> Tox21MultiPrediction:
        """
        Execute multi-endpoint prediction across all 12 Tox21 assays.
        """
        from src.models.tox21 import TOX21_ENDPOINTS

        if isinstance(smiles_or_fp, str):
            val_res = validate_smiles(smiles_or_fp)
            if not val_res.is_valid:
                return Tox21MultiPrediction(
                    endpoints={},
                    total_endpoints=0,
                    active_endpoints_count=0,
                    status="INVALID_INPUT",
                )
            fp_res = generate_morgan_fingerprint(val_res.canonical_smiles, radius=2, n_bits=2048)
            X = fp_res.numpy_vector.reshape(1, -1)
        elif isinstance(smiles_or_fp, np.ndarray):
            X = smiles_or_fp.reshape(1, -1)
        else:
            return Tox21MultiPrediction(
                endpoints={},
                total_endpoints=0,
                active_endpoints_count=0,
                status="INVALID_INPUT",
            )

        endpoint_results: Dict[str, Tox21EndpointPrediction] = {}
        active_count = 0

        for ep in TOX21_ENDPOINTS:
            loaded = self._load_tox21_endpoint(ep)
            if loaded is None:
                endpoint_results[ep] = Tox21EndpointPrediction(
                    endpoint=ep,
                    probability=0.0,
                    predicted_class=0,
                    is_active=False,
                    status="UNAVAILABLE",
                    error_message="Model artifact not found",
                )
                continue

            model, meta = loaded
            pos_idx = meta.get("positive_class_index", 1)
            probs = model.predict_proba(X)[0]
            prob_active = float(probs[pos_idx])
            pred_class = int(prob_active >= 0.5)
            is_active = (pred_class == 1)

            if is_active:
                active_count += 1

            endpoint_results[ep] = Tox21EndpointPrediction(
                endpoint=ep,
                probability=round(prob_active, 4),
                predicted_class=pred_class,
                is_active=is_active,
                status="SUCCESS",
            )

        return Tox21MultiPrediction(
            endpoints=endpoint_results,
            total_endpoints=len(endpoint_results),
            active_endpoints_count=active_count,
            status="SUCCESS",
        )
