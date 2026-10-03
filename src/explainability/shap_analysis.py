"""Explainable AI engine via SHAP TreeExplainer and Morgan substructure mapping (Module M08)."""

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import joblib
import shap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint, FingerprintResult
from src.config import CLINTOX_ARTIFACT_DIR, resolve_path

logger = logging.getLogger(__name__)

SCIENTIFIC_EXPLANATION_DISCLAIMER = (
    "SCIENTIFIC NOTICE: SHAP values quantify the statistical contribution of individual "
    "fingerprint bits to the Random Forest model's predictions. These values describe "
    "mathematical model behavior and do NOT establish biological causality, toxicophore "
    "mechanisms, or verified cellular pathways."
)


@dataclass(frozen=True)
class FeatureAttribution:
    """Individual fingerprint bit attribution."""
    bit_id: int
    shap_value: float
    is_active: bool
    direction: str  # "increases_toxicity_risk" or "decreases_toxicity_risk"
    atom_indices: List[int]
    radius: int


@dataclass(frozen=True)
class ExplanationResult:
    """Complete explainability outcome for a query molecule."""
    query_smiles: str
    explained_class: str
    base_value: float
    top_features: List[FeatureAttribution]
    highlighted_svg: Optional[str]
    status: str
    disclaimer: str
    error_message: Optional[str] = None


class ClinToxExplainer:
    """
    SHAP TreeExplainer wrapper for the ClinTox Random Forest classifier.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path if model_path is not None else str(CLINTOX_ARTIFACT_DIR / "model.joblib")
        self._model = None
        self._explainer = None

    def _ensure_loaded(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ClinTox model artifact not found at '{self.model_path}'.")
            self._model = joblib.load(self.model_path)
            self._explainer = shap.TreeExplainer(self._model)

    def explain_molecule(
        self,
        smiles: str,
        top_k_bits: int = 5,
        width: int = 350,
        height: int = 300,
    ) -> ExplanationResult:
        """
        Compute local SHAP feature attributions and map active bits to molecular substructures.

        Parameters
        ----------
        smiles : str
            Query chemical structure.
        top_k_bits : int, default=5
            Number of top active fingerprint bits to report.
        width, height : int
            SVG rendering dimensions.

        Returns
        -------
        ExplanationResult
            Contains top contributing bits, atom indices, SVG visualization, and disclaimers.
        """
        try:
            self._ensure_loaded()
        except Exception as exc:
            return ExplanationResult(
                query_smiles=smiles,
                explained_class="CT_TOX (Positive Class / Toxic)",
                base_value=0.0,
                top_features=[],
                highlighted_svg=None,
                status="MODEL_UNAVAILABLE",
                disclaimer=SCIENTIFIC_EXPLANATION_DISCLAIMER,
                error_message=str(exc),
            )

        val_res = validate_smiles(smiles)
        if not val_res.is_valid or not val_res.canonical_smiles:
            return ExplanationResult(
                query_smiles=smiles,
                explained_class="CT_TOX (Positive Class / Toxic)",
                base_value=0.0,
                top_features=[],
                highlighted_svg=None,
                status="INVALID_INPUT",
                disclaimer=SCIENTIFIC_EXPLANATION_DISCLAIMER,
                error_message=val_res.error_message,
            )

        # Generate fingerprint with atom environment metadata
        try:
            mol = Chem.MolFromSmiles(val_res.canonical_smiles)
            fp_res = generate_morgan_fingerprint(
                mol,
                radius=2,
                n_bits=2048,
                return_bit_info=True,
            )
            X = fp_res.numpy_vector.reshape(1, -1)
        except Exception as exc:
            return ExplanationResult(
                query_smiles=smiles,
                explained_class="CT_TOX (Positive Class / Toxic)",
                base_value=0.0,
                top_features=[],
                highlighted_svg=None,
                status="FEATURIZATION_ERROR",
                disclaimer=SCIENTIFIC_EXPLANATION_DISCLAIMER,
                error_message=str(exc),
            )

        try:
            # TreeExplainer on binary RF returns array of shape (1, 2048, 2)
            shap_values = self._explainer.shap_values(X, check_additivity=False)
            
            # Extract base value for positive class (class 1)
            expected_vals = self._explainer.expected_value
            if isinstance(expected_vals, (list, np.ndarray)):
                base_val = float(expected_vals[1]) if len(expected_vals) > 1 else float(expected_vals[0])
            else:
                base_val = float(expected_vals)

            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # Shape: (samples, features, classes) -> positive class is index 1
                toxic_shap = shap_values[0, :, 1]
            elif isinstance(shap_values, list) and len(shap_values) == 2:
                # List of [class_0, class_1]
                toxic_shap = shap_values[1][0]
            else:
                toxic_shap = shap_values[0]

            # Find active bits in this molecule
            active_bit_indices = np.where(fp_res.numpy_vector > 0)[0]

            # Rank active bits by absolute SHAP attribution
            active_shap = [(int(idx), float(toxic_shap[idx])) for idx in active_bit_indices]
            active_shap.sort(key=lambda item: abs(item[1]), reverse=True)

            top_features: List[FeatureAttribution] = []
            highlight_atoms: Set[int] = set()

            for bit_id, sv in active_shap[:top_k_bits]:
                envs = fp_res.bit_info.get(bit_id, [])
                atom_indices = []
                max_radius = 0
                for atom_idx, radius in envs:
                    atom_indices.append(int(atom_idx))
                    if radius > max_radius:
                        max_radius = radius
                    highlight_atoms.add(int(atom_idx))

                direction = "increases_toxicity_risk" if sv > 0 else "decreases_toxicity_risk"
                top_features.append(
                    FeatureAttribution(
                        bit_id=bit_id,
                        shap_value=round(sv, 5),
                        is_active=True,
                        direction=direction,
                        atom_indices=atom_indices,
                        radius=max_radius,
                    )
                )

            # Generate SVG visualization highlighting the contributing atoms
            svg_text: Optional[str] = None
            try:
                d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
                d2d.drawOptions().padding = 0.1
                d2d.DrawMolecule(mol, highlightAtoms=list(highlight_atoms))
                d2d.FinishDrawing()
                svg_text = d2d.GetDrawingText()
            except Exception as draw_exc:
                logger.warning(f"Could not render highlight SVG: {draw_exc}")

            return ExplanationResult(
                query_smiles=val_res.canonical_smiles,
                explained_class="CT_TOX (Clinical Trial Toxicity Failure)",
                base_value=round(base_val, 4),
                top_features=top_features,
                highlighted_svg=svg_text,
                status="SUCCESS",
                disclaimer=SCIENTIFIC_EXPLANATION_DISCLAIMER,
                error_message=None,
            )

        except Exception as exc:
            logger.error(f"SHAP explanation failed for '{smiles}': {exc}")
            return ExplanationResult(
                query_smiles=smiles,
                explained_class="CT_TOX (Positive Class / Toxic)",
                base_value=0.0,
                top_features=[],
                highlighted_svg=None,
                status="EXPLANATION_FAILED",
                disclaimer=SCIENTIFIC_EXPLANATION_DISCLAIMER,
                error_message=str(exc),
            )
