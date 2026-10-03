"""Explainability engine using SHAP TreeExplainer and substructure highlights."""

from src.explainability.shap_analysis import (
    ClinToxExplainer,
    ExplanationResult,
    FeatureAttribution,
    SCIENTIFIC_EXPLANATION_DISCLAIMER,
)

__all__ = [
    "ClinToxExplainer",
    "ExplanationResult",
    "FeatureAttribution",
    "SCIENTIFIC_EXPLANATION_DISCLAIMER",
]
