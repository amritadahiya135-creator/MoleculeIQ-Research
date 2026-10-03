"""Integration tests for explainability engine and SHAP substructure mapping (Module M08)."""

import pytest
from src.explainability.shap_analysis import (
    ClinToxExplainer,
    ExplanationResult,
    FeatureAttribution,
    SCIENTIFIC_EXPLANATION_DISCLAIMER,
)


class TestExplainabilityEngine:
    """Tests for Module M08: SHAP Explainability Engine."""

    @pytest.fixture
    def explainer(self):
        return ClinToxExplainer()

    def test_explain_valid_molecule_aspirin(self, explainer):
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        result = explainer.explain_molecule(aspirin, top_k_bits=5)

        assert isinstance(result, ExplanationResult)
        assert result.status == "SUCCESS"
        assert result.explained_class == "CT_TOX (Clinical Trial Toxicity Failure)"
        assert len(result.top_features) > 0
        assert len(result.top_features) <= 5

        # Check structure of top feature
        feat = result.top_features[0]
        assert isinstance(feat, FeatureAttribution)
        assert feat.is_active is True
        assert feat.direction in ["increases_toxicity_risk", "decreases_toxicity_risk"]
        assert isinstance(feat.bit_id, int)
        assert isinstance(feat.shap_value, float)

        # Check SVG was generated
        assert result.highlighted_svg is not None
        assert "<svg" in result.highlighted_svg

        # Check disclaimer presence
        assert result.disclaimer == SCIENTIFIC_EXPLANATION_DISCLAIMER

    def test_explain_invalid_molecule_fails_gracefully(self, explainer):
        result = explainer.explain_molecule("INVALID_CHEMICAL_STRING")
        assert result.status == "INVALID_INPUT"
        assert result.error_message is not None
        assert result.highlighted_svg is None
        assert len(result.top_features) == 0
