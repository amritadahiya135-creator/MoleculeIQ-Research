"""End-to-end integration test for Workflow A: Single Molecule Analysis."""

import pytest
from src.services.orchestration import ScreeningOrchestrator
from src.services.contracts import MoleculeAnalysisResult


class TestSingleMoleculeWorkflow:
    """Verifies complete Workflow A execution."""

    @pytest.fixture
    def orchestrator(self):
        return ScreeningOrchestrator()

    def test_complete_aspirin_pipeline(self, orchestrator):
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        result = orchestrator.analyze_single_molecule(aspirin, compute_explanation=True)

        assert isinstance(result, MoleculeAnalysisResult)
        assert result.is_valid is True
        assert result.status == "SUCCESS"

        # Validation
        assert result.validation.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert result.validation.formula == "C9H8O4"

        # Descriptors
        assert result.descriptors is not None
        assert pytest.approx(result.descriptors.molecular_weight, 0.1) == 180.16
        assert result.descriptors.lipinski_pass is True

        # ClinTox prediction
        assert result.clintox is not None
        assert result.clintox.status == "SUCCESS"
        assert 0.0 <= result.clintox.probability <= 1.0

        # Tox21 multi-prediction
        assert result.tox21 is not None
        assert result.tox21.status == "SUCCESS"
        assert result.tox21.total_endpoints == 12

        # Similarity search
        assert result.similarity is not None
        assert result.similarity.status == "SUCCESS"
        assert result.similarity.best_similarity == 1.0  # Aspirin is in reference set

        # Prioritization score
        assert result.prioritization is not None
        assert result.prioritization.is_complete is True
        assert 0.0 <= result.prioritization.overall_score <= 100.0

        # SHAP explainability
        assert result.explanation is not None
        assert result.explanation.status == "SUCCESS"
        assert result.explanation.highlighted_svg is not None
        assert "<svg" in result.explanation.highlighted_svg
