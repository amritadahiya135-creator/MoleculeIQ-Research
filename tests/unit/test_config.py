import os
from pathlib import Path
from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    ARTIFACTS_DIR,
    CLINTOX_ARTIFACT_DIR,
    TOX21_ARTIFACT_DIR,
    REFERENCE_CACHE_PATH,
    resolve_path,
)
from src.models.inference import ToxicityInferenceService
from src.explainability.shap_analysis import ClinToxExplainer


class TestConfigAndPathResolution:
    """Verifies that centralized path resolution works across modules."""

    def test_project_root_anchored_correctly(self):
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "src").exists()
        assert (PROJECT_ROOT / "data").exists()

    def test_resolve_path_absolute_and_relative(self):
        # Relative path gets anchored to PROJECT_ROOT
        rel = resolve_path("artifacts/clintox")
        assert Path(rel).is_absolute()
        assert rel.endswith(os.path.join("artifacts", "clintox"))

        # Already absolute path remains unchanged
        abs_p = str(PROJECT_ROOT / "data" / "raw")
        assert resolve_path(abs_p) == abs_p

    def test_services_default_to_absolute_paths(self):
        inf_service = ToxicityInferenceService()
        assert Path(inf_service.clintox_artifact_dir).is_absolute()
        assert Path(inf_service.tox21_artifact_dir).is_absolute()

        explainer = ClinToxExplainer()
        assert Path(explainer.model_path).is_absolute()
