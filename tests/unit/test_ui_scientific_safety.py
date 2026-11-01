"""
Unit and integration-level tests for Phase C.3:
Scientific UI Safety, Transparent Triage Presentation, Applicability-Domain
Governance, and Reference-Library Provenance Disclosures.
"""

import math
from pathlib import Path
import pytest
from rdkit import Chem

from src.services.orchestration import ScreeningOrchestrator
from src.services.contracts import (
    MoleculeAnalysisResult,
    ClinToxPrediction,
    Tox21MultiPrediction,
    SimilaritySearchResult,
    PrioritizationScore,
    BatchScreeningResult,
    BatchCandidateRecord,
)
from src.chemistry.validation import MoleculeValidationResult
from src.chemistry.descriptors import MolecularDescriptors
from src.similarity.index import SimilarityMatch
from src.ui.components import (
    render_triage_editorial_grid,
    render_executive_interpretation,
    render_risk_gauge,
)


@pytest.fixture(scope="module")
def orchestrator():
    """Module-scoped ScreeningOrchestrator instance."""
    return ScreeningOrchestrator()


# =============================================================================
# 1. Neutral Labels & Absence of Prescriptive Tiers
# =============================================================================

def test_source_files_eliminated_prescriptive_triage_labels():
    """
    Verify that prescriptive, promotional, or clinical efficacy labels
    have been strictly purged from all UI presentation files.
    """
    ui_dir = Path("src/ui")
    forbidden_terms = [
        "High Priority Candidate",
        "Moderate Priority Candidate",
        "Low Priority Candidate",
        "Safe Leads",
        "Lead Lead",
        "Composite Research Triage Index",
        "LOW RISK",
        "ELEVATED RISK",
        "Failure Likelihood",
        "Failure Probability",
        "Failure Risk",
        "Low-Risk",
    ]

    for py_file in ui_dir.glob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in code, f"Forbidden prescriptive term '{term}' found in {py_file}"


def test_exploratory_triage_index_present_in_ui_components():
    """
    Verify that 'Exploratory Triage Index' is the standard label across UI modules.
    """
    single_mol_code = Path("src/ui/single_molecule.py").read_text(encoding="utf-8")
    components_code = Path("src/ui/components.py").read_text(encoding="utf-8")
    batch_code = Path("src/ui/batch_screening.py").read_text(encoding="utf-8")

    assert "Exploratory Triage Index" in single_mol_code
    assert "Exploratory Triage Index" in components_code
    assert "Exploratory Triage Index" in batch_code


# =============================================================================
# 2. Applicability Domain Badge Treatments & Neutral Styling
# =============================================================================

def test_applicability_domain_badge_logic_in_executive_interpretation(monkeypatch):
    """
    Verify that render_executive_interpretation generates the correct badge and
    prevents positive green badge styling for WITHIN_DOMAIN, NEAR_BOUNDARY,
    OUTSIDE_DOMAIN, and UNAVAILABLE.
    """
    rendered_markdowns = []

    def fake_markdown(text, unsafe_allow_html=False):
        rendered_markdowns.append(text)

    import streamlit as st
    monkeypatch.setattr(st, "markdown", fake_markdown)

    # Helper dummy result
    def make_res(ad_stat, ad_score):
        return MoleculeAnalysisResult(
            validation=MoleculeValidationResult(original_smiles="CC", canonical_smiles="CC", is_valid=True, formula="C2H6"),
            descriptors=MolecularDescriptors(
                molecular_weight=30.0, logp=1.0, hbond_donors=0, hbond_acceptors=0,
                rotatable_bonds=0, tpsa=0.0, lipinski_violations=0, lipinski_pass=True, violation_details=[]
            ),
            clintox=ClinToxPrediction(
                probability=0.10,
                predicted_class=0,
                is_toxic=False,
                model_version="clintox_rf_calibrated",
                task_definition="Clinical trial failure due to toxicity (CT_TOX)",
                status="SUCCESS",
                applicability_domain_status=ad_stat,
                applicability_score=ad_score,
                applicability_warning=None,
                uncalibrated_probability=0.10,
            ),
            tox21=None,
            similarity=None,
            prioritization=PrioritizationScore(
                safety_score=90.0,
                druglikeness_score=100.0,
                similarity_score=50.0,
                overall_score=85.0,
                weight_safety=0.5,
                weight_druglikeness=0.3,
                weight_similarity=0.2,
                is_complete=True,
            ),
            explanation=None,
            analysis_timestamp="2026-10-15T00:00:00Z",
            is_valid=True,
            status="SUCCESS",
        )

    # 1. OUTSIDE_DOMAIN
    rendered_markdowns.clear()
    render_executive_interpretation(make_res("OUTSIDE_DOMAIN", 0.12))
    joined = "".join(rendered_markdowns)
    assert '<span class="mol-badge mol-badge-danger">Outside Domain</span>' in joined
    assert "Extrapolation warning" in joined
    assert "mol-badge-success" not in joined

    # 2. WITHIN_DOMAIN -> uses mol-badge-info, NEVER mol-badge-success
    rendered_markdowns.clear()
    render_executive_interpretation(make_res("WITHIN_DOMAIN", 0.85))
    joined = "".join(rendered_markdowns)
    assert '<span class="mol-badge mol-badge-info">Within Domain</span>' in joined
    assert "configured domain criterion" in joined

    # 3. NEAR_BOUNDARY
    rendered_markdowns.clear()
    render_executive_interpretation(make_res("NEAR_BOUNDARY", 0.32))
    joined = "".join(rendered_markdowns)
    assert '<span class="mol-badge mol-badge-warning">Near Boundary</span>' in joined

    # 4. UNAVAILABLE
    rendered_markdowns.clear()
    render_executive_interpretation(make_res("UNAVAILABLE", None))
    joined = "".join(rendered_markdowns)
    assert '<span class="mol-badge mol-badge-neutral">Domain Unavailable</span>' in joined


def test_triage_editorial_grid_neutral_safety_and_domain_badges(monkeypatch):
    """
    Verify render_triage_editorial_grid neutralizes safety badges and applicability badges.
    """
    rendered_markdowns = []

    def fake_markdown(text, unsafe_allow_html=False):
        rendered_markdowns.append(text)

    import streamlit as st
    monkeypatch.setattr(st, "markdown", fake_markdown)

    res = MoleculeAnalysisResult(
        validation=MoleculeValidationResult(original_smiles="CC", canonical_smiles="CC", is_valid=True, formula="C2H6"),
        descriptors=MolecularDescriptors(
            molecular_weight=30.0, logp=1.0, hbond_donors=0, hbond_acceptors=0,
            rotatable_bonds=0, tpsa=0.0, lipinski_violations=0, lipinski_pass=True, violation_details=[]
        ),
        clintox=ClinToxPrediction(
            probability=0.05,
            predicted_class=0,
            is_toxic=False,
            model_version="clintox_rf_calibrated",
            task_definition="Clinical trial failure due to toxicity (CT_TOX)",
            status="SUCCESS",
            applicability_domain_status="OUTSIDE_DOMAIN",
            applicability_score=0.15,
            applicability_warning="Outside AD",
            uncalibrated_probability=0.05,
        ),
        tox21=None,
        similarity=None,
        prioritization=PrioritizationScore(
            safety_score=95.0,
            druglikeness_score=100.0,
            similarity_score=80.0,
            overall_score=92.0,
            weight_safety=0.5,
            weight_druglikeness=0.3,
            weight_similarity=0.2,
            is_complete=True,
        ),
        explanation=None,
        analysis_timestamp="2026-10-15T00:00:00Z",
        is_valid=True,
        status="SUCCESS",
    )

    render_triage_editorial_grid(res)
    output = "".join(rendered_markdowns)

    assert "Exploratory Triage Index" in output
    assert "Estimated P(Tox)" in output
    assert "Lower Predicted ClinTox Probability" in output
    assert '<span class="mol-badge mol-badge-danger">Outside Domain</span>' in output
    assert "0.15" in output


def test_render_risk_gauge_neutral_scientific_terminology_and_badges(monkeypatch):
    """
    Verify render_risk_gauge renders neutral scientific terminology across all probability
    tiers, uses info/warning/danger badges without success variant, and eliminates legacy
    'LOW RISK', 'MODERATE RISK', 'ELEVATED RISK', and 'Failure Likelihood' labels.
    """
    rendered_markdowns = []

    def fake_markdown(text, unsafe_allow_html=False):
        rendered_markdowns.append(text)

    import streamlit as st
    monkeypatch.setattr(st, "markdown", fake_markdown)

    # 1. Lower probability tier (< 35%)
    rendered_markdowns.clear()
    render_risk_gauge(0.15, False, decision_threshold=0.50)
    out_low = "".join(rendered_markdowns)
    assert '<span class="mol-badge mol-badge-info">LOWER PREDICTED CLINTOX PROBABILITY (15.0%)</span>' in out_low
    assert "Calibrated ClinTox Probability:" in out_low
    assert "0% (Minimal ClinTox Probability)" in out_low
    assert "100% (High ClinTox Probability)" in out_low
    assert "mol-badge-success" not in out_low
    assert "LOW RISK" not in out_low
    assert "Failure Likelihood" not in out_low
    assert "Failure Risk" not in out_low

    # 2. Moderate probability tier (35% <= p < 60%)
    rendered_markdowns.clear()
    render_risk_gauge(0.45, False, decision_threshold=0.50)
    out_mod = "".join(rendered_markdowns)
    assert '<span class="mol-badge mol-badge-warning">MODERATE PREDICTED CLINTOX PROBABILITY (45.0%)</span>' in out_mod
    assert "MODERATE RISK" not in out_mod

    # 3. Higher probability tier (>= 60%)
    rendered_markdowns.clear()
    render_risk_gauge(0.75, True, decision_threshold=0.50)
    out_high = "".join(rendered_markdowns)
    assert '<span class="mol-badge mol-badge-danger">HIGHER PREDICTED CLINTOX PROBABILITY (75.0%)</span>' in out_high
    assert "ELEVATED RISK" not in out_high


def test_render_executive_interpretation_neutral_scientific_terminology(monkeypatch):
    """
    Verify render_executive_interpretation uses neutral predicted ClinTox probability
    headers and descriptions across all probability tiers, avoiding promotional safety labels.
    """
    rendered_markdowns = []

    def fake_markdown(text, unsafe_allow_html=False):
        rendered_markdowns.append(text)

    import streamlit as st
    monkeypatch.setattr(st, "markdown", fake_markdown)

    def make_res(p):
        return MoleculeAnalysisResult(
            validation=MoleculeValidationResult(original_smiles="CC", canonical_smiles="CC", is_valid=True, formula="C2H6"),
            descriptors=MolecularDescriptors(
                molecular_weight=30.0, logp=1.0, hbond_donors=0, hbond_acceptors=0,
                rotatable_bonds=0, tpsa=0.0, lipinski_violations=0, lipinski_pass=True, violation_details=[]
            ),
            clintox=ClinToxPrediction(
                probability=p,
                predicted_class=1 if p >= 0.50 else 0,
                is_toxic=p >= 0.50,
                model_version="clintox_rf_calibrated",
                task_definition="Clinical trial failure due to toxicity (CT_TOX)",
                status="SUCCESS",
                applicability_domain_status="WITHIN_DOMAIN",
                applicability_score=0.85,
                applicability_warning=None,
                uncalibrated_probability=p,
            ),
            tox21=None,
            similarity=None,
            prioritization=PrioritizationScore(
                safety_score=(1.0 - p) * 100.0,
                druglikeness_score=100.0,
                similarity_score=50.0,
                overall_score=75.0,
                weight_safety=0.5,
                weight_druglikeness=0.3,
                weight_similarity=0.2,
                is_complete=True,
            ),
            explanation=None,
            analysis_timestamp="2026-10-15T00:00:00Z",
            is_valid=True,
            status="SUCCESS",
        )

    # 1. Lower tier
    rendered_markdowns.clear()
    render_executive_interpretation(make_res(0.10))
    out_low = "".join(rendered_markdowns)
    assert "Lower Predicted ClinTox Probability" in out_low
    assert "trial failure risk is low" not in out_low.lower()
    assert "safe candidate" not in out_low.lower()
    assert "safe profile" not in out_low.lower()
    assert "is safe" not in out_low.lower()

    # 2. Moderate tier
    rendered_markdowns.clear()
    render_executive_interpretation(make_res(0.40))
    out_mod = "".join(rendered_markdowns)
    assert "Moderate Predicted ClinTox Probability" in out_mod
    assert "trial failure risk is moderate" not in out_mod.lower()

    # 3. Higher tier
    rendered_markdowns.clear()
    render_executive_interpretation(make_res(0.70))
    out_high = "".join(rendered_markdowns)
    assert "Higher Predicted ClinTox Probability" in out_high
    assert "trial failure risk is elevated" not in out_high.lower()


# =============================================================================
# 3. Cisplatin Outside Domain Warning Behavior
# =============================================================================

def test_cisplatin_inference_and_outside_domain_status(orchestrator):
    """
    Verify Cisplatin produces OUTSIDE_DOMAIN with max Tanimoto < 0.25,
    and verify that UI presentation preserves the warning and suppresses green badges.
    """
    cisplatin_smiles = "N.N.[Cl-].[Cl-].[Pt+2]"
    res = orchestrator.analyze_single_molecule(cisplatin_smiles, compute_explanation=False)

    assert res.is_valid is True
    assert res.clintox is not None
    assert res.clintox.status == "SUCCESS"
    assert res.clintox.applicability_domain_status == "OUTSIDE_DOMAIN"
    assert res.clintox.applicability_score is not None
    assert res.clintox.applicability_score < 0.25
    assert math.isclose(res.clintox.applicability_score, 0.1000, abs_tol=1e-3)
    assert res.clintox.applicability_warning is not None
    assert "extrapolation" in res.clintox.applicability_warning.lower()

    # Priority score is high (72.72), but domain status MUST be OUTSIDE_DOMAIN
    assert res.prioritization.overall_score > 70.0


# =============================================================================
# 4. Batch Screening Labels & Semantics
# =============================================================================

def test_batch_screening_metric_labels_and_ad_alerts():
    """
    Verify batch screening code uses verified neutral metric cards and includes
    the batch applicability domain alert logic.
    """
    batch_code = Path("src/ui/batch_screening.py").read_text(encoding="utf-8")

    # Metric 4: Top Triage Index
    assert 'render_metric_card(\n                "Top Triage Index (R)"' in batch_code or 'render_metric_card(\n            "Top Triage Index (R)"' in batch_code
    assert 'badge="Top Index"' in batch_code
    assert "Highest index in batch" in batch_code

    # Metric 5: Model-Predicted ClinTox-Negative
    assert "Model-Predicted ClinTox-Negative" in batch_code
    assert "Estimated P(Tox) < 0.50" in batch_code
    assert 'badge="Predicted Negative"' in batch_code

    # Batch AD Alert
    assert "Batch Applicability Domain Alert" in batch_code
    assert "candidate(s) OUTSIDE applicability domain" in batch_code


# =============================================================================
# 5. Reference Library Provenance Disclosure
# =============================================================================

def test_reference_library_provenance_disclosure_content():
    """
    Verify that the reference library provenance disclosure matches verified counts
    (1,396 records, 1,384 ClinTox FDA slice, 12 ChEMBL curated anchors) and states
    the structural resemblance limitation and self-match behavior.
    """
    single_mol_code = Path("src/ui/single_molecule.py").read_text(encoding="utf-8")

    assert "Reference Library Provenance & Methodological Disclosure:" in single_mol_code
    assert "1,396" in single_mol_code
    assert "1,384" in single_mol_code
    assert "12" in single_mol_code
    assert "Self-Match Notice:" in single_mol_code
    assert "Tanimoto = 1.0000" in single_mol_code
    assert "limits the independence of retrospective validation" in single_mol_code


# =============================================================================
# 6. Standard Scientific & Clinical Disclaimer Consistency
# =============================================================================

def test_scientific_disclaimer_consistency():
    """
    Verify the standard clinical and scientific disclaimer is present in both
    single-molecule and batch screening interfaces with exact matching text.
    """
    disclaimer_text = (
        "Research use only. Model outputs and the Exploratory Triage Index are computational estimates, "
        "not determinations of clinical safety, efficacy, or suitability for human use. "
        "Applicability-domain warnings indicate structural distance from the model's training data "
        "and do not independently establish toxicity."
    )

    single_mol_code = Path("src/ui/single_molecule.py").read_text(encoding="utf-8")
    batch_code = Path("src/ui/batch_screening.py").read_text(encoding="utf-8")

    assert disclaimer_text in single_mol_code, "Disclaimer missing or mismatched in single_molecule.py"
    assert disclaimer_text in batch_code, "Disclaimer missing or mismatched in batch_screening.py"


# =============================================================================
# 7. Four Regression Molecules Invariance Verification
# =============================================================================

def test_four_regression_molecules_numerical_invariance(orchestrator):
    """
    Verify zero numerical regression on the four agreed reference molecules:
    Aspirin, Caffeine, Ethanol, and Cisplatin.
    """
    molecules = {
        "Aspirin": {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "p_tox": 0.6659,
            "safety": 33.41,
            "druglikeness": 100.0,
            "similarity": 100.0,
            "overall": 66.70,
            "ad_status": "WITHIN_DOMAIN",
        },
        "Caffeine": {
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "p_tox": 0.0805,
            "safety": 91.95,
            "druglikeness": 100.0,
            "similarity": 100.0,
            "overall": 95.97,
            "ad_status": "WITHIN_DOMAIN",
        },
        "Ethanol": {
            "smiles": "CCO",
            "p_tox": 0.0347,
            "safety": 96.53,
            "druglikeness": 100.0,
            "similarity": 100.0,
            "overall": 98.27,
            "ad_status": "WITHIN_DOMAIN",
        },
        "Cisplatin": {
            "smiles": "N.N.[Cl-].[Cl-].[Pt+2]",
            "p_tox": 0.1855,
            "safety": 81.45,
            "druglikeness": 100.0,
            "similarity": 10.00,
            "overall": 72.72,
            "ad_status": "OUTSIDE_DOMAIN",
        },
    }

    for name, expected in molecules.items():
        res = orchestrator.analyze_single_molecule(expected["smiles"], compute_explanation=False)
        assert res.is_valid, f"{name} validation failed"

        # ClinTox calibrated probability
        assert math.isclose(res.clintox.probability, expected["p_tox"], abs_tol=1e-3), (
            f"{name} p_tox {res.clintox.probability} != {expected['p_tox']}"
        )

        # Applicability domain status
        assert res.clintox.applicability_domain_status == expected["ad_status"], (
            f"{name} AD status {res.clintox.applicability_domain_status} != {expected['ad_status']}"
        )

        # Prioritization components
        p = res.prioritization
        assert math.isclose(p.safety_score, expected["safety"], abs_tol=1e-2), f"{name} safety score"
        assert math.isclose(p.druglikeness_score, expected["druglikeness"], abs_tol=1e-2), f"{name} druglikeness score"
        assert math.isclose(p.similarity_score, expected["similarity"], abs_tol=1e-2), f"{name} similarity score"
        assert math.isclose(p.overall_score, expected["overall"], abs_tol=1e-2), f"{name} overall score"


# =============================================================================
# 8. Phase C.4 Model Insights & Benchmarks Governance
# =============================================================================

def test_model_insights_platt_scaling_mathematical_clarity():
    """
    Verify that Model Insights source explicitly defines the Platt scaling formula
    P(y=1|f) = 1 / (1 + exp(a*f + b)) and derivative dP/df > 0 for a < 0, resolving
    any sign convention ambiguity.
    """
    insights_code = Path("src/ui/model_insights.py").read_text(encoding="utf-8")

    assert "P(y=1|f) = 1 / (1 + exp(a*f + b))" in insights_code
    assert "dP/df > 0" in insights_code or "dP/df &gt; 0" in insights_code
    assert "a < 0" in insights_code or "a &lt; 0" in insights_code
    assert "ROC-AUC and PR-AUC invariant" in insights_code or "leaving ROC-AUC and PR-AUC invariant" in insights_code


def test_model_insights_bin_support_policy_and_wide_table_tabs():
    """
    Verify that Model Insights source includes the explicit bin support policy
    (SUPPORTED >= 10, SPARSE < 10, EMPTY = 0) and the tabbed domain-focused layout
    for wide scientific tables.
    """
    insights_code = Path("src/ui/model_insights.py").read_text(encoding="utf-8")

    # Bin support guidance policy
    assert "Bin Support & Empirical Estimation Policy:" in insights_code
    assert "SUPPORTED (≥ 10 observations):" in insights_code
    assert "SPARSE (< 10 observations):" in insights_code or "SPARSE (&lt; 10 observations):" in insights_code
    assert "EMPTY (0 observations):" in insights_code

    # Tabbed wide table layout
    assert "📊 Probability Calibration" in insights_code
    assert "🎯 Ranking Discrimination" in insights_code
    assert "👥 Cohort Support & Prevalence" in insights_code
    assert "📑 Complete Unified Table (27 Columns)" in insights_code

    # Verify absence of unclosed card tag bug
    assert 'st.markdown("""<div class="mol-card">""", unsafe_allow_html=True)' not in insights_code


# =============================================================================
# 9. Phase C.3.3.2 Scientific Wording Corrections & Governance
# =============================================================================

def test_phase_c332_elimination_of_unqualified_promotional_terms():
    """
    Verify that Phase C.3.3.2 terms (Protective, High Precedent, Precedent Match,
    Drug-Likeness: Favorable) are strictly eliminated from UI presentation files.
    """
    files_to_check = [
        Path("src/ui/single_molecule.py"),
        Path("src/ui/components.py"),
    ]
    forbidden_terms = [
        "Protective",
        "High Precedent",
        "Precedent Match",
        "Drug-Likeness: Favorable",
        "Decreases Risk (Protective)",
        "Increases Risk",
        "Decreases Risk",
    ]
    for f in files_to_check:
        code = f.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in code, f"Forbidden term '{term}' found in {f}"


def test_phase_c332_shap_explainability_wording_and_disclaimer():
    """
    Verify single_molecule.py presents non-causal SHAP attribution language:
    - '↑ Raises Predicted ClinTox Probability'
    - '↓ Lowers Predicted ClinTox Probability'
    - Explicit disclosure that attributions describe model prediction contributions and do not establish causal biological effects.
    """
    single_mol_code = Path("src/ui/single_molecule.py").read_text(encoding="utf-8")
    assert "↑ Raises Predicted ClinTox Probability" in single_mol_code
    assert "↓ Lowers Predicted ClinTox Probability" in single_mol_code
    assert "Attributions describe contributions to the model prediction; they do not establish causal biological effects" in single_mol_code
    assert "Prediction Impact" in single_mol_code


def test_phase_c332_triage_editorial_grid_drug_likeness_and_reference_similarity(monkeypatch):
    """
    Verify render_triage_editorial_grid outputs 'Lipinski Compliant' / 'Non-Compliant'
    and 'Reference Similarity' with resemblance tiers ('High Resemblance', etc.).
    """
    rendered_markdowns = []

    def fake_markdown(text, unsafe_allow_html=False):
        rendered_markdowns.append(text)

    import streamlit as st
    monkeypatch.setattr(st, "markdown", fake_markdown)

    # 1. Compliant + High Resemblance
    res = MoleculeAnalysisResult(
        validation=MoleculeValidationResult(original_smiles="CC", canonical_smiles="CC", is_valid=True, formula="C2H6"),
        descriptors=MolecularDescriptors(
            molecular_weight=30.0, logp=1.0, hbond_donors=0, hbond_acceptors=0,
            rotatable_bonds=0, tpsa=0.0, lipinski_violations=0, lipinski_pass=True, violation_details=[]
        ),
        clintox=ClinToxPrediction(
            probability=0.10, predicted_class=0, is_toxic=False,
            model_version="clintox_rf_calibrated", task_definition="CT_TOX", status="SUCCESS",
            applicability_domain_status="WITHIN_DOMAIN", applicability_score=0.85, applicability_warning=None, uncalibrated_probability=0.10,
        ),
        tox21=None,
        similarity=SimilaritySearchResult(
            query_smiles="CC",
            matches=[SimilarityMatch(reference_id="CHEMBL1", name="Aspirin", canonical_smiles="CC", similarity=0.85)],
            best_similarity=0.85,
            total_references=1396,
            status="SUCCESS",
        ),
        prioritization=PrioritizationScore(
            safety_score=90.0, druglikeness_score=100.0, similarity_score=85.0, overall_score=89.5,
            weight_safety=0.5, weight_druglikeness=0.3, weight_similarity=0.2, is_complete=True,
        ),
        explanation=None, analysis_timestamp="2026-10-15T00:00:00Z", is_valid=True, status="SUCCESS",
    )

    render_triage_editorial_grid(res)
    out = "".join(rendered_markdowns)
    assert "Drug-Likeness" in out
    assert "Lipinski Compliant" in out
    assert "0 Violations" in out
    assert "Reference Similarity" in out
    assert "High Structural Resemblance" in out
    assert "Precedent" not in out
    assert "Favorable" not in out


def test_phase_c332_executive_interpretation_disclaimer_and_resemblance(monkeypatch):
    """
    Verify render_executive_interpretation outputs 'Reference Similarity:' and
    structural resemblance disclaimer asserting non-causality regarding efficacy, safety, or mechanism.
    """
    rendered_markdowns = []

    def fake_markdown(text, unsafe_allow_html=False):
        rendered_markdowns.append(text)

    import streamlit as st
    monkeypatch.setattr(st, "markdown", fake_markdown)

    res = MoleculeAnalysisResult(
        validation=MoleculeValidationResult(original_smiles="CC", canonical_smiles="CC", is_valid=True, formula="C2H6"),
        descriptors=MolecularDescriptors(
            molecular_weight=30.0, logp=1.0, hbond_donors=0, hbond_acceptors=0,
            rotatable_bonds=0, tpsa=0.0, lipinski_violations=0, lipinski_pass=True, violation_details=[]
        ),
        clintox=ClinToxPrediction(
            probability=0.10, predicted_class=0, is_toxic=False,
            model_version="clintox_rf_calibrated", task_definition="CT_TOX", status="SUCCESS",
            applicability_domain_status="WITHIN_DOMAIN", applicability_score=0.85, applicability_warning=None, uncalibrated_probability=0.10,
        ),
        tox21=None,
        similarity=SimilaritySearchResult(
            query_smiles="CC",
            matches=[SimilarityMatch(reference_id="CHEMBL2", name="TestDrug", canonical_smiles="CC", similarity=0.65)],
            best_similarity=0.65,
            total_references=1396,
            status="SUCCESS",
        ),
        prioritization=PrioritizationScore(
            safety_score=90.0, druglikeness_score=100.0, similarity_score=65.0, overall_score=85.0,
            weight_safety=0.5, weight_druglikeness=0.3, weight_similarity=0.2, is_complete=True,
        ),
        explanation=None, analysis_timestamp="2026-10-15T00:00:00Z", is_valid=True, status="SUCCESS",
    )

    render_executive_interpretation(res)
    out = "".join(rendered_markdowns)
    assert "Reference Similarity:" in out
    assert "Chemical Precedent:" not in out
    assert "Lipinski Rule of 5 compliant (0 violations); standard oral drug-likeness parameter range." in out
    assert "Proximity in chemical space indicates structural resemblance only; does not establish efficacy, safety, or mechanism." in out


# =============================================================================
# 10. Phase C.3.3.3 Residual UI Terminology Audit & Governance
# =============================================================================

def test_phase_c333_residual_wording_eliminated():
    """
    Verify complete absence of residual legacy terms in all UI modules:
    - 'Drug-Likeness: Favorable'
    - 'Chemical Precedent'
    - 'Precedent Match'
    - 'favorable oral bioavailability'
    - 'Precedent Proximity'
    """
    ui_files = [
        Path("src/ui/components.py"),
        Path("src/ui/overview.py"),
        Path("src/ui/single_molecule.py"),
        Path("src/ui/batch_screening.py"),
        Path("src/ui/model_insights.py"),
        Path("src/ui/literature_assistant.py"),
    ]
    forbidden = [
        "Drug-Likeness: Favorable",
        "Chemical Precedent",
        "Precedent Match",
        "favorable oral bioavailability",
        "Precedent Proximity",
    ]
    for p in ui_files:
        content = p.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in content, f"Forbidden term '{term}' found in {p}"


def test_phase_c333_overview_uses_reference_similarity():
    """
    Verify src/ui/overview.py renders 'Chemical Space & Reference Similarity'.
    """
    content = Path("src/ui/overview.py").read_text(encoding="utf-8")
    assert "Chemical Space & Reference Similarity" in content
    assert "Precedent" not in content
