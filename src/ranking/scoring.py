"""Candidate prioritization and transparent composite ranking engine (Module M07)."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass(frozen=True)
class PrioritizationScore:
    """Individual candidate ranking evaluation with component breakdown."""
    safety_score: float         # S = (1 - P_tox) * 100
    druglikeness_score: float   # D = 100 if Lipinski pass else 40
    similarity_score: float     # M = 100 * best Tanimoto
    overall_score: float        # R = w_s*S + w_d*D + w_m*M
    weight_safety: float
    weight_druglikeness: float
    weight_similarity: float
    is_complete: bool
    formula_version: str = "v1.0_baseline"
    status: str = "SUCCESS"
    error_message: Optional[str] = None


def calculate_prioritization_score(
    p_tox: Optional[float],
    lipinski_pass: Optional[bool],
    best_similarity: Optional[float],
    weight_safety: float = 0.5,
    weight_druglikeness: float = 0.3,
    weight_similarity: float = 0.2,
    pass_druglikeness_val: float = 100.0,
    fail_druglikeness_val: float = 40.0,
) -> PrioritizationScore:
    """
    Compute transparent composite candidate score according to the PRD/SRD baseline formula:

        S = (1 - P_tox) * 100
        D = 100 if Lipinski_pass else 40
        M = 100 * best_similarity
        R = 0.5 * S + 0.3 * D + 0.2 * M

    Parameters
    ----------
    p_tox : Optional[float]
        Predicted ClinTox toxicity probability in [0.0, 1.0].
    lipinski_pass : Optional[bool]
        Lipinski Rule of 5 compliance flag.
    best_similarity : Optional[float]
        Maximum Tanimoto coefficient to approved reference compound in [0.0, 1.0].
    weight_safety : float, default=0.5
    weight_druglikeness : float, default=0.3
    weight_similarity : float, default=0.2

    Returns
    -------
    PrioritizationScore
        Structured object containing overall score R and individual components S, D, M.
    """
    # Guard against missing or unavailable upstream calculations
    if p_tox is None or lipinski_pass is None or best_similarity is None:
        return PrioritizationScore(
            safety_score=0.0 if p_tox is None else round((1.0 - p_tox) * 100.0, 2),
            druglikeness_score=0.0 if lipinski_pass is None else (pass_druglikeness_val if lipinski_pass else fail_druglikeness_val),
            similarity_score=0.0 if best_similarity is None else round(best_similarity * 100.0, 2),
            overall_score=0.0,
            weight_safety=weight_safety,
            weight_druglikeness=weight_druglikeness,
            weight_similarity=weight_similarity,
            is_complete=False,
            status="INCOMPLETE_DATA",
            error_message="One or more required scoring components (P_tox, Lipinski, Similarity) are missing.",
        )

    # Component 1: Safety score (S)
    p_clamped = max(0.0, min(1.0, float(p_tox)))
    s_score = (1.0 - p_clamped) * 100.0

    # Component 2: Drug-likeness score (D)
    d_score = pass_druglikeness_val if bool(lipinski_pass) else fail_druglikeness_val

    # Component 3: Reference similarity score (M)
    sim_clamped = max(0.0, min(1.0, float(best_similarity)))
    m_score = sim_clamped * 100.0

    # Composite weighted score (R)
    r_score = (
        (weight_safety * s_score)
        + (weight_druglikeness * d_score)
        + (weight_similarity * m_score)
    )

    return PrioritizationScore(
        safety_score=round(s_score, 2),
        druglikeness_score=round(d_score, 2),
        similarity_score=round(m_score, 2),
        overall_score=round(r_score, 2),
        weight_safety=weight_safety,
        weight_druglikeness=weight_druglikeness,
        weight_similarity=weight_similarity,
        is_complete=True,
        status="SUCCESS",
        error_message=None,
    )


def perform_sensitivity_analysis(
    p_tox_list: List[float],
    lipinski_list: List[bool],
    sim_list: List[float],
    perturbation: float = 0.1,
) -> Dict[str, Any]:
    """
    Perform weight perturbation analysis to assess ranking stability across candidate molecules.

    Tests baseline weights (0.5, 0.3, 0.2) against alternative plausible configurations.
    """
    scenarios = {
        "baseline": (0.5, 0.3, 0.2),
        "safety_heavy": (0.7, 0.2, 0.1),
        "druglike_heavy": (0.3, 0.5, 0.2),
        "similarity_heavy": (0.4, 0.2, 0.4),
        "balanced": (0.34, 0.33, 0.33),
    }

    scenario_scores: Dict[str, List[float]] = {}
    for name, (ws, wd, wm) in scenarios.items():
        scores = []
        for p, lip, sim in zip(p_tox_list, lipinski_list, sim_list):
            res = calculate_prioritization_score(p, lip, sim, weight_safety=ws, weight_druglikeness=wd, weight_similarity=wm)
            scores.append(res.overall_score)
        scenario_scores[name] = scores

    return {
        "scenarios": scenarios,
        "scores_by_scenario": scenario_scores,
    }
