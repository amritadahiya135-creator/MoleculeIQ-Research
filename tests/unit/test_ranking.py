"""Unit tests for candidate prioritization and ranking engine (Module M07)."""

import pytest
from src.ranking.scoring import calculate_prioritization_score, perform_sensitivity_analysis, PrioritizationScore


class TestCandidateRanking:
    """Tests for Module M07: Candidate Ranking Engine."""

    def test_srd_reference_acceptance_vector(self):
        """
        Verify exact SRD acceptance test requirement:
        P_tox = 0.20, Lipinski = Pass, best similarity = 0.75
        S = (1 - 0.20) * 100 = 80.0
        D = 100.0
        M = 100 * 0.75 = 75.0
        R = 0.5 * 80 + 0.3 * 100 + 0.2 * 75 = 40 + 30 + 15 = 85.0
        """
        score = calculate_prioritization_score(
            p_tox=0.20,
            lipinski_pass=True,
            best_similarity=0.75,
        )

        assert isinstance(score, PrioritizationScore)
        assert score.is_complete is True
        assert score.status == "SUCCESS"
        assert score.safety_score == 80.0
        assert score.druglikeness_score == 100.0
        assert score.similarity_score == 75.0
        assert score.overall_score == 85.0

    def test_lipinski_fail_score_penalty(self):
        """
        P_tox = 0.10 (S = 90.0)
        Lipinski = Fail (D = 40.0)
        Similarity = 0.50 (M = 50.0)
        R = 0.5 * 90 + 0.3 * 40 + 0.2 * 50 = 45 + 12 + 10 = 67.0
        """
        score = calculate_prioritization_score(
            p_tox=0.10,
            lipinski_pass=False,
            best_similarity=0.50,
        )
        assert score.is_complete is True
        assert score.druglikeness_score == 40.0
        assert score.overall_score == 67.0

    def test_boundary_zero_and_maximum_values(self):
        # Best possible candidate: P_tox = 0.0, Lipinski = Pass, Sim = 1.0 -> R = 100.0
        best_score = calculate_prioritization_score(p_tox=0.0, lipinski_pass=True, best_similarity=1.0)
        assert best_score.overall_score == 100.0

        # Worst possible candidate: P_tox = 1.0, Lipinski = Fail, Sim = 0.0 -> R = 0.3 * 40 = 12.0
        worst_score = calculate_prioritization_score(p_tox=1.0, lipinski_pass=False, best_similarity=0.0)
        assert worst_score.overall_score == 12.0

    def test_missing_data_returns_incomplete_status(self):
        score1 = calculate_prioritization_score(p_tox=None, lipinski_pass=True, best_similarity=0.8)
        assert score1.is_complete is False
        assert score1.status == "INCOMPLETE_DATA"

        score2 = calculate_prioritization_score(p_tox=0.3, lipinski_pass=None, best_similarity=0.8)
        assert score2.is_complete is False
        assert score2.status == "INCOMPLETE_DATA"

        score3 = calculate_prioritization_score(p_tox=0.3, lipinski_pass=True, best_similarity=None)
        assert score3.is_complete is False
        assert score3.status == "INCOMPLETE_DATA"

    def test_sensitivity_analysis(self):
        p_list = [0.1, 0.5, 0.9]
        lip_list = [True, True, False]
        sim_list = [0.8, 0.4, 0.2]

        analysis = perform_sensitivity_analysis(p_list, lip_list, sim_list)
        assert "scenarios" in analysis
        assert "scores_by_scenario" in analysis
        assert len(analysis["scores_by_scenario"]["baseline"]) == 3
        # Candidate 1 should always outrank Candidate 3 across all reasonable scenarios
        assert analysis["scores_by_scenario"]["baseline"][0] > analysis["scores_by_scenario"]["baseline"][2]
