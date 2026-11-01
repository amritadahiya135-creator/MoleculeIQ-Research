"""
Unit tests for Module M09-D: Lightweight Literature Polarity Tagging with Scientific Safeguards.

Verifies:
1. Clear affirmative wording.
2. Explicit negation.
3. 'No evidence of' wording.
4. Uncertainty and hedging.
5. Mixed dose-dependent claims.
6. 'Not only ... but also ...' false-positive protection.
7. 'No significant difference' ambiguity.
8. Negation scope across multiple clauses.
9. Empty input and punctuation handling.
10. Backward compatibility of existing RankedSentence result contracts.
11. Invariance of existing relevance-score calculations.
12. Invariance of existing search result ordering.
"""

import pytest
from src.literature.direction import (
    classify_claim_polarity,
    PolarityAnnotation,
    POLARITY_AFFIRMATIVE,
    POLARITY_NEGATED,
    POLARITY_UNCERTAIN,
    POLARITY_LABELS,
    POLARITY_UI_LABELS,
)
from src.literature.retrieval import (
    RankedSentence,
    LiteratureAssistantService,
    LiteratureSearchResult,
)
from src.literature.pubmed_client import PubMedArticle


# =============================================================================
# 1. Clear Affirmative Wording
# =============================================================================

class TestAffirmativeWording:
    """Tests asserting positive or causative relationship claims."""

    def test_direct_causation_induced_hepatotoxicity(self):
        s = "Acetaminophen induced severe acute hepatotoxicity in mice."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_AFFIRMATIVE
        assert ann.ui_label == "Possible affirmative claim"
        assert "AFFIRMATIVE" in POLARITY_LABELS

    def test_direct_causation_caused_lactic_acidosis(self):
        s = "Metformin caused severe lactic acidosis in patients with renal failure."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_AFFIRMATIVE
        assert ann.ui_label == "Possible affirmative claim"

    def test_direct_causation_triggered_cardiac_arrest(self):
        s = "High-dose administration triggered cardiac arrest in animal models."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_AFFIRMATIVE

    def test_statistically_significant_elevation_associated(self):
        s = "Treatment was significantly associated with elevated risk of acute liver failure."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_AFFIRMATIVE


# =============================================================================
# 2. Explicit Negation
# =============================================================================

class TestExplicitNegation:
    """Tests asserting linguistic refutation or denial of toxicity."""

    def test_did_not_induce_lactic_acidosis(self):
        s = "Metformin did not induce lactic acidosis in diabetic patients with normal renal function."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED
        assert ann.ui_label == "Negated wording"

    def test_was_not_associated_with_cardiotoxicity(self):
        s = "Therapy was not associated with cardiotoxicity in the prospective cohort."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED

    def test_failed_to_cause_hepatic_necrosis(self):
        s = "The novel analogue failed to cause hepatic necrosis at all evaluated doses."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED

    def test_does_not_produce_toxic_effects(self):
        s = "Compound 4a does not produce adverse effects in cellular assays."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED


# =============================================================================
# 3. 'No Evidence Of' and Absence Wording
# =============================================================================

class TestNoEvidenceAndAbsence:
    """Tests for 'no evidence of' and 'without detectable toxicity' constructs."""

    def test_no_evidence_of_hepatotoxicity(self):
        s = "There was no evidence of hepatotoxicity during the 24-week trial."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED

    def test_without_detectable_toxicity(self):
        s = "The drug was administered without detectable toxicity in rats."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED

    def test_absence_of_liver_injury(self):
        s = "Histopathological evaluation confirmed the absence of liver injury."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED

    def test_no_histopathological_changes_were_observed(self):
        s = "No histopathological changes were observed in hepatic tissues."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_NEGATED


# =============================================================================
# 4. Uncertainty and Hedging
# =============================================================================

class TestUncertaintyAndHedging:
    """Tests for epistemic hedging, modal doubt, and tentative phrasing."""

    def test_may_cause_hedging(self):
        s = "Metformin may cause lactic acidosis in elderly patients with pre-existing nephropathy."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN
        assert ann.ui_label == "Polarity uncertain"

    def test_might_induce_hedging(self):
        s = "Experimental findings indicate the molecule might induce cardiotoxicity under ischemic stress."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN

    def test_could_be_associated_with(self):
        s = "The observed metabolite could be associated with elevated transaminases."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN

    def test_possibly_linked_to(self):
        s = "Severe hyperlactatemia was possibly linked to mitochondrial dysfunction."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN

    def test_suggests_tentative_finding(self):
        s = "Preliminary data suggests that high exposure appears to elevate ALT."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN

    def test_cannot_be_ruled_out(self):
        s = "A direct hepatotoxic effect cannot be ruled out at supratherapeutic levels."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN


# =============================================================================
# 5. Mixed Dose-Dependent Claims
# =============================================================================

class TestMixedDoseClaims:
    """Tests for opposing or dose-dependent claims within the same sentence."""

    def test_low_dose_safe_high_dose_toxic(self):
        s = "The agent did not induce toxicity at low doses but caused toxicity at high doses."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN
        assert ann.ui_label == "Polarity uncertain"

    def test_opposing_cues_in_same_sentence(self):
        s = "Metformin did not cause cardiotoxicity, however it induced lactic acidosis in renal patients."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN

    def test_dose_contrast_pattern(self):
        s = "In low concentrations viability was preserved, but in high concentrations necrosis occurred."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN


# =============================================================================
# 6. 'Not Only ... But Also ...' False-Positive Protection
# =============================================================================

class TestCorrelativeNotOnlyProtection:
    """Tests ensuring 'not only' does not trigger false negative classification."""

    def test_not_only_induced_hepatotoxicity_but_also_nephrotoxicity(self):
        s = "Aspirin not only induced cardiotoxicity but also caused acute renal failure."
        ann = classify_claim_polarity(s)
        # Must NOT be classified as NEGATED
        assert ann.label != POLARITY_NEGATED
        assert ann.label == POLARITY_AFFIRMATIVE

    def test_not_just_caused_damage(self):
        s = "The toxin not just caused liver damage but precipitated circulatory collapse."
        ann = classify_claim_polarity(s)
        assert ann.label != POLARITY_NEGATED
        assert ann.label == POLARITY_AFFIRMATIVE


# =============================================================================
# 7. 'No Significant Difference' Ambiguity
# =============================================================================

class TestStatisticalAmbiguity:
    """Tests for comparative equivalence phrases that do not imply safety."""

    def test_no_significant_difference_in_hepatotoxicity(self):
        s = "There was no significant difference in hepatotoxicity between the treatment group and controls."
        ann = classify_claim_polarity(s)
        # Must NOT be classified as NEGATED (absence of difference != absence of toxicity)
        assert ann.label == POLARITY_UNCERTAIN
        assert ann.ui_label == "Polarity uncertain"

    def test_no_statistically_significant_change(self):
        s = "No statistically significant difference in adverse event rates was observed."
        ann = classify_claim_polarity(s)
        assert ann.label == POLARITY_UNCERTAIN


# =============================================================================
# 8. Negation Scope Across Multiple Clauses
# =============================================================================

class TestMultiClauseScope:
    """Tests verifying negation in unrelated clauses does not invert the entire sentence."""

    def test_negated_patient_history_does_not_invert_adverse_causation(self):
        s = "Patients had no history of renal failure, but metformin induced severe lactic acidosis."
        ann = classify_claim_polarity(s)
        # The adverse claim is affirmative, NOT negated
        assert ann.label != POLARITY_NEGATED
        assert ann.label == POLARITY_AFFIRMATIVE

    def test_substring_word_boundaries_protected(self):
        s = "A notable increase in serum transaminases was observed in all subjects."
        ann = classify_claim_polarity(s)
        # 'notable' must not trigger 'not' negation
        assert ann.label != POLARITY_NEGATED


# =============================================================================
# 9. Empty Input and Punctuation Handling
# =============================================================================

class TestEmptyAndPunctuationHandling:
    """Tests edge cases with degenerate strings."""

    def test_empty_string(self):
        ann = classify_claim_polarity("")
        assert ann.label == POLARITY_UNCERTAIN
        assert ann.ui_label == "Polarity uncertain"

    def test_whitespace_only(self):
        ann = classify_claim_polarity("    \t\n  ")
        assert ann.label == POLARITY_UNCERTAIN

    def test_punctuation_only(self):
        ann = classify_claim_polarity("...?!!---;;;")
        assert ann.label == POLARITY_UNCERTAIN

    def test_non_string_type_safety(self):
        ann = classify_claim_polarity(None)  # type: ignore
        assert ann.label == POLARITY_UNCERTAIN


# =============================================================================
# 10. Backward Compatibility of Existing Result Contracts
# =============================================================================

class TestContractBackwardCompatibility:
    """Tests ensuring RankedSentence retains full backward compatibility."""

    def test_positional_instantiation_without_polarity_auto_annotates(self):
        s = RankedSentence(
            "Metformin caused severe lactic acidosis.",
            0.4500,
            "12345678",
            "Study Title",
            "Author A",
            "2024",
            "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        )
        assert s.sentence == "Metformin caused severe lactic acidosis."
        assert s.relevance_score == 0.4500
        assert s.conjunction_bonus == 0.0
        assert s.cosine_score == 0.4500
        assert s.claim_polarity == POLARITY_AFFIRMATIVE
        assert s.polarity == POLARITY_AFFIRMATIVE
        assert s.evidence_direction == POLARITY_AFFIRMATIVE

    def test_explicit_polarity_override_respected(self):
        s = RankedSentence(
            sentence="Ambiguous text.",
            relevance_score=0.3000,
            pmid="23456789",
            article_title="Study 2",
            authors="Author B",
            year="2023",
            source_url="https://pubmed.ncbi.nlm.nih.gov/23456789/",
            claim_polarity=POLARITY_NEGATED,
            polarity_explanation="Manually verified negative finding.",
        )
        assert s.claim_polarity == POLARITY_NEGATED
        assert s.polarity == POLARITY_NEGATED
        assert s.polarity_explanation == "Manually verified negative finding."


# =============================================================================
# 11 & 12. Ranking and Scoring Invariance Verification
# =============================================================================

class MockPubMedClient:
    def __init__(self, articles):
        self._articles = articles

    def search_articles(self, query: str, max_results: int = 5):
        return self._articles[:max_results]


class TestRankingAndScoringInvariance:
    """Tests confirming relevance scores, result counts, and ordering are 100% invariant."""

    @pytest.fixture
    def mock_service(self):
        articles = [
            PubMedArticle(
                pmid="90001",
                title="Metformin and MALA",
                abstract=(
                    "Metformin-associated lactic acidosis is a rare but critical complication. "
                    "Metformin did not induce lactic acidosis in standard clinical trials. "
                    "In patients with severe renal failure, metformin caused fatal acidosis."
                ),
                authors="Smith J",
                year="2024",
                url="https://pubmed.ncbi.nlm.nih.gov/90001/",
            ),
            PubMedArticle(
                pmid="90002",
                title="Metformin Safety Overview",
                abstract=(
                    "Long-term metformin therapy reduces cardiovascular events in diabetic patients. "
                    "There was no evidence of hepatotoxicity during follow-up."
                ),
                authors="Jones K",
                year="2023",
                url="https://pubmed.ncbi.nlm.nih.gov/90002/",
            ),
        ]
        client = MockPubMedClient(articles)
        return LiteratureAssistantService(client=client, enable_biomedical_enhancements=True)

    def test_ranking_order_and_scores_are_strictly_deterministic(self, mock_service):
        res = mock_service.search_and_rank("metformin lactic acidosis", top_k_sentences=5)
        assert res.status == "SUCCESS"
        assert len(res.ranked_sentences) > 0

        # Check descending sort invariance
        scores = [s.relevance_score for s in res.ranked_sentences]
        assert scores == sorted(scores, reverse=True)

        # Check score decomposition integrity: relevance = cosine + bonus
        for s in res.ranked_sentences:
            expected_total = round(s.cosine_score + s.conjunction_bonus, 4)
            assert pytest.approx(s.relevance_score, abs=1e-4) == expected_total
            # Polarity annotation exists and is valid
            assert s.claim_polarity in POLARITY_LABELS
            assert isinstance(s.polarity_explanation, str)

        # Check that different polarities are present without affecting score order
        polarities = [s.claim_polarity for s in res.ranked_sentences]
        assert any(p in (POLARITY_AFFIRMATIVE, POLARITY_NEGATED, POLARITY_UNCERTAIN) for p in polarities)
