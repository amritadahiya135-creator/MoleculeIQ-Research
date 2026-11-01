"""
Unit tests for Phase D.4: Controlled Biomedical Literature Retrieval Improvements.

Verifies:
- Curated biomedical synonym normalization
- Clinical morphology normalization
- Concept conjunction detection and scoring bonus
- Additive score transparency (cosine_score + conjunction_bonus = relevance_score)
- Intra-article and cross-sentence duplicate suppression
- Toggle behavior (enable_biomedical_enhancements=True vs False)
- Robustness against missing conjunction_bonus attribute (Phase D.4.2 regression fix)
- Negative tests for substring and acronym collisions (e.g. ASA physical status)
- Safe fallback behavior for uncurated multiword queries (zero bonus)
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.literature.pubmed_client import PubMedArticle
from src.literature.retrieval import (
    LiteratureAssistantService,
    RankedSentence,
    apply_biomedical_synonyms,
    apply_biomedical_morphology,
    detect_concept_conjunction,
    split_into_sentences,
    normalize_sentence_key,
    BIOMEDICAL_SYNONYM_MAPPINGS,
    BIOMEDICAL_MORPHOLOGY_MAPPINGS,
)


class TestBiomedicalSynonymExpansion:
    """Tests for curated domain-specific synonym expansion."""

    def test_paracetamol_acetaminophen_expansion(self):
        text = "Patients taking paracetamol had elevated ALT."
        expanded = apply_biomedical_synonyms(text)
        assert "acetaminophen" in expanded.lower()
        assert "paracetamol" in expanded.lower()

    def test_hepatotoxicity_liver_injury_expansion(self):
        text = "Drug-induced hepatotoxicity is a clinical concern."
        expanded = apply_biomedical_synonyms(text)
        assert "liver injury" in expanded.lower()
        assert "hepatotoxicity" in expanded.lower()

    def test_cardiotoxicity_cardiac_expansion(self):
        text = "Assessment of cardiotoxicity risk."
        expanded = apply_biomedical_synonyms(text)
        assert "cardiovascular toxicity" in expanded.lower()
        assert "cardiotoxicity" in expanded.lower()

    def test_lactic_acidosis_hyperlactatemia_expansion(self):
        text = "Severe lactic acidosis occurred."
        expanded = apply_biomedical_synonyms(text)
        assert "hyperlactatemia" in expanded.lower()
        assert "lactic acidosis" in expanded.lower()

    def test_no_unwanted_mutations_on_unrelated_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        expanded = apply_biomedical_synonyms(text)
        assert expanded == text


class TestBiomedicalMorphologyNormalization:
    """Tests for inflectional and derivational biomedical morphology normalization."""

    def test_cardiotoxic_to_cardiotoxicity(self):
        text = "The agent is cardiotoxic in high doses."
        normalized = apply_biomedical_morphology(text)
        assert "cardiotoxicity" in normalized

    def test_hepatotoxic_to_hepatotoxicity(self):
        text = "Known hepatotoxic metabolites were detected."
        normalized = apply_biomedical_morphology(text)
        assert "hepatotoxicity" in normalized

    def test_acidemic_to_acidosis(self):
        text = "Severe acidemia was observed."
        normalized = apply_biomedical_morphology(text)
        assert "acidosis" in normalized

    def test_necrotic_to_necrosis(self):
        text = "Submassive necrotic tissue changes."
        normalized = apply_biomedical_morphology(text)
        assert "necrosis" in normalized


class TestConceptConjunctionDetection:
    """Tests for multi-concept conjunction detection (drug entity AND adverse endpoint)."""

    def test_conjunction_positive_both_present(self):
        query = "aspirin cardiotoxicity"
        sentence = "Aspirin administration was associated with cardiotoxicity in elderly patients."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence)
        assert has_drug is True
        assert has_endpoint is True
        assert has_conj is True

    def test_conjunction_negative_drug_only(self):
        query = "aspirin cardiotoxicity"
        sentence = "Aspirin is an acetylated salicylate widely used as an antiplatelet agent."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence)
        assert has_drug is True
        assert has_endpoint is False
        assert has_conj is False

    def test_conjunction_negative_endpoint_only(self):
        query = "aspirin cardiotoxicity"
        sentence = "Cardiotoxicity remains a major cause of compound attrition during drug discovery."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence)
        assert has_drug is False
        assert has_endpoint is True
        assert has_conj is False

    def test_conjunction_with_synonyms(self):
        query = "paracetamol hepatotoxicity"
        sentence = "Severe acetaminophen overdose leads to acute liver injury."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence)
        assert has_drug is True
        assert has_endpoint is True
        assert has_conj is True

    def test_endpoint_only_sentence_without_queried_drug_yields_no_conjunction(self):
        """Finding 2 Regression Test: Endpoint-only sentence must NEVER receive conjunction bonus."""
        query = "atorvastatin hepatotoxicity and liver safety"
        sentence = "Hepatotoxicity and liver injury are important endpoints in pharmaceutical safety."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence)
        assert has_drug is False
        assert has_endpoint is True
        assert has_conj is False

    def test_unrecognized_multiword_query_yields_no_conjunction(self):
        """Finding 2 Regression Test: Unrecognized queries must not blindly bisect into false groups."""
        query = "compound_xyz kidney tubular necrosis"
        sentence = "Compound_xyz administration produced acute kidney injury."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence)
        # Because compound_xyz is uncurated, neither concept group is falsely manufactured
        assert has_conj is False

    def test_asa_physical_status_does_not_trigger_aspirin(self):
        """Finding 2 Regression Test: Acronym ASA in surgical classification must not trigger aspirin."""
        query = "aspirin cardiotoxicity"
        sentence1 = "Patients presenting with ASA physical status III experienced cardiotoxicity."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence1)
        assert has_drug is False
        assert has_endpoint is True
        assert has_conj is False

        sentence2 = "ASA score was 2 with severe myocardial infarction."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence2)
        assert has_drug is False
        assert has_conj is False

        sentence3 = "Patients categorized by ASA classification II had cardiovascular events."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence3)
        assert has_drug is False
        assert has_conj is False

    def test_valid_asa_aspirin_alias_still_matches(self):
        """Confirm valid ASA drug mentions in medical context still match."""
        query = "aspirin cardiotoxicity"
        sentence = "Daily low-dose ASA therapy reduced ponatinib-induced cardiotoxicity."
        has_drug, has_endpoint, has_conj = detect_concept_conjunction(query, sentence)
        assert has_drug is True
        assert has_endpoint is True
        assert has_conj is True


class TestRankedSentenceRobustnessAndFormatting:
    """Tests for RankedSentence model integrity and backward compatibility (Finding 1)."""

    def test_positional_backward_compatibility_defaults(self):
        """Verify 7-argument positional instantiation sets valid mathematical defaults."""
        s = RankedSentence(
            sentence="Metformin reduces hepatic gluconeogenesis.",
            relevance_score=0.2850,
            pmid="12345678",
            article_title="Metformin study",
            authors="Author A",
            year="2024",
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        )
        assert s.conjunction_bonus == 0.0
        assert s.cosine_score == 0.2850
        assert pytest.approx(s.relevance_score, abs=1e-5) == s.cosine_score + s.conjunction_bonus

    def test_explicit_decomposition_preserved(self):
        """Verify explicit score decomposition retains exact component values."""
        s = RankedSentence(
            sentence="Aspirin reduces ponatinib cardiotoxicity.",
            relevance_score=0.3799,
            pmid="39357328",
            article_title="Aspirin cardio study",
            authors="Zhang et al.",
            year="2024",
            source_url="https://pubmed.ncbi.nlm.nih.gov/39357328/",
            cosine_score=0.3299,
            conjunction_bonus=0.0500,
        )
        assert s.conjunction_bonus == 0.0500
        assert s.cosine_score == 0.3299
        assert pytest.approx(s.relevance_score, abs=1e-4) == s.cosine_score + s.conjunction_bonus

    def test_reproduce_and_handle_legacy_missing_attribute(self):
        """Reproduces legacy object lacking conjunction_bonus and tests safe UI formatting."""
        class LegacyRankedSentenceMock:
            def __init__(self, sentence, relevance_score, pmid, title, authors, year, source_url):
                self.sentence = sentence
                self.relevance_score = relevance_score
                self.pmid = pmid
                self.article_title = title
                self.authors = authors
                self.year = year
                self.source_url = source_url
                # Missing cosine_score and conjunction_bonus

        legacy_obj = LegacyRankedSentenceMock(
            "Legacy statement text.", 0.2500, "1001", "Old Study", "Author", "2020", "url"
        )

        # Direct attribute access would fail:
        with pytest.raises(AttributeError):
            _ = legacy_obj.conjunction_bonus

        # UI robust resolution:
        score_val = legacy_obj.relevance_score
        bon_val = getattr(legacy_obj, "conjunction_bonus", 0.0) or 0.0
        cos_val = getattr(legacy_obj, "cosine_score", None)
        if cos_val is None:
            cos_val = round(score_val - bon_val, 4)

        assert bon_val == 0.0
        assert cos_val == 0.2500
        badge_text = f"TF-IDF Cosine: {score_val:.3f}" if bon_val == 0.0 else f"Score: {score_val:.3f}"
        assert badge_text == "TF-IDF Cosine: 0.250"


class TestLiteratureAssistantServiceEnhancements:
    """Integration tests for enhanced LiteratureAssistantService ranking."""

    def _create_mock_client(self, query: str, abstract: str) -> MagicMock:
        mock_article = PubMedArticle(
            pmid="12345678",
            title="Cardiovascular pharmacology study",
            abstract=abstract,
            authors=["Smith J", "Doe A"],
            year="2024",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        )
        client = MagicMock()
        client.search_articles.return_value = [mock_article]
        client.last_error = None
        return client

    def test_conjunction_bonus_applied_and_transparent(self):
        abstract = (
            "Aspirin reduces cardiovascular thrombosis. "
            "In rare cases aspirin induces cardiotoxicity in compromised myocardium. "
            "Nonsteroidal anti-inflammatory agents have distinct safety profiles."
        )
        mock_client = self._create_mock_client("aspirin cardiotoxicity", abstract)
        service = LiteratureAssistantService(
            client=mock_client,
            enable_biomedical_enhancements=True,
            conjunction_bonus_weight=0.05,
        )

        result = service.search_and_rank("aspirin cardiotoxicity", max_articles=1, top_k_sentences=5)
        assert result.status == "SUCCESS"
        assert len(result.ranked_sentences) > 0

        # The sentence containing both aspirin and cardiotoxicity should have conjunction_bonus > 0
        conj_sents = [s for s in result.ranked_sentences if "induces cardiotoxicity" in s.sentence]
        assert len(conj_sents) == 1
        target_sent = conj_sents[0]

        assert target_sent.conjunction_bonus == 0.05
        assert target_sent.cosine_score is not None
        assert pytest.approx(target_sent.relevance_score, abs=1e-5) == target_sent.cosine_score + 0.05

    def test_disabled_enhancements_preserve_pure_cosine(self):
        abstract = (
            "Aspirin reduces cardiovascular thrombosis. "
            "In rare cases aspirin induces cardiotoxicity in compromised myocardium."
        )
        mock_client = self._create_mock_client("aspirin cardiotoxicity", abstract)
        service = LiteratureAssistantService(
            client=mock_client,
            enable_biomedical_enhancements=False,
        )

        result = service.search_and_rank("aspirin cardiotoxicity", max_articles=1, top_k_sentences=5)
        for s in result.ranked_sentences:
            assert s.conjunction_bonus == 0.0
            assert pytest.approx(s.relevance_score, abs=1e-5) == s.cosine_score

    def test_provenance_and_duplicate_suppression(self):
        abstract = (
            "Aspirin cardiotoxicity was assessed in clinical cohorts. "
            "Aspirin cardiotoxicity was assessed in clinical cohorts. "
            "Biomarkers showed no significant elevation."
        )
        mock_client = self._create_mock_client("aspirin cardiotoxicity", abstract)
        service = LiteratureAssistantService(
            client=mock_client,
            enable_biomedical_enhancements=True,
        )

        result = service.search_and_rank("aspirin cardiotoxicity", max_articles=1, top_k_sentences=5)
        matching = [s for s in result.ranked_sentences if "assessed in clinical cohorts" in s.sentence]
        assert len(matching) == 1
        for s in result.ranked_sentences:
            assert s.pmid == "12345678"
            assert s.article_title == "Cardiovascular pharmacology study"
            assert s.year == "2024"
            assert "12345678" in s.source_url
