"""Integration tests for PubMed literature retrieval and semantic sentence ranking (Module M09)."""

import pytest
from src.literature.pubmed_client import PubMedArticle
from src.literature.retrieval import (
    LiteratureAssistantService,
    LiteratureSearchResult,
    RankedSentence,
    split_into_sentences,
    LITERATURE_DISCLAIMER,
)


class TestLiteratureEngine:
    """Tests for Module M09: Literature Retrieval and Semantic Ranking."""

    def test_sentence_segmentation(self):
        abstract = (
            "Aspirin is a nonsteroidal anti-inflammatory drug. It inhibits cyclooxygenase-1 and 2. "
            "High doses may induce gastrointestinal bleeding or hepatic injury in rare cases."
        )
        sentences = split_into_sentences(abstract)
        assert len(sentences) == 3
        assert "Aspirin is a nonsteroidal" in sentences[0]
        assert "It inhibits cyclooxygenase-1" in sentences[1]

    def test_empty_query_handling(self):
        service = LiteratureAssistantService()
        result = service.search_and_rank("   ")
        assert isinstance(result, LiteratureSearchResult)
        assert result.status == "EMPTY_QUERY"
        assert len(result.ranked_sentences) == 0

    def test_live_or_mock_retrieval_and_ranking(self):
        service = LiteratureAssistantService()
        result = service.search_and_rank("aspirin cardiotoxicity", max_articles=2, top_k_sentences=3)

        assert isinstance(result, LiteratureSearchResult)
        assert result.disclaimer == LITERATURE_DISCLAIMER

        # If network connected, verify rank ordering and source URLs
        if result.status == "SUCCESS":
            assert len(result.ranked_sentences) > 0
            top_sent = result.ranked_sentences[0]
            assert isinstance(top_sent, RankedSentence)
            assert top_sent.source_url.startswith("https://pubmed.ncbi.nlm.nih.gov/")
            assert 0.0 <= top_sent.relevance_score <= 1.0

            # Verify descending order of relevance scores
            scores = [s.relevance_score for s in result.ranked_sentences]
            assert scores == sorted(scores, reverse=True)
