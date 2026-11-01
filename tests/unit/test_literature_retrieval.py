"""Unit tests for Literature Intelligence Assistant retrieval, deduplication, and provenance (Phase D.1)."""

import pytest
from typing import List, Optional
from src.literature.pubmed_client import PubMedArticle, PubMedClient
from src.literature.retrieval import (
    LiteratureAssistantService,
    LiteratureSearchResult,
    RankedSentence,
    normalize_sentence_key,
    split_into_sentences,
    LITERATURE_DISCLAIMER,
)


class DeterministicMockPubMedClient:
    """Mock PubMedClient returning pre-configured deterministic articles without network access."""

    def __init__(
        self,
        articles: Optional[List[PubMedArticle]] = None,
        last_error: Optional[str] = None,
    ):
        self._articles = articles or []
        self.last_error = last_error

    def search_articles(self, query: str, max_results: int = 5) -> List[PubMedArticle]:
        if self.last_error:
            return []
        return self._articles[:max_results]


class TestNormalizationAndSegmentation:
    """Unit tests for text normalization and sentence splitting."""

    def test_normalize_sentence_key_collapses_whitespace(self):
        s1 = "Metformin  inhibits   mitochondrial complex I."
        s2 = "metformin inhibits mitochondrial complex i"
        assert normalize_sentence_key(s1) == normalize_sentence_key(s2)

    def test_normalize_sentence_key_typographical_quotes_and_dashes(self):
        s1 = "Metformin’s dual-edged profile—glycemic efficacy."
        s2 = "Metformin's dual-edged profile-glycemic efficacy"
        assert normalize_sentence_key(s1) == normalize_sentence_key(s2)

    def test_normalize_sentence_key_strips_trailing_punctuation(self):
        s1 = "Lactic acidosis is a rare complication."
        s2 = "Lactic acidosis is a rare complication"
        s3 = "Lactic acidosis is a rare complication!"
        assert normalize_sentence_key(s1) == normalize_sentence_key(s2)
        assert normalize_sentence_key(s1) == normalize_sentence_key(s3)

    def test_split_into_sentences_filters_trivial_fragments(self):
        abstract = "Fig. 1 shows data. Metformin was administered at 500 mg daily. Primary outcome was clinically evaluated at 12 weeks."
        sentences = split_into_sentences(abstract)
        # "Fig. 1 shows data." is <= 25 characters, so filtered out
        assert len(sentences) == 2
        assert sentences[0] == "Metformin was administered at 500 mg daily."
        assert sentences[1] == "Primary outcome was clinically evaluated at 12 weeks."


class TestDeduplicationAndRanking:
    """Deterministic tests for sentence deduplication and semantic ranking."""

    def test_exact_duplicate_sentences_same_article_suppressed(self):
        dup_text = "Metformin-associated lactic acidosis (MALA) is a rare but life-threatening complication."
        article = PubMedArticle(
            pmid="10001",
            title="Metformin Review",
            abstract=f"{dup_text} Patients present with severe metabolic acidosis. {dup_text}",
            authors="Smith J",
            year="2023",
            url="https://pubmed.ncbi.nlm.nih.gov/10001/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin lactic acidosis", top_k_sentences=5)

        assert result.status == "SUCCESS"
        texts = [s.sentence for s in result.ranked_sentences]
        # Exact duplicate sentence must only appear once
        assert texts.count(dup_text) == 1
        assert len(result.ranked_sentences) == 2

    def test_whitespace_and_capitalization_deduplication(self):
        s1 = "Metformin inhibits hepatic gluconeogenesis and mitochondrial respiration."
        s2 = "  METFORMIN INHIBITS HEPATIC GLUCONEOGENESIS AND MITOCHONDRIAL RESPIRATION.  "
        article = PubMedArticle(
            pmid="10002",
            title="Mechanisms of Metformin",
            abstract=f"{s1} Peripheral glucose uptake is enhanced in muscle. {s2}",
            authors="Doe A",
            year="2024",
            url="https://pubmed.ncbi.nlm.nih.gov/10002/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin gluconeogenesis", top_k_sentences=5)

        assert result.status == "SUCCESS"
        # Only 1 instance of the gluconeogenesis sentence should be present
        matching = [s for s in result.ranked_sentences if "gluconeogenesis" in s.sentence.lower()]
        assert len(matching) == 1

    def test_similar_meaningfully_different_sentences_preserved(self):
        s1 = "Metformin decreases hepatic glucose production in type 2 diabetes."
        s2 = "Metformin increases peripheral insulin sensitivity in skeletal muscle."
        article = PubMedArticle(
            pmid="10003",
            title="Metformin Actions",
            abstract=f"{s1} {s2} Fasting blood glucose was significantly reduced.",
            authors="Lee C",
            year="2022",
            url="https://pubmed.ncbi.nlm.nih.gov/10003/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin glucose", top_k_sentences=5)

        assert result.status == "SUCCESS"
        texts = [s.sentence for s in result.ranked_sentences]
        assert s1 in texts
        assert s2 in texts

    def test_distinct_statements_same_pmid_preserved(self):
        s1 = "Massive metformin overdose results in extreme lactic acidosis and acute kidney injury."
        s2 = "Continuous renal replacement therapy was initiated with rapid clearance."
        article = PubMedArticle(
            pmid="10004",
            title="Case Report of Overdose",
            abstract=f"{s1} {s2}",
            authors="Kim D, Park S",
            year="2024",
            url="https://pubmed.ncbi.nlm.nih.gov/10004/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin lactic acidosis therapy", top_k_sentences=5)

        assert result.status == "SUCCESS"
        pmids = [s.pmid for s in result.ranked_sentences]
        assert pmids == ["10004", "10004"]
        assert len(result.ranked_sentences) == 2

    def test_duplicate_across_different_articles_suppressed(self):
        shared_sentence = "Metformin is the first-line oral antihyperglycemic medication for type 2 diabetes mellitus."
        a1 = PubMedArticle(
            pmid="20001",
            title="Study One",
            abstract=f"{shared_sentence} Primary outcome was HbA1c reduction at 24 weeks.",
            authors="Alpha A",
            year="2021",
            url="https://pubmed.ncbi.nlm.nih.gov/20001/",
        )
        a2 = PubMedArticle(
            pmid="20002",
            title="Study Two",
            abstract=f"{shared_sentence} Adverse events were primarily gastrointestinal.",
            authors="Beta B",
            year="2022",
            url="https://pubmed.ncbi.nlm.nih.gov/20002/",
        )
        client = DeterministicMockPubMedClient(articles=[a1, a2])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin first-line diabetes", top_k_sentences=5)

        assert result.status == "SUCCESS"
        texts = [s.sentence for s in result.ranked_sentences]
        assert texts.count(shared_sentence) == 1

    def test_stable_source_attribution_after_deduplication(self):
        sent = "Metformin inhibits respiratory complex I in hepatocytes."
        article = PubMedArticle(
            pmid="30001",
            title="Mitochondrial Inhibition by Biguanides",
            abstract=f"{sent} ATP production was reduced by forty percent.",
            authors="Miller T, Zhang Y",
            year="2023",
            url="https://pubmed.ncbi.nlm.nih.gov/30001/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin complex I", top_k_sentences=2)

        assert result.status == "SUCCESS"
        first = result.ranked_sentences[0]
        assert first.pmid == "30001"
        assert first.article_title == "Mitochondrial Inhibition by Biguanides"
        assert first.authors == "Miller T, Zhang Y"
        assert first.year == "2023"
        assert first.source_url == "https://pubmed.ncbi.nlm.nih.gov/30001/"


class TestEdgeCasesAndErrorHandling:
    """Deterministic tests for error modes, missing metadata, and empty states."""

    def test_missing_metadata_handled_gracefully(self):
        article = PubMedArticle(
            pmid="Unknown",
            title="No title provided",
            abstract="Metformin was evaluated in pre-clinical mouse models of diabetes.",
            authors="Unknown Authors",
            year="Undated",
            url="https://pubmed.ncbi.nlm.nih.gov/Unknown/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin mouse models", top_k_sentences=2)

        assert result.status == "SUCCESS"
        assert len(result.ranked_sentences) == 1
        s = result.ranked_sentences[0]
        assert s.pmid == "Unknown"
        assert s.article_title == "No title provided"
        assert s.authors == "Unknown Authors"
        assert s.year == "Undated"

    def test_empty_pubmed_results(self):
        client = DeterministicMockPubMedClient(articles=[])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("nonexistent drug 999xyz")

        assert result.status == "NO_ARTICLES_FOUND"
        assert len(result.ranked_sentences) == 0
        assert "No PubMed publications found" in result.error_message

    def test_pubmed_api_retrieval_error(self):
        client = DeterministicMockPubMedClient(last_error="ConnectionError: HTTPSConnectionPool failed")
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin")

        assert result.status == "API_ERROR"
        assert len(result.ranked_sentences) == 0
        assert "NCBI E-utilities request failed" in result.error_message

    def test_no_abstracts_available(self):
        article = PubMedArticle(
            pmid="40001",
            title="Title Only Announcement",
            abstract="",
            authors="Editor E",
            year="2020",
            url="https://pubmed.ncbi.nlm.nih.gov/40001/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin")

        assert result.status == "NO_ABSTRACTS_AVAILABLE"
        assert len(result.ranked_sentences) == 0

    def test_no_relevant_statements_when_zero_term_overlap(self):
        article = PubMedArticle(
            pmid="50001",
            title="Unrelated Topic",
            abstract="Statistical analyses were executed using standard software. Participants completed surveys.",
            authors="Jones M",
            year="2021",
            url="https://pubmed.ncbi.nlm.nih.gov/50001/",
        )
        client = DeterministicMockPubMedClient(articles=[article])
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin lactic acidosis")

        assert result.status == "NO_RELEVANT_STATEMENTS"
        assert len(result.ranked_sentences) == 0


class TestDeterministicBenchmarkQueries:
    """Benchmark queries run deterministically using curated synthetic abstracts."""

    def test_deterministic_metformin_type2_diabetes(self):
        articles = [
            PubMedArticle(
                pmid="60001",
                title="Efficacy of Metformin in T2DM",
                abstract=(
                    "Metformin is the primary pharmacotherapy for type 2 diabetes mellitus. "
                    "It significantly reduces glycosylated hemoglobin without causing hypoglycemia. "
                    "Weight neutrality or modest weight loss is frequently observed."
                ),
                authors="Davies M, Kahn S",
                year="2023",
                url="https://pubmed.ncbi.nlm.nih.gov/60001/",
            ),
            PubMedArticle(
                pmid="60002",
                title="Cardiovascular Outcomes of Metformin in Diabetes",
                abstract=(
                    "Long-term metformin therapy in type 2 diabetes shows cardioprotective trends. "
                    "Mitochondrial AMPK activation mediates cellular energy homeostasis."
                ),
                authors="Holman R, Neil A",
                year="2022",
                url="https://pubmed.ncbi.nlm.nih.gov/60002/",
            ),
        ]
        client = DeterministicMockPubMedClient(articles=articles)
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin type 2 diabetes", top_k_sentences=5)

        assert result.status == "SUCCESS"
        assert result.articles_retrieved == 2
        assert len(result.ranked_sentences) > 0
        scores = [s.relevance_score for s in result.ranked_sentences]
        assert scores == sorted(scores, reverse=True)
        # All sentences must have non-zero cosine score
        assert all(s > 0.0 for s in scores)
        # Check top sentence contains key query terms
        top_sent = result.ranked_sentences[0].sentence.lower()
        assert "metformin" in top_sent and "type 2 diabetes" in top_sent

    def test_deterministic_metformin_lactic_acidosis(self):
        articles = [
            PubMedArticle(
                pmid="70001",
                title="Clinical Review of Metformin-Associated Lactic Acidosis",
                abstract=(
                    "Metformin-associated lactic acidosis is a rare but critical emergency in patients with renal failure. "
                    "Inhibition of mitochondrial respiratory chain complex I leads to excess lactate accumulation. "
                    "Mortality rates approach thirty to fifty percent in severe clinical presentations."
                ),
                authors="Lalau J, Race J",
                year="2024",
                url="https://pubmed.ncbi.nlm.nih.gov/70001/",
            ),
            PubMedArticle(
                pmid="70002",
                title="Management of Severe MALA Overdose",
                abstract=(
                    "Metformin overdose may precipitate severe high-anion-gap lactic acidosis. "
                    "Extracorporeal continuous venovenous hemodiafiltration provides effective clearance of metformin and lactate. "
                    "Metformin overdose may precipitate severe high-anion-gap lactic acidosis."  # duplicate
                ),
                authors="Calello D, Gosselin S",
                year="2023",
                url="https://pubmed.ncbi.nlm.nih.gov/70002/",
            ),
        ]
        client = DeterministicMockPubMedClient(articles=articles)
        service = LiteratureAssistantService(client=client)

        result = service.search_and_rank("metformin lactic acidosis", top_k_sentences=6)

        assert result.status == "SUCCESS"
        assert result.articles_retrieved == 2
        # Verify duplicate from article 70002 was suppressed
        mala_overdose_count = sum(
            1 for s in result.ranked_sentences
            if "metformin overdose may precipitate severe high-anion-gap lactic acidosis" in s.sentence.lower()
        )
        assert mala_overdose_count == 1

        # Scores strictly descending and positive
        scores = [s.relevance_score for s in result.ranked_sentences]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)
