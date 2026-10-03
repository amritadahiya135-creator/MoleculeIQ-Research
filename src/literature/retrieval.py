"""Semantic literature retrieval and sentence ranking engine (Module M09)."""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.literature.pubmed_client import PubMedClient, PubMedArticle

logger = logging.getLogger(__name__)

LITERATURE_DISCLAIMER = (
    "LITERATURE NOTICE: Retrieved statements represent reported findings from published PubMed "
    "abstracts ranked by textual and semantic relevance to the query. Semantic ranking does not "
    "independently confirm experimental validity, biological causality, or clinical consensus."
)


@dataclass(frozen=True)
class RankedSentence:
    """Individual scientific statement ranked by relevance with full source provenance."""
    sentence: str
    relevance_score: float
    pmid: str
    article_title: str
    authors: str
    year: str
    source_url: str


@dataclass(frozen=True)
class LiteratureSearchResult:
    """Aggregate outcome of a literature search and ranking session."""
    query: str
    articles_retrieved: int
    total_sentences_analyzed: int
    ranked_sentences: List[RankedSentence]
    engine_used: str
    status: str
    disclaimer: str
    error_message: Optional[str] = None


def split_into_sentences(text: str) -> List[str]:
    """
    Split biomedical abstract text into clean, coherent sentences.
    """
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []
    # Split on sentence terminals followed by space and uppercase/number
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_text)
    # Filter out trivial fragments (e.g. abbreviations, section headers)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 25]


class LiteratureAssistantService:
    """
    Retrieves scientific literature and ranks individual sentences by semantic relevance.
    """

    def __init__(self, client: Optional[PubMedClient] = None):
        self.client = client or PubMedClient()
        self._transformer_model = None
        self._has_transformer = False

        # Attempt to detect sentence_transformers if present
        try:
            from sentence_transformers import SentenceTransformer
            self._transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._has_transformer = True
            logger.info("Neural SentenceTransformer (all-MiniLM-L6-v2) initialized.")
        except Exception:
            logger.info("SentenceTransformer not available; using scikit-learn TF-IDF semantic cosine engine.")

    def search_and_rank(
        self,
        query: str,
        max_articles: int = 5,
        top_k_sentences: int = 8,
    ) -> LiteratureSearchResult:
        """
        Execute literature search, sentence segmentation, and semantic relevance ranking.

        Parameters
        ----------
        query : str
            Target research query (e.g. "aspirin hepatotoxicity" or chemical name).
        max_articles : int, default=5
            Number of PubMed articles to fetch.
        top_k_sentences : int, default=8
            Number of highest-ranking sentences to present.

        Returns
        -------
        LiteratureSearchResult
            Ranked sentence excerpts with article provenance.
        """
        clean_query = query.strip()
        if not clean_query:
            return LiteratureSearchResult(
                query=query,
                articles_retrieved=0,
                total_sentences_analyzed=0,
                ranked_sentences=[],
                engine_used="None",
                status="EMPTY_QUERY",
                disclaimer=LITERATURE_DISCLAIMER,
                error_message="Search query is empty.",
            )

        articles = self.client.search_articles(clean_query, max_results=max_articles)
        if not articles:
            return LiteratureSearchResult(
                query=clean_query,
                articles_retrieved=0,
                total_sentences_analyzed=0,
                ranked_sentences=[],
                engine_used="NCBI E-utilities",
                status="NO_ARTICLES_FOUND",
                disclaimer=LITERATURE_DISCLAIMER,
                error_message=f"No PubMed publications found for query '{clean_query}'.",
            )

        # Decompose abstracts into sentence items with provenance
        sentence_pool: List[Tuple[str, PubMedArticle]] = []
        for art in articles:
            sentences = split_into_sentences(art.abstract)
            for s in sentences:
                sentence_pool.append((s, art))

        if not sentence_pool:
            return LiteratureSearchResult(
                query=clean_query,
                articles_retrieved=len(articles),
                total_sentences_analyzed=0,
                ranked_sentences=[],
                engine_used="NCBI E-utilities",
                status="NO_ABSTRACTS_AVAILABLE",
                disclaimer=LITERATURE_DISCLAIMER,
                error_message="Retrieved articles do not contain extractable abstract text.",
            )

        texts = [s[0] for s in sentence_pool]

        # Compute semantic similarities
        if self._has_transformer and self._transformer_model is not None:
            engine_name = "Neural SentenceTransformer (all-MiniLM-L6-v2)"
            query_emb = self._transformer_model.encode([clean_query])
            sent_embs = self._transformer_model.encode(texts)
            sims = cosine_similarity(query_emb, sent_embs)[0]
        else:
            engine_name = "TF-IDF Semantic Cosine Engine"
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform([clean_query] + texts)
            query_vec = tfidf_matrix[0:1]
            sent_vecs = tfidf_matrix[1:]
            sims = cosine_similarity(query_vec, sent_vecs)[0]

        # Rank in descending order of similarity
        sorted_indices = np.argsort(sims)[::-1][:top_k_sentences]

        ranked: List[RankedSentence] = []
        for idx in sorted_indices:
            score = float(sims[idx])
            sent_text, article = sentence_pool[idx]
            ranked.append(
                RankedSentence(
                    sentence=sent_text,
                    relevance_score=round(score, 4),
                    pmid=article.pmid,
                    article_title=article.title,
                    authors=article.authors,
                    year=article.year,
                    source_url=article.url,
                )
            )

        return LiteratureSearchResult(
            query=clean_query,
            articles_retrieved=len(articles),
            total_sentences_analyzed=len(texts),
            ranked_sentences=ranked,
            engine_used=engine_name,
            status="SUCCESS",
            disclaimer=LITERATURE_DISCLAIMER,
            error_message=None,
        )
