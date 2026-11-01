"""Semantic literature retrieval and sentence ranking engine (Module M09)."""

import os
import re
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.literature.pubmed_client import PubMedClient, PubMedArticle
from src.literature.direction import classify_claim_polarity, PolarityAnnotation, POLARITY_UNCERTAIN

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
    cosine_score: Optional[float] = None
    conjunction_bonus: Optional[float] = 0.0
    claim_polarity: Optional[str] = None
    polarity_explanation: Optional[str] = None

    def __post_init__(self):
        if self.conjunction_bonus is None:
            object.__setattr__(self, "conjunction_bonus", 0.0)
        if self.cosine_score is None:
            object.__setattr__(
                self,
                "cosine_score",
                round(float(self.relevance_score) - float(self.conjunction_bonus), 4),
            )
        if self.claim_polarity is None:
            ann = classify_claim_polarity(self.sentence)
            object.__setattr__(self, "claim_polarity", ann.label)
            if self.polarity_explanation is None:
                object.__setattr__(self, "polarity_explanation", ann.explanation)

    @property
    def polarity(self) -> str:
        """Alias for claim_polarity supporting multiple nomenclature conventions."""
        return self.claim_polarity or POLARITY_UNCERTAIN

    @property
    def evidence_direction(self) -> str:
        """Alias for claim_polarity supporting directionality convention."""
        return self.claim_polarity or POLARITY_UNCERTAIN


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


def normalize_sentence_key(text: str) -> str:
    """
    Generate a normalized lookup key for sentence redundancy detection.

    Collapses whitespace, converts to lowercase, normalizes typographical
    quotes, apostrophes, dashes, and strips trailing punctuation.
    """
    norm = re.sub(r"\s+", " ", text).strip().lower()
    norm = norm.replace("’", "'").replace("‘", "'").replace("`", "'")
    norm = norm.replace("“", '"').replace("”", '"')
    norm = norm.replace("–", "-").replace("—", "-")
    norm = norm.replace("\ufffd", "'")
    norm = norm.rstrip(".!?:;")
    return norm


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


def reconstruct_annotated_sentences(
    articles: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    strict: bool = True,
    strict_hash_check: Optional[bool] = None,
    return_audit_summary: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Reconstruct fully populated relevance annotations from hydrated PubMed benchmark articles.

    Merges excerpt-free annotation records (v2 schema) with locally hydrated abstracts
    via (pmid, article_sentence_index), verifying each reconstructed sentence against
    sentence_hash_sha256.

    Fail-closed validation enforces 6 strict integrity requirements:
    1. An annotation references a PMID that is present in the hydrated corpus.
    2. The referenced article has a usable, non-empty abstract that splits into sentences.
    3. article_sentence_index is an integer >= 0 and strictly within abstract sentence bounds.
    4. sentence_hash_sha256 is present, valid, and a 64-character hexadecimal SHA-256 digest.
    5. The reconstructed normalized sentence hash matches sentence_hash_sha256 exactly.
    6. Hydrated corpus contains no duplicate or ambiguous article identifiers (PMIDs).

    Parameters
    ----------
    articles : List[Dict[str, Any]]
        List of hydrated PubMed articles containing 'pmid' and 'abstract'.
    annotations : List[Dict[str, Any]]
        List of v2 annotation records containing 'pmid', 'article_sentence_index', and 'sentence_hash_sha256'.
    strict : bool, default True
        If True, raises an exception on ANY integrity failure or ambiguity.
    strict_hash_check : Optional[bool], default None
        Backward-compatibility alias. If provided, overrides strict.
    return_audit_summary : bool, default False
        If True or if strict is False, returns a tuple of (reconstructed_annotations, audit_summary).

    Returns
    -------
    Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]
        List of reconstructed annotations, or (reconstructed, audit_summary).

    Raises
    ------
    KeyError
        If an annotation references a PMID absent from the hydrated corpus.
    IndexError
        If article_sentence_index is missing, negative, invalid, or out of range.
    ValueError
        If abstract is missing/empty, sentence_hash_sha256 is missing/malformed/mismatched,
        or duplicate PMIDs are detected in the corpus.
    """
    is_strict = strict if strict_hash_check is None else (strict and strict_hash_check)

    # 1. Check for duplicate or ambiguous article identifiers
    articles_by_pmid: Dict[str, Dict[str, Any]] = {}
    duplicate_pmids: Set[str] = set()
    for art in articles:
        pmid = str(art.get("pmid", "")).strip() if art.get("pmid") is not None else ""
        if not pmid:
            continue
        if pmid in articles_by_pmid:
            duplicate_pmids.add(pmid)
        articles_by_pmid[pmid] = art

    if duplicate_pmids and is_strict:
        raise ValueError(
            f"Duplicate or ambiguous article identifiers detected in hydrated corpus: PMIDs {sorted(list(duplicate_pmids))}. "
            "Reconstruction refused in strict mode to prevent ambiguous sentence mapping."
        )

    reconstructed: List[Dict[str, Any]] = []
    audit_failures: List[Dict[str, Any]] = []
    missing_pmid_count = 0
    missing_abstract_count = 0
    invalid_index_count = 0
    malformed_hash_count = 0
    hash_mismatch_count = 0

    for ann in annotations:
        sid = ann.get("sentence_id", "UNKNOWN_SID")
        pmid = str(ann.get("pmid", "")).strip() if ann.get("pmid") is not None else ""

        # Condition 1: Missing PMID in hydrated corpus
        if not pmid or pmid not in articles_by_pmid:
            missing_pmid_count += 1
            err_msg = f"Annotation {sid} references PMID '{pmid}' not found in hydrated articles."
            if is_strict:
                raise KeyError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "MISSING_PMID", "message": err_msg})
            continue

        art = articles_by_pmid[pmid]
        abstract = art.get("abstract", "")

        # Condition 2: Article has no usable abstract
        if not abstract or not str(abstract).strip():
            missing_abstract_count += 1
            err_msg = f"PMID {pmid} (referenced by {sid}) has missing or empty abstract in hydrated articles."
            if is_strict:
                raise ValueError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "EMPTY_ABSTRACT", "message": err_msg})
            continue

        sentences = split_into_sentences(str(abstract))
        if not sentences:
            missing_abstract_count += 1
            err_msg = f"PMID {pmid} (referenced by {sid}) abstract yielded 0 usable sentences after splitting."
            if is_strict:
                raise ValueError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "NO_SENTENCES", "message": err_msg})
            continue

        # Condition 3: article_sentence_index is missing, invalid, negative, or outside sentence range
        sent_idx = ann.get("article_sentence_index")
        if sent_idx is None or type(sent_idx) is not int:
            invalid_index_count += 1
            err_msg = f"Annotation {sid} has invalid or non-integer article_sentence_index: {sent_idx!r}."
            if is_strict:
                raise IndexError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "INVALID_INDEX", "message": err_msg})
            continue

        if sent_idx < 0:
            invalid_index_count += 1
            err_msg = f"Annotation {sid} specifies negative article_sentence_index: {sent_idx}."
            if is_strict:
                raise IndexError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "NEGATIVE_INDEX", "message": err_msg})
            continue

        if sent_idx >= len(sentences):
            invalid_index_count += 1
            err_msg = (
                f"Annotation {sid} specifies article_sentence_index {sent_idx}, "
                f"but PMID {pmid} abstract only contains {len(sentences)} sentences."
            )
            if is_strict:
                raise IndexError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "INDEX_OUT_OF_RANGE", "message": err_msg})
            continue

        # Condition 4: sentence_hash_sha256 missing, malformed, or not a valid SHA-256 hexadecimal digest
        raw_hash = ann.get("sentence_hash_sha256")
        if not raw_hash or not isinstance(raw_hash, str):
            malformed_hash_count += 1
            err_msg = f"Annotation {sid} is missing required 'sentence_hash_sha256' cryptographic digest."
            if is_strict:
                raise ValueError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "MISSING_HASH", "message": err_msg})
            continue

        expected_hash = raw_hash.strip().lower()
        if len(expected_hash) != 64 or not all(c in "0123456789abcdef" for c in expected_hash):
            malformed_hash_count += 1
            err_msg = (
                f"Annotation {sid} contains malformed 'sentence_hash_sha256': {raw_hash!r}. "
                "Must be a 64-character hexadecimal SHA-256 digest."
            )
            if is_strict:
                raise ValueError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "MALFORMED_HASH", "message": err_msg})
            continue

        sentence_text = sentences[sent_idx]
        norm_key = normalize_sentence_key(sentence_text)

        # Condition 5: Reconstructed sentence hash does not match expected hash
        actual_hash = hashlib.sha256(norm_key.encode("utf-8")).hexdigest().lower()
        if actual_hash != expected_hash:
            hash_mismatch_count += 1
            err_msg = (
                f"Annotation {sid} (PMID {pmid}, index {sent_idx}) hash mismatch! "
                f"Expected {expected_hash}, got {actual_hash}. Abstract text may have drifted upstream."
            )
            if is_strict:
                raise ValueError(err_msg)
            audit_failures.append({"sentence_id": sid, "pmid": pmid, "error_type": "HASH_MISMATCH", "message": err_msg})
            continue

        rec_record = dict(ann)
        rec_record["sentence_text"] = sentence_text
        rec_record["normalized_key"] = norm_key
        rec_record["article_title"] = art.get("title", "")
        reconstructed.append(rec_record)

    # Build audit summary
    is_complete = (
        len(reconstructed) == len(annotations)
        and len(audit_failures) == 0
        and len(duplicate_pmids) == 0
    )
    audit_summary: Dict[str, Any] = {
        "strict_mode": is_strict,
        "total_annotations": len(annotations),
        "reconstructed_count": len(reconstructed),
        "failures_count": len(audit_failures),
        "is_complete_benchmark": is_complete,
        "benchmark_status": "COMPLETE_VERIFIED" if is_complete else "INCOMPLETE_FAILED",
        "duplicate_pmids": sorted(list(duplicate_pmids)),
        "missing_pmid_count": missing_pmid_count,
        "missing_abstract_count": missing_abstract_count,
        "invalid_index_count": invalid_index_count,
        "malformed_hash_count": malformed_hash_count,
        "hash_mismatch_count": hash_mismatch_count,
        "failures": audit_failures,
    }

    if not is_strict or return_audit_summary:
        return reconstructed, audit_summary

    return reconstructed



# Curated clinical toxicology synonym dictionary (Phase D.4.1)
# Normalized with word boundary protection to prevent false positive substring collisions
BIOMEDICAL_SYNONYM_MAPPINGS = [
    # Paracetamol / Acetaminophen / APAP
    (
        re.compile(r"\b(paracetamol|acetaminophen|apap)\b", re.IGNORECASE),
        "paracetamol acetaminophen",
    ),
    # Hepatotoxicity / Liver injury / Hepatic damage / DILI / AILI
    (
        re.compile(
            r"\b(hepatotoxicity|hepatotoxic|liver\s+injury|acute\s+liver\s+injury|liver\s+damage|hepatic\s+damage|hepatic\s+injury|dili|aili)\b",
            re.IGNORECASE,
        ),
        "hepatotoxicity liver injury",
    ),
    # Aspirin / Acetylsalicylic acid / ASA
    (
        re.compile(r"\b(aspirin|acetylsalicylic\s+acid|asa)\b", re.IGNORECASE),
        "aspirin acetylsalicylic",
    ),
    # Cardiotoxicity / Cardiovascular toxicity / Cardiac toxicity
    (
        re.compile(
            r"\b(cardiotoxicity|cardiotoxic|cardiovascular\s+toxicity|cardiac\s+toxicity|cardiovascular\s+toxic)\b",
            re.IGNORECASE,
        ),
        "cardiotoxicity cardiovascular toxicity",
    ),
    # Lactic acidosis / MALA / Hyperlactatemia
    (
        re.compile(
            r"\b(lactic\s+acidosis|mala|hyperlactatemia|metformin-associated\s+lactic\s+acidosis)\b",
            re.IGNORECASE,
        ),
        "lactic acidosis hyperlactatemia",
    ),
]

# Narrowly scoped clinical morphology normalization (Phase D.4.3)
BIOMEDICAL_MORPHOLOGY_MAPPINGS = [
    # -toxic / -toxicity
    (re.compile(r"\b(\w+)?toxic\b", re.IGNORECASE), r"\1toxicity"),
    # -acidemic / -acidemia / -acidosis
    (re.compile(r"\b(\w+)?acidemi[ac]\b", re.IGNORECASE), r"\1acidosis"),
    # -necrotic / -necrotizing / -necrosis
    (re.compile(r"\b(\w+)?necrot(ic|izing)\b", re.IGNORECASE), r"\1necrosis"),
]

# Concept registry for conjunction detection (Phase D.4.2)
CURATED_DRUG_PATTERNS = {
    "metformin": re.compile(r"\b(metformin|glucophage)\b", re.IGNORECASE),
    "aspirin": re.compile(
        r"\b(aspirin|acetylsalicylic\s+acid)\b|(?:\basa\b(?!\s*(?:physical\s+status|status|score|class|grade|classification)))",
        re.IGNORECASE,
    ),
    "paracetamol": re.compile(r"\b(paracetamol|acetaminophen|apap)\b", re.IGNORECASE),
    "atorvastatin": re.compile(r"\b(atorvastatin|lipitor)\b", re.IGNORECASE),
    "caffeine": re.compile(r"\b(caffeine|1,3,7-trimethylxanthine)\b", re.IGNORECASE),
    "ponatinib": re.compile(r"\b(ponatinib|iclusig)\b", re.IGNORECASE),
    "tirzepatide": re.compile(r"\b(tirzepatide)\b", re.IGNORECASE),
}

CURATED_ENDPOINT_PATTERNS = {
    "lactic_acidosis": re.compile(
        r"\b(lactic\s+acidosis|lactate|acidemia|acidosis|mala|hyperlactatemia)\b",
        re.IGNORECASE,
    ),
    "cardiotoxicity": re.compile(
        r"\b(cardiotox\w*|cardiovascular\s+tox\w*|cardiac\s+tox\w*|heart\s+failure|myocardial\s+infarct\w*|thromb\w*)\b",
        re.IGNORECASE,
    ),
    "hepatotoxicity": re.compile(
        r"\b(hepatotox\w*|liver\s+injur\w*|liver\s+damage|hepatic\s+damage|hepatic\s+injur\w*|dili|aili|acute\s+liver\s+failure)\b",
        re.IGNORECASE,
    ),
}


def apply_biomedical_synonyms(text: str) -> str:
    """Apply curated biomedical synonym normalization for search representation."""
    norm = text
    for pattern, replacement in BIOMEDICAL_SYNONYM_MAPPINGS:
        norm = pattern.sub(replacement, norm)
    return norm


def apply_biomedical_morphology(text: str) -> str:
    """Apply narrowly scoped clinical morphological normalization."""
    norm = text
    for pattern, replacement in BIOMEDICAL_MORPHOLOGY_MAPPINGS:
        norm = pattern.sub(replacement, norm)
    return norm


def detect_concept_conjunction(query: str, sentence_text: str) -> Tuple[bool, bool, bool]:
    """
    Detect if query contains recognized drug and adverse endpoint concepts, and whether
    candidate sentence contains textual evidence for both concept groups.

    Conservative rule (Phase D.4.2):
    - Conjunction bonus is awarded ONLY when BOTH a recognized chemical entity AND
      a recognized adverse endpoint are independently identified in the query AND
      co-occur in the sentence.
    - Query tokens are never blindly bisected into ad-hoc concept groups.
    - If either concept cannot be safely established, returns (False, False, False).

    Returns (has_drug_evidence, has_endpoint_evidence, has_both_conjunction).
    """
    matched_drug_pat = None
    for name, pat in CURATED_DRUG_PATTERNS.items():
        if pat.search(query):
            matched_drug_pat = pat
            break

    matched_endpoint_pat = None
    for name, pat in CURATED_ENDPOINT_PATTERNS.items():
        if pat.search(query):
            matched_endpoint_pat = pat
            break

    # Strictly require both recognized concepts; never guess or bisect uncurated queries
    if not matched_drug_pat or not matched_endpoint_pat:
        return False, False, False

    has_drug = bool(matched_drug_pat.search(sentence_text))
    has_endpoint = bool(matched_endpoint_pat.search(sentence_text))
    has_conjunction = has_drug and has_endpoint
    return has_drug, has_endpoint, has_conjunction


class LiteratureAssistantService:
    """
    Retrieves scientific literature and ranks individual sentences by semantic relevance.
    """

    def __init__(
        self,
        client: Optional[PubMedClient] = None,
        enable_biomedical_enhancements: Optional[bool] = None,
        conjunction_bonus_weight: float = 0.05,
    ):
        self.client = client or PubMedClient()
        if enable_biomedical_enhancements is None:
            # Check environment variable, defaulting to False to preserve frozen baseline evaluation runner
            env_val = os.environ.get("MOLECULEIQ_LITERATURE_ENHANCEMENTS", "0").strip().lower()
            self.enable_biomedical_enhancements = env_val in ("1", "true", "yes")
        else:
            self.enable_biomedical_enhancements = bool(enable_biomedical_enhancements)
        self.conjunction_bonus_weight = conjunction_bonus_weight
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
            api_err = getattr(self.client, "last_error", None)
            if api_err:
                return LiteratureSearchResult(
                    query=clean_query,
                    articles_retrieved=0,
                    total_sentences_analyzed=0,
                    ranked_sentences=[],
                    engine_used="NCBI E-utilities",
                    status="API_ERROR",
                    disclaimer=LITERATURE_DISCLAIMER,
                    error_message=f"NCBI E-utilities request failed: {api_err}. Please verify network connectivity or retry.",
                )
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

        # Decompose abstracts into sentence items with provenance, suppressing intra-article duplicates
        sentence_pool: List[Tuple[str, PubMedArticle]] = []
        for art in articles:
            sentences = split_into_sentences(art.abstract)
            art_seen_keys = set()
            for s in sentences:
                norm_k = normalize_sentence_key(s)
                if norm_k not in art_seen_keys:
                    art_seen_keys.add(norm_k)
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
            final_scores = np.copy(sims)
            bonus_values = np.zeros_like(sims)
        elif self.enable_biomedical_enhancements:
            engine_name = "Biomedical TF-IDF Conjunction Engine"
            # Apply curated morphology and synonym normalization for search representations (original texts remain untouched)
            search_query = apply_biomedical_morphology(clean_query)
            search_query = apply_biomedical_synonyms(search_query)

            search_texts = [apply_biomedical_morphology(t) for t in texts]
            search_texts = [apply_biomedical_synonyms(t) for t in search_texts]

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform([search_query] + search_texts)
            query_vec = tfidf_matrix[0:1]
            sent_vecs = tfidf_matrix[1:]
            sims = cosine_similarity(query_vec, sent_vecs)[0]

            # Compute concept conjunction bonuses for candidate sentences
            final_scores = np.copy(sims)
            bonus_values = np.zeros_like(sims)
            for idx, (raw_sent, _) in enumerate(sentence_pool):
                if sims[idx] > 0.0:  # Only bonus sentences with verified lexical relevance
                    _, _, has_conjunction = detect_concept_conjunction(clean_query, raw_sent)
                    if has_conjunction:
                        bonus = self.conjunction_bonus_weight
                        bonus_values[idx] = bonus
                        final_scores[idx] += bonus
        else:
            engine_name = "TF-IDF Semantic Cosine Engine"
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform([clean_query] + texts)
            query_vec = tfidf_matrix[0:1]
            sent_vecs = tfidf_matrix[1:]
            sims = cosine_similarity(query_vec, sent_vecs)[0]
            final_scores = np.copy(sims)
            bonus_values = np.zeros_like(sims)

        # Rank in descending order of final score, suppressing genuinely redundant statements
        sorted_indices = np.argsort(final_scores)[::-1]

        ranked: List[RankedSentence] = []
        seen_keys = set()
        for idx in sorted_indices:
            score = float(final_scores[idx])
            cos_score = float(sims[idx])
            bonus_score = float(bonus_values[idx])
            # Exclude orthogonal sentences with zero term overlap
            if score <= 0.0 or cos_score <= 0.0:
                continue
            sent_text, article = sentence_pool[idx]
            norm_key = normalize_sentence_key(sent_text)
            if norm_key in seen_keys:
                continue
            seen_keys.add(norm_key)

            polarity_ann = classify_claim_polarity(sent_text, query=clean_query)

            ranked.append(
                RankedSentence(
                    sentence=sent_text,
                    relevance_score=round(score, 4),
                    pmid=article.pmid,
                    article_title=article.title,
                    authors=article.authors,
                    year=article.year,
                    source_url=article.url,
                    cosine_score=round(cos_score, 4),
                    conjunction_bonus=round(bonus_score, 4),
                    claim_polarity=polarity_ann.label,
                    polarity_explanation=polarity_ann.explanation,
                )
            )
            if len(ranked) >= top_k_sentences:
                break

        # Handle case where all analyzed sentences had zero similarity to query
        if not ranked:
            return LiteratureSearchResult(
                query=clean_query,
                articles_retrieved=len(articles),
                total_sentences_analyzed=len(texts),
                ranked_sentences=[],
                engine_used=engine_name,
                status="NO_RELEVANT_STATEMENTS",
                disclaimer=LITERATURE_DISCLAIMER,
                error_message=f"No abstract sentences exhibited lexical overlap with query '{clean_query}'. Try broader terms or synonyms.",
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
