"""PubMed literature retrieval and semantic sentence ranking engine."""

from src.literature.pubmed_client import PubMedClient, PubMedArticle
from src.literature.retrieval import (
    LiteratureAssistantService,
    LiteratureSearchResult,
    RankedSentence,
    LITERATURE_DISCLAIMER,
)
from src.literature.direction import (
    classify_claim_polarity,
    PolarityAnnotation,
    POLARITY_AFFIRMATIVE,
    POLARITY_NEGATED,
    POLARITY_UNCERTAIN,
    POLARITY_LABELS,
    POLARITY_UI_LABELS,
)

__all__ = [
    "PubMedClient",
    "PubMedArticle",
    "LiteratureAssistantService",
    "LiteratureSearchResult",
    "RankedSentence",
    "LITERATURE_DISCLAIMER",
    "classify_claim_polarity",
    "PolarityAnnotation",
    "POLARITY_AFFIRMATIVE",
    "POLARITY_NEGATED",
    "POLARITY_UNCERTAIN",
    "POLARITY_LABELS",
    "POLARITY_UI_LABELS",
]
