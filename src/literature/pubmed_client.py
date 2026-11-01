import os
import ssl
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
import urllib3

logger = logging.getLogger(__name__)

NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class SecureSSLContextAdapter(HTTPAdapter):
    """
    HTTPAdapter that injects a customized SSLContext equipped with
    operating system native root certificates and custom enterprise CA bundles.
    """

    def __init__(self, ssl_context: ssl.SSLContext, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)


def create_secure_pubmed_session() -> requests.Session:
    """
    Construct a requests Session with comprehensive SSL certificate validation.

    Loads certificates from:
      1. Default OpenSSL / certifi trust store.
      2. Windows native root and CA stores (if running on Windows).
      3. Custom CA bundle paths defined in REQUESTS_CA_BUNDLE or SSL_CERT_FILE.
    """
    session = requests.Session()
    ctx = ssl.create_default_context()

    # Load native Windows root certificates if available
    if hasattr(ssl, "enum_certificates"):
        pem_certs = []
        for store_name in ("ROOT", "CA"):
            try:
                for cert_der, encoding_type, trust in ssl.enum_certificates(store_name):
                    try:
                        pem_certs.append(ssl.DER_cert_to_PEM_cert(cert_der))
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug(f"Could not enumerate Windows cert store '{store_name}': {exc}")

        if pem_certs:
            try:
                ctx.load_verify_locations(cadata="\n".join(pem_certs))
            except Exception as exc:
                logger.warning(f"Failed to load Windows root certificates into SSL context: {exc}")

    # Load custom CA bundle if defined
    custom_ca = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if custom_ca and os.path.exists(custom_ca):
        try:
            ctx.load_verify_locations(cafile=custom_ca)
        except Exception as exc:
            logger.warning(f"Failed to load custom CA bundle from '{custom_ca}': {exc}")

    session.mount("https://", SecureSSLContextAdapter(ctx))
    return session


@dataclass(frozen=True)
class PubMedArticle:
    """Retrieved scientific publication from PubMed."""
    pmid: str
    title: str
    abstract: str
    authors: str
    year: str
    url: str


class PubMedClient:
    """
    Client for NCBI Entrez E-utilities supporting rate-limiting and secure SSL verification.
    """

    def __init__(self, timeout: int = 15, session: Optional[requests.Session] = None):
        self.timeout = timeout
        self.session = session or create_secure_pubmed_session()
        self._cache: Dict[str, List[PubMedArticle]] = {}
        self.last_error: Optional[str] = None

    def _get(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """Execute HTTP request with strict SSL verification and conditional explicit fallback."""
        try:
            return self.session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.SSLError as ssl_err:
            allow_insecure = os.getenv("MOLECULEIQ_ALLOW_INSECURE_SSL", "0").lower() in ("1", "true", "yes")
            if allow_insecure:
                logger.warning(
                    "SECURITY WARNING: SSL certificate verification failed for NCBI request. "
                    "Retrying with verify=False because MOLECULEIQ_ALLOW_INSECURE_SSL is explicitly enabled."
                )
                with urllib3.warnings.catch_warnings():
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    return requests.get(url, params=params, verify=False, timeout=self.timeout)
            logger.error(
                "SSL certificate verification failed on NCBI request. Insecure fallback is disabled. "
                "Configure a valid CA bundle via REQUESTS_CA_BUNDLE or explicitly enable "
                "MOLECULEIQ_ALLOW_INSECURE_SSL=1 if operating in an air-gapped test environment."
            )
            raise ssl_err

    def search_articles(self, query: str, max_results: int = 5) -> List[PubMedArticle]:
        """
        Search PubMed for articles matching the research query.

        Parameters
        ----------
        query : str
            Biomedical query or chemical compound name.
        max_results : int, default=5
            Maximum articles to retrieve.

        Returns
        -------
        List[PubMedArticle]
            List of parsed article records with abstracts.
        """
        self.last_error = None
        clean_query = query.strip()
        if not clean_query:
            return []

        if clean_query in self._cache:
            return self._cache[clean_query]

        # Step 1: ESearch to obtain PMIDs
        search_params = {
            "db": "pubmed",
            "term": clean_query,
            "retmode": "json",
            "retmax": max_results,
        }

        try:
            search_resp = self._get(NCBI_ESEARCH_URL, search_params)
            search_resp.raise_for_status()
            search_data = search_resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"PubMed search failed for query '{clean_query}': {exc}")
            return []

        if not id_list:
            return []

        # Step 2: EFetch to retrieve article XML and abstracts
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        }

        articles: List[PubMedArticle] = []
        try:
            fetch_resp = self._get(NCBI_EFETCH_URL, fetch_params)
            fetch_resp.raise_for_status()
            root = ET.fromstring(fetch_resp.content)

            for article_elem in root.findall(".//PubmedArticle"):
                pmid_elem = article_elem.find(".//MedlineCitation/PMID")
                pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else "Unknown"

                title_elem = article_elem.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()).strip() if title_elem is not None else "No title provided"
                if not title:
                    title = "No title provided"

                # Extract primary abstract; fallback to OtherAbstract only if primary is absent
                abstract_elems = article_elem.findall(".//MedlineCitation/Article/Abstract/AbstractText")
                if not abstract_elems:
                    abstract_elems = article_elem.findall(".//Abstract/AbstractText")
                if not abstract_elems:
                    abstract_elems = article_elem.findall(".//OtherAbstract/AbstractText")

                abstract_parts = []
                for elem in abstract_elems:
                    txt = "".join(elem.itertext()).strip()
                    if txt:
                        abstract_parts.append(txt)
                abstract = " ".join(abstract_parts)

                # Extract authors
                author_names = []
                for author_elem in article_elem.findall(".//AuthorList/Author")[:4]:
                    last = author_elem.find("LastName")
                    fore = author_elem.find("ForeName")
                    last_text = last.text.strip() if last is not None and last.text else ""
                    fore_text = fore.text.strip() if fore is not None and fore.text else ""
                    if last_text:
                        initial = fore_text[0] if fore_text else ""
                        name = f"{last_text} {initial}".strip()
                        author_names.append(name)
                authors = ", ".join(author_names) if author_names else "Unknown Authors"

                # Extract Year
                year_elem = article_elem.find(".//Journal/JournalIssue/PubDate/Year")
                if year_elem is None or not (year_elem.text and year_elem.text.strip()):
                    year_elem = article_elem.find(".//DateCompleted/Year")
                if year_elem is None or not (year_elem.text and year_elem.text.strip()):
                    year_elem = article_elem.find(".//Journal/JournalIssue/PubDate/MedlineDate")

                if year_elem is not None and year_elem.text and year_elem.text.strip():
                    year_match = re.search(r"\b(19\d\d|20\d\d)\b", year_elem.text.strip())
                    year = year_match.group(1) if year_match else year_elem.text.strip()[:4]
                else:
                    year = "Undated"

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                articles.append(
                    PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        year=year,
                        url=url,
                    )
                )

        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"PubMed fetch failed for PMIDs {id_list}: {exc}")

        self._cache[clean_query] = articles
        return articles
