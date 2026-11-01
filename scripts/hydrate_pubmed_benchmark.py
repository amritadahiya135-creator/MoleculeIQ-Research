#!/usr/bin/env python3
"""
MoleculeIQ — Authentic PubMed Benchmark Hydration Utility (Phase 2E).

Hydrates the PubMed benchmark dataset (pubmed_benchmark_v1.json) from the
redistributable manifest (pubmed_benchmark_manifest.json) by retrieving
authentic abstract texts directly from NCBI Entrez E-utilities (efetch).

This architecture allows the MoleculeIQ open-source repository to distribute
query structures, PMIDs, citation metadata, and human relevance annotations
without permanently bundling third-party commercial journal abstract text.

Usage:
    python scripts/hydrate_pubmed_benchmark.py [--force] [--verify-only]
"""

import os
import sys
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.literature.pubmed_client import (
    PubMedClient,
    NCBI_EFETCH_URL,
    create_secure_pubmed_session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hydrate_pubmed_benchmark")

MANIFEST_PATH = REPO_ROOT / "data" / "benchmarks" / "literature" / "pubmed_benchmark_manifest.json"
TARGET_BENCHMARK_PATH = REPO_ROOT / "data" / "benchmarks" / "literature" / "pubmed_benchmark_v1.json"


def verify_benchmark_file(path: Path) -> bool:
    """Verify that a target pubmed_benchmark_v1.json file exists and is fully populated."""
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = data.get("articles", [])
        if len(articles) != 18:
            return False
        for art in articles:
            abstract = art.get("abstract", "")
            if not abstract or len(abstract) < 200:
                return False
        return True
    except Exception as exc:
        logger.debug(f"Verification failed on {path}: {exc}")
        return False


def fetch_abstracts_from_ncbi(pmids: List[str], client: Optional[PubMedClient] = None) -> Dict[str, str]:
    """
    Fetch authentic abstracts from NCBI E-utilities via a secure batch request.

    Parameters
    ----------
    pmids : List[str]
        List of PubMed IDs to fetch.
    client : Optional[PubMedClient]
        Configured PubMed client instance.

    Returns
    -------
    Dict[str, str]
        Mapping of PMID -> full abstract text.
    """
    client = client or PubMedClient()
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    logger.info(f"Submitting batch EFetch request for {len(pmids)} PMIDs to NCBI...")
    resp = client._get(NCBI_EFETCH_URL, fetch_params)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    abstracts_by_pmid: Dict[str, str] = {}

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//MedlineCitation/PMID")
        if pmid_elem is None or not pmid_elem.text:
            continue
        pmid = pmid_elem.text.strip()

        # Locate abstract elements across standard and fallback XML tags
        abstract_elems = article_elem.findall(".//MedlineCitation/Article/Abstract/AbstractText")
        if not abstract_elems:
            abstract_elems = article_elem.findall(".//Abstract/AbstractText")
        if not abstract_elems:
            abstract_elems = article_elem.findall(".//OtherAbstract/AbstractText")

        parts = []
        for elem in abstract_elems:
            txt = "".join(elem.itertext()).strip()
            if txt:
                parts.append(txt)
        full_abstract = " ".join(parts).strip()
        abstracts_by_pmid[pmid] = full_abstract

    logger.info(f"Successfully retrieved and parsed {len(abstracts_by_pmid)} abstracts from NCBI.")
    return abstracts_by_pmid


def hydrate_benchmark(
    manifest_path: Path = MANIFEST_PATH,
    target_path: Path = TARGET_BENCHMARK_PATH,
    force: bool = False,
) -> bool:
    """
    Hydrate the local benchmark JSON by merging manifest metadata with fetched abstracts.
    """
    if not manifest_path.exists():
        logger.error(f"Benchmark manifest not found at: {manifest_path}")
        return False

    if target_path.exists() and not force:
        if verify_benchmark_file(target_path):
            logger.info(f"Target benchmark '{target_path.name}' is already hydrated and verified.")
            return True
        logger.warning(f"Target benchmark '{target_path.name}' exists but has missing or incomplete abstracts. Rehydrating...")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    articles = manifest.get("articles", [])
    pmids = [art["pmid"] for art in articles if "pmid" in art]
    if not pmids:
        logger.error("No PMIDs found in manifest.")
        return False

    try:
        abstracts = fetch_abstracts_from_ncbi(pmids)
    except Exception as exc:
        logger.error(f"Failed to fetch abstracts from NCBI: {exc}")
        return False

    hydrated_articles = []
    missing_pmids = []
    for art in articles:
        pmid = art["pmid"]
        abstract_text = abstracts.get(pmid, "")
        if not abstract_text:
            missing_pmids.append(pmid)
        art_copy = dict(art)
        art_copy["abstract"] = abstract_text
        art_copy["abstract_character_length"] = len(abstract_text)
        art_copy["has_complete_abstract"] = bool(abstract_text and len(abstract_text) > 200)
        hydrated_articles.append(art_copy)

    if missing_pmids:
        logger.error(f"Failed to retrieve abstracts for {len(missing_pmids)} PMIDs: {missing_pmids}")
        return False

    hydrated_data = {
        "schema_version": manifest.get("schema_version", "1.0.0"),
        "dataset_name": "MoleculeIQ Authentic PubMed Benchmark",
        "created_at_utc": manifest.get("created_at_utc", ""),
        "total_queries": manifest.get("total_queries", 3),
        "total_articles": len(hydrated_articles),
        "unique_pmids": len(set(pmids)),
        "queries": manifest.get("queries", []),
        "articles": hydrated_articles,
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(hydrated_data, f, indent=2)

    logger.info(f"Hydrated benchmark saved to {target_path} (18 articles, complete abstract texts).")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Hydrate authentic PubMed benchmark data from NCBI Entrez E-utilities."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rehydration even if target benchmark file already exists.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check whether the benchmark is already hydrated and valid without network calls.",
    )
    args = parser.parse_args()

    if args.verify_only:
        is_valid = verify_benchmark_file(TARGET_BENCHMARK_PATH)
        if is_valid:
            logger.info("Verification PASSED: Target PubMed benchmark file is hydrated and valid.")
            sys.exit(0)
        else:
            logger.error("Verification FAILED: Target PubMed benchmark file is missing or incomplete.")
            sys.exit(1)

    success = hydrate_benchmark(force=args.force)
    if success:
        logger.info("Hydration workflow completed successfully.")
        sys.exit(0)
    else:
        logger.error("Hydration workflow failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
