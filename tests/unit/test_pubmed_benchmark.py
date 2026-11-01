"""
Unit tests for authentic PubMed benchmark dataset integrity and evaluation runner.

Verifies:
1. Benchmark JSON schema, completeness, unique PMIDs, and valid PubMed URLs.
2. Relevance annotations schema, 4-tier grade bounds [0, 3], binary mapping consistency.
3. Provenance linkage between annotations and benchmark articles.
4. Determinism and mathematical correctness of evaluation metrics (MRR, P@5, Recall, nDCG).
"""

import json
import os
import pytest
import numpy as np

from data.benchmarks.literature.run_baseline_evaluation import (
    FrozenBenchmarkPubMedClient,
    compute_dcg,
    compute_idcg,
    run_baseline_evaluation,
)
from data.benchmarks.literature.run_enhanced_evaluation import run_enhanced_evaluation
from src.literature.retrieval import (
    split_into_sentences,
    normalize_sentence_key,
    reconstruct_annotated_sentences,
)

BENCHMARK_PATH = "data/benchmarks/literature/pubmed_benchmark_v1.json"
ANNOTATIONS_PATH = "data/benchmarks/literature/relevance_annotations_v2.json"


@pytest.fixture(scope="module")
def benchmark_data():
    assert os.path.exists(BENCHMARK_PATH), f"Missing benchmark file: {BENCHMARK_PATH}"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def annotations_data():
    assert os.path.exists(ANNOTATIONS_PATH), f"Missing annotations file: {ANNOTATIONS_PATH}"
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestPubMedBenchmarkIntegrity:
    """Validate authentic PubMed benchmark dataset integrity."""

    def test_benchmark_schema_and_counts(self, benchmark_data):
        assert benchmark_data["schema_version"] == "1.0.0"
        assert benchmark_data["total_queries"] == 3
        assert benchmark_data["total_articles"] == 18
        assert benchmark_data["unique_pmids"] == 18

        articles = benchmark_data["articles"]
        assert len(articles) == 18

        # 6 articles per query
        q_counts = {}
        for art in articles:
            q_counts[art["query_id"]] = q_counts.get(art["query_id"], 0) + 1
        assert q_counts == {
            "Q1_METFORMIN_LACTIC_ACIDOSIS": 6,
            "Q2_ASPIRIN_CARDIOTOXICITY": 6,
            "Q3_PARACETAMOL_HEPATOTOXICITY": 6,
        }

    def test_article_fields_and_authenticity(self, benchmark_data):
        pmids = set()
        for art in benchmark_data["articles"]:
            pmid = art["pmid"]
            assert pmid.isdigit() and len(pmid) >= 7, f"Invalid PMID: {pmid}"
            assert pmid not in pmids, f"Duplicate PMID in benchmark: {pmid}"
            pmids.add(pmid)

            assert art["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            assert len(art["title"]) > 10
            assert len(art["abstract"]) > 500, f"Abstract too short for PMID {pmid}"
            assert art["has_complete_abstract"] is True
            assert int(art["year"]) >= 2024, f"Unexpected year {art['year']} for PMID {pmid}"


class TestRelevanceAnnotationsIntegrity:
    """Validate relevance annotations schema and grading consistency (v2 Excerpt-Free)."""

    def test_annotations_schema_and_distribution(self, annotations_data):
        assert annotations_data["schema_version"] == "2.0.0"
        assert annotations_data["total_annotated_sentences"] == 198

        ann_list = annotations_data["annotations"]
        assert len(ann_list) == 198

        dist = annotations_data["label_distribution_total"]
        assert dist["grade_3_core"] == 43
        assert dist["grade_2_contextual"] == 56
        assert dist["grade_1_background"] == 47
        assert dist["grade_0_irrelevant"] == 52
        assert dist["binary_relevant_total"] == 99
        assert dist["binary_non_relevant_total"] == 99

    def test_annotation_consistency_rules(self, annotations_data, benchmark_data):
        benchmark_pmids = {art["pmid"] for art in benchmark_data["articles"]}
        seen_sids = set()

        for ann in annotations_data["annotations"]:
            sid = ann["sentence_id"]
            assert sid not in seen_sids, f"Duplicate sentence_id: {sid}"
            seen_sids.add(sid)

            assert ann["pmid"] in benchmark_pmids, f"Annotation PMID {ann['pmid']} not in benchmark"
            assert ann["gold_label"] in (0, 1, 2, 3), f"Invalid gold grade {ann['gold_label']}"

            # Binary label definition: 1 if grade >= 2, else 0
            expected_binary = 1 if ann["gold_label"] >= 2 else 0
            assert ann["binary_label"] == expected_binary, f"Binary label mismatch in {sid}"

            # Verify no verbatim text in public v2 records
            assert "sentence_text" not in ann, f"Forbidden verbatim sentence_text in {sid}"
            assert "normalized_key" not in ann, f"Forbidden verbatim normalized_key in {sid}"
            assert len(ann["sentence_hash_sha256"]) == 64, f"Invalid SHA-256 hash in {sid}"
            assert isinstance(ann["article_sentence_index"], int) and ann["article_sentence_index"] >= 0

            assert len(ann["rationale"]) > 10
            assert ann["status"] == "AUDITED"


class TestPubMedExcerptFreeReconstruction:
    """Validate deterministic in-memory reconstruction from hydrated PubMed abstracts."""

    def test_no_verbatim_excerpts_in_public_v2_file(self, annotations_data):
        for ann in annotations_data["annotations"]:
            assert "sentence_text" not in ann
            assert "normalized_key" not in ann
            assert "abstract" not in ann
            assert "sentence" not in ann

    def test_successful_reconstruction_all_198_annotations(self, benchmark_data, annotations_data):
        reconstructed = reconstruct_annotated_sentences(
            articles=benchmark_data["articles"],
            annotations=annotations_data["annotations"],
            strict_hash_check=True,
        )
        assert len(reconstructed) == 198
        for rec in reconstructed:
            assert "sentence_text" in rec and len(rec["sentence_text"]) > 20
            assert "normalized_key" in rec and len(rec["normalized_key"]) > 20
            assert rec["article_title"] is not None

    def test_strict_reconstruction_rejects_missing_pmid(self, benchmark_data, annotations_data):
        broken_annotations = [dict(annotations_data["annotations"][0])]
        broken_annotations[0]["pmid"] = "999999999"  # Non-existent PMID
        with pytest.raises(KeyError, match="not found in hydrated articles"):
            reconstruct_annotated_sentences(
                articles=benchmark_data["articles"],
                annotations=broken_annotations,
                strict=True,
            )

    def test_strict_reconstruction_rejects_missing_abstract(self, benchmark_data, annotations_data):
        broken_articles = [dict(art) for art in benchmark_data["articles"]]
        broken_articles[0]["abstract"] = "   "  # Whitespace only / missing
        with pytest.raises(ValueError, match="missing or empty abstract"):
            reconstruct_annotated_sentences(
                articles=broken_articles,
                annotations=annotations_data["annotations"][:1],
                strict=True,
            )

    def test_strict_reconstruction_rejects_negative_index(self, benchmark_data, annotations_data):
        broken_annotations = [dict(annotations_data["annotations"][0])]
        broken_annotations[0]["article_sentence_index"] = -1  # Negative index
        with pytest.raises(IndexError, match="negative article_sentence_index"):
            reconstruct_annotated_sentences(
                articles=benchmark_data["articles"],
                annotations=broken_annotations,
                strict=True,
            )

    def test_strict_reconstruction_rejects_out_of_range_index(self, benchmark_data, annotations_data):
        broken_annotations = [dict(annotations_data["annotations"][0])]
        broken_annotations[0]["article_sentence_index"] = 999  # Out of range
        with pytest.raises(IndexError, match="specifies article_sentence_index 999"):
            reconstruct_annotated_sentences(
                articles=benchmark_data["articles"],
                annotations=broken_annotations,
                strict=True,
            )

    def test_strict_reconstruction_rejects_missing_hash(self, benchmark_data, annotations_data):
        broken_annotations = [dict(annotations_data["annotations"][0])]
        del broken_annotations[0]["sentence_hash_sha256"]  # Missing hash
        with pytest.raises(ValueError, match="missing required 'sentence_hash_sha256'"):
            reconstruct_annotated_sentences(
                articles=benchmark_data["articles"],
                annotations=broken_annotations,
                strict=True,
            )

    def test_strict_reconstruction_rejects_malformed_hash(self, benchmark_data, annotations_data):
        broken_annotations = [dict(annotations_data["annotations"][0])]
        broken_annotations[0]["sentence_hash_sha256"] = "malformed_short_hash"  # Non-64 hex
        with pytest.raises(ValueError, match="malformed 'sentence_hash_sha256'"):
            reconstruct_annotated_sentences(
                articles=benchmark_data["articles"],
                annotations=broken_annotations,
                strict=True,
            )

    def test_strict_reconstruction_rejects_incorrect_hash(self, benchmark_data, annotations_data):
        broken_annotations = [dict(annotations_data["annotations"][0])]
        broken_annotations[0]["sentence_hash_sha256"] = "f" * 64  # Incorrect hash
        with pytest.raises(ValueError, match="hash mismatch"):
            reconstruct_annotated_sentences(
                articles=benchmark_data["articles"],
                annotations=broken_annotations,
                strict=True,
            )

    def test_strict_reconstruction_rejects_duplicate_pmid_ambiguity(self, benchmark_data, annotations_data):
        # Create duplicate articles with identical PMID in hydrated corpus
        dup_articles = list(benchmark_data["articles"]) + [dict(benchmark_data["articles"][0])]
        with pytest.raises(ValueError, match="Duplicate or ambiguous article identifiers"):
            reconstruct_annotated_sentences(
                articles=dup_articles,
                annotations=annotations_data["annotations"][:1],
                strict=True,
            )

    def test_strict_reconstruction_accepts_fully_valid_annotation(self, benchmark_data, annotations_data):
        valid = [dict(annotations_data["annotations"][0])]
        reconstructed = reconstruct_annotated_sentences(
            articles=benchmark_data["articles"],
            annotations=valid,
            strict=True,
        )
        assert len(reconstructed) == 1
        assert "sentence_text" in reconstructed[0] and len(reconstructed[0]["sentence_text"]) > 20
        assert "normalized_key" in reconstructed[0] and len(reconstructed[0]["normalized_key"]) > 20
        assert reconstructed[0]["article_title"] is not None

    def test_non_strict_mode_returns_complete_audit_summary_and_reports_incomplete(self, benchmark_data, annotations_data):
        broken = [dict(annotations_data["annotations"][0])]
        broken[0]["pmid"] = "999999999"
        reconstructed, summary = reconstruct_annotated_sentences(
            articles=benchmark_data["articles"],
            annotations=broken,
            strict=False,
        )
        assert len(reconstructed) == 0
        assert summary["is_complete_benchmark"] is False
        assert summary["benchmark_status"] == "INCOMPLETE_FAILED"
        assert summary["missing_pmid_count"] == 1
        assert len(summary["failures"]) == 1

    def test_baseline_evaluation_refuses_to_report_on_reconstruction_failure(self, tmp_path):
        bad_annot_file = tmp_path / "bad_annotations.json"
        bad_data = {
            "schema_version": "2.0.0",
            "total_annotated_sentences": 1,
            "annotations": [
                {
                    "sentence_id": "Q1_S01",
                    "query_id": "Q1_METFORMIN_LACTIC_ACIDOSIS",
                    "pmid": "999999999",
                    "article_sentence_index": 0,
                    "sentence_hash_sha256": "0" * 64,
                    "gold_label": 3,
                    "binary_label": 1,
                    "relevance_tier": "HIGHLY_RELEVANT",
                    "rationale": "Independent clinical test rationale.",
                    "status": "AUDITED",
                }
            ],
        }
        with open(bad_annot_file, "w", encoding="utf-8") as f:
            json.dump(bad_data, f)

        out_report = tmp_path / "baseline_out.json"
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            run_baseline_evaluation(
                benchmark_file=BENCHMARK_PATH,
                annotations_file=str(bad_annot_file),
                output_report_file=str(out_report),
            )
        assert not out_report.exists()

    def test_enhanced_evaluation_refuses_to_report_on_reconstruction_failure(self, tmp_path):
        bad_annot_file = tmp_path / "bad_annotations.json"
        bad_data = {
            "schema_version": "2.0.0",
            "total_annotated_sentences": 1,
            "annotations": [
                {
                    "sentence_id": "Q1_S01",
                    "query_id": "Q1_METFORMIN_LACTIC_ACIDOSIS",
                    "pmid": "999999999",
                    "article_sentence_index": 0,
                    "sentence_hash_sha256": "0" * 64,
                    "gold_label": 3,
                    "binary_label": 1,
                    "relevance_tier": "HIGHLY_RELEVANT",
                    "rationale": "Independent clinical test rationale.",
                    "status": "AUDITED",
                }
            ],
        }
        with open(bad_annot_file, "w", encoding="utf-8") as f:
            json.dump(bad_data, f)

        out_report = tmp_path / "enhanced_out.json"
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            run_enhanced_evaluation(
                benchmark_file=BENCHMARK_PATH,
                annotations_file=str(bad_annot_file),
                output_report_file=str(out_report),
            )
        assert not out_report.exists()


class TestEvaluationMetricsMath:
    """Verify mathematical correctness of IR metric calculations."""

    def test_dcg_calculation_known_fixture(self):
        # Grades: [3, 2, 0]
        # Rank 1: (2^3 - 1) / log2(2) = 7 / 1 = 7.0
        # Rank 2: (2^2 - 1) / log2(3) = 3 / 1.5849625 = 1.892789
        # Rank 3: (2^0 - 1) / log2(4) = 0 / 2 = 0.0
        grades = [3, 2, 0]
        dcg = compute_dcg(grades, k=3, use_exponential=True)
        expected = 7.0 + (3.0 / np.log2(3.0))
        assert pytest.approx(dcg, rel=1e-5) == expected

    def test_idcg_and_ndcg_bounds(self):
        pool = [1, 0, 3, 2, 3, 0]
        idcg = compute_idcg(pool, k=3, use_exponential=True)
        # Ideal order for top 3: [3, 3, 2]
        expected_idcg = (7.0 / np.log2(2.0)) + (7.0 / np.log2(3.0)) + (3.0 / np.log2(4.0))
        assert pytest.approx(idcg, rel=1e-5) == expected_idcg

        # Worst possible ranking [0, 0, 1]
        worst_dcg = compute_dcg([0, 0, 1], k=3, use_exponential=True)
        ndcg_worst = worst_dcg / idcg
        assert 0.0 <= ndcg_worst <= 1.0


class TestBaselineEvaluationExecution:
    """Verify execution determinism and baseline performance reproducibility."""

    def test_deterministic_baseline_run(self):
        report = run_baseline_evaluation()
        macro = report["aggregate_metrics"]["macro_averages"]

        # Exact reproducible baseline metrics
        assert report["total_queries"] == 3
        assert report["total_pool_sentences"] == 198
        assert report["total_pool_relevant"] == 99

        assert macro["mrr"] == 1.0
        assert pytest.approx(macro["precision_at_5_fixed"], abs=1e-4) == 0.8667
        assert pytest.approx(macro["precision_at_k_variable"], abs=1e-4) == 0.8667
        assert pytest.approx(macro["recall_at_5"], abs=1e-4) == 0.1500
        assert pytest.approx(macro["ndcg_at_5_exponential"], abs=1e-4) == 0.7904
        assert macro["duplicate_return_rate"] == 0.0
