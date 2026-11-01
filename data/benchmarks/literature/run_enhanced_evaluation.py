"""
MoleculeIQ — Literature Retrieval Enhanced Evaluation Runner (Phase D.4).

Executes MoleculeIQ's enhanced biomedical TF-IDF ranking pipeline (curated synonym
normalization, concept conjunction bonus, and clinical morphology rules) on the
frozen authentic PubMed benchmark (pubmed_benchmark_v1) without network dependencies.
Computes MRR, Precision@5, Precision@k, Recall@5, nDCG@5, zero-overlap filter rate,
and duplicate return rate across both macro and micro aggregations.
Saves results to data/benchmarks/literature/enhanced_evaluation_results.json.
"""

import os
import sys
import json
import math
from typing import List, Dict, Any, Optional
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.literature.pubmed_client import PubMedArticle
from src.literature.retrieval import (
    LiteratureAssistantService,
    split_into_sentences,
    normalize_sentence_key,
    reconstruct_annotated_sentences,
)


class FrozenBenchmarkPubMedClient:
    """Mock PubMed client that serves articles strictly from the frozen benchmark JSON."""

    def __init__(self, benchmark_path: str):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_data = json.load(f)
        self.articles_by_query: Dict[str, List[PubMedArticle]] = {}
        for art in self.benchmark_data["articles"]:
            q_text = art["query_text"].strip().lower()
            article_obj = PubMedArticle(
                pmid=art["pmid"],
                title=art["title"],
                abstract=art["abstract"],
                authors=art["authors"],
                year=art["year"],
                url=art["url"],
            )
            self.articles_by_query.setdefault(q_text, []).append(article_obj)
        self.last_error = None

    def search_articles(self, query: str, max_results: int = 6) -> List[PubMedArticle]:
        q_clean = query.strip().lower()
        articles = self.articles_by_query.get(q_clean, [])
        return articles[:max_results]


def compute_dcg(grades: List[int], k: int = 5, use_exponential: bool = True) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, g in enumerate(grades[:k]):
        rank = i + 1
        gain = (2.0 ** g - 1.0) if use_exponential else float(g)
        discount = math.log2(rank + 1.0)
        dcg += gain / discount
    return dcg


def compute_idcg(pool_grades: List[int], k: int = 5, use_exponential: bool = True) -> float:
    """Compute Ideal Discounted Cumulative Gain at rank k by sorting pool grades descending."""
    sorted_grades = sorted(pool_grades, reverse=True)
    return compute_dcg(sorted_grades, k=k, use_exponential=use_exponential)


def run_enhanced_evaluation(
    benchmark_file: str = "data/benchmarks/literature/pubmed_benchmark_v1.json",
    annotations_file: str = "data/benchmarks/literature/relevance_annotations_v2.json",
    output_report_file: str = "data/benchmarks/literature/enhanced_evaluation_results.json",
    conjunction_bonus_weight: float = 0.05,
) -> Dict[str, Any]:
    """Execute complete deterministic evaluation of enhanced TF-IDF ranking pipeline."""
    # Load frozen benchmark articles
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Load frozen annotations
    with open(annotations_file, "r", encoding="utf-8") as f:
        annot_data = json.load(f)

    raw_annotations = annot_data["annotations"]
    expected_count = annot_data.get("total_annotated_sentences", len(raw_annotations))
    # Reconstruct sentences if v2 schema or missing normalized_key
    if annot_data.get("schema_version", "1.0.0").startswith("2.") or (raw_annotations and "normalized_key" not in raw_annotations[0]):
        annotations_list = reconstruct_annotated_sentences(
            articles=benchmark_data.get("articles", []),
            annotations=raw_annotations,
            strict=True,
        )
    else:
        annotations_list = raw_annotations

    if len(annotations_list) != expected_count:
        raise RuntimeError(
            f"Benchmark reconstruction failure: expected {expected_count} annotated sentences, "
            f"but successfully reconstructed only {len(annotations_list)}. "
            "Refusing to compute benchmark metrics on incomplete data."
        )

    # Build lookup by (query_id, normalized_sentence_key)
    annotations_lookup: Dict[str, Dict[str, Any]] = {}
    pool_by_query: Dict[str, List[Dict[str, Any]]] = {}
    for ann in annotations_list:
        qid = ann["query_id"]
        norm_k = ann["normalized_key"]
        annotations_lookup[(qid, norm_k)] = ann
        pool_by_query.setdefault(qid, []).append(ann)

    # Initialize service with offline frozen client and enhanced features enabled
    mock_client = FrozenBenchmarkPubMedClient(benchmark_file)
    service = LiteratureAssistantService(
        client=mock_client,
        enable_biomedical_enhancements=True,
        conjunction_bonus_weight=conjunction_bonus_weight,
    )

    queries_config = [
        {"query_id": "Q1_METFORMIN_LACTIC_ACIDOSIS", "query_text": "metformin lactic acidosis", "max_articles": 6},
        {"query_id": "Q2_ASPIRIN_CARDIOTOXICITY", "query_text": "aspirin cardiotoxicity", "max_articles": 6},
        {"query_id": "Q3_PARACETAMOL_HEPATOTOXICITY", "query_text": "paracetamol hepatotoxicity", "max_articles": 6},
    ]

    query_results = []
    all_top5_binary = []
    all_top5_grades = []

    for qcfg in queries_config:
        qid = qcfg["query_id"]
        qtext = qcfg["query_text"]
        pool = pool_by_query[qid]
        pool_grades = [ann["gold_label"] for ann in pool]
        total_pool_relevant = sum(1 for ann in pool if ann["binary_label"] == 1)

        # Run enhanced search and rank requesting top 8 sentences
        result = service.search_and_rank(qtext, max_articles=qcfg["max_articles"], top_k_sentences=8)

        returned_ranked = result.ranked_sentences
        top5 = returned_ranked[:5]

        # Calculate metrics for top 5
        top5_grades = []
        top5_binary = []
        top5_details = []
        seen_keys_top5 = set()
        duplicate_count_top5 = 0

        for r_idx, sent_obj in enumerate(top5):
            norm_k = normalize_sentence_key(sent_obj.sentence)
            if norm_k in seen_keys_top5:
                duplicate_count_top5 += 1
            seen_keys_top5.add(norm_k)

            ann = annotations_lookup.get((qid, norm_k))
            if ann is None:
                # Fallback matching if minor normalization differences occur
                for p_item in pool:
                    if p_item["pmid"] == sent_obj.pmid and p_item["normalized_key"] == norm_k:
                        ann = p_item
                        break
            grade = ann["gold_label"] if ann else 0
            binary = ann["binary_label"] if ann else 0
            rationale = ann["rationale"] if ann else "Unannotated"
            top5_grades.append(grade)
            top5_binary.append(binary)

            top5_details.append({
                "rank": r_idx + 1,
                "pmid": sent_obj.pmid,
                "relevance_score": sent_obj.relevance_score,
                "cosine_score": sent_obj.cosine_score,
                "conjunction_bonus": sent_obj.conjunction_bonus,
                "gold_grade": grade,
                "binary_label": binary,
                "sentence": sent_obj.sentence,
                "rationale": rationale,
            })

        # Reciprocal Rank (first rank with binary == 1)
        rr = 0.0
        for i, b in enumerate(top5_binary):
            if b == 1:
                rr = 1.0 / (i + 1)
                break

        # Precision@5 (fixed cutoff: denominator = 5)
        p_at_5_fixed = sum(top5_binary) / 5.0

        # Precision@k (variable denominator: denominator = min(5, returned))
        k_returned = min(5, len(top5))
        p_at_k_variable = (sum(top5_binary) / float(k_returned)) if k_returned > 0 else 0.0

        # Recall@5
        recall_at_5 = (sum(top5_binary) / float(total_pool_relevant)) if total_pool_relevant > 0 else 0.0

        # nDCG@5 (exponential gain)
        dcg5_exp = compute_dcg(top5_grades, k=5, use_exponential=True)
        idcg5_exp = compute_idcg(pool_grades, k=5, use_exponential=True)
        ndcg5_exp = (dcg5_exp / idcg5_exp) if idcg5_exp > 0 else 0.0

        # nDCG@5 (linear gain)
        dcg5_lin = compute_dcg(top5_grades, k=5, use_exponential=False)
        idcg5_lin = compute_idcg(pool_grades, k=5, use_exponential=False)
        ndcg5_lin = (dcg5_lin / idcg5_lin) if idcg5_lin > 0 else 0.0

        # Duplicate rate in top-5
        duplicate_rate_top5 = duplicate_count_top5 / 5.0

        # Zero-overlap filter rate in pool
        zero_overlap_rate = (len(pool) - len(returned_ranked)) / float(len(pool)) if len(pool) > 0 else 0.0

        q_report = {
            "query_id": qid,
            "query_text": qtext,
            "engine_used": result.engine_used,
            "status": result.status,
            "pool_total_sentences": len(pool),
            "pool_relevant_sentences": total_pool_relevant,
            "pool_relevance_prevalence": round(total_pool_relevant / float(len(pool)), 4),
            "returned_total_sentences": len(returned_ranked),
            "top5_metrics": {
                "reciprocal_rank": round(rr, 4),
                "precision_at_5_fixed": round(p_at_5_fixed, 4),
                "precision_at_k_variable": round(p_at_k_variable, 4),
                "recall_at_5": round(recall_at_5, 4),
                "ndcg_at_5_exponential": round(ndcg5_exp, 4),
                "ndcg_at_5_linear": round(ndcg5_lin, 4),
                "duplicate_rate": round(duplicate_rate_top5, 4),
            },
            "zero_overlap_filter_rate": round(zero_overlap_rate, 4),
            "top5_sentences": top5_details,
        }
        query_results.append(q_report)
        all_top5_binary.extend(top5_binary)
        all_top5_grades.extend(top5_grades)

    # Compute Macro-averages
    macro_mrr = float(np.mean([q["top5_metrics"]["reciprocal_rank"] for q in query_results]))
    macro_p5_fixed = float(np.mean([q["top5_metrics"]["precision_at_5_fixed"] for q in query_results]))
    macro_p5_variable = float(np.mean([q["top5_metrics"]["precision_at_k_variable"] for q in query_results]))
    macro_recall5 = float(np.mean([q["top5_metrics"]["recall_at_5"] for q in query_results]))
    macro_ndcg5_exp = float(np.mean([q["top5_metrics"]["ndcg_at_5_exponential"] for q in query_results]))
    macro_ndcg5_lin = float(np.mean([q["top5_metrics"]["ndcg_at_5_linear"] for q in query_results]))
    macro_zero_overlap = float(np.mean([q["zero_overlap_filter_rate"] for q in query_results]))
    macro_duplicate = float(np.mean([q["top5_metrics"]["duplicate_rate"] for q in query_results]))

    # Compute Micro-averages (pooled over the 15 returned top-5 sentences)
    micro_p5_fixed = float(np.mean(all_top5_binary))
    total_relevant_all_pools = sum(q["pool_relevant_sentences"] for q in query_results)
    micro_recall5 = float(sum(all_top5_binary) / float(total_relevant_all_pools))

    final_report = {
        "benchmark_name": "MoleculeIQ Authentic PubMed Benchmark Enhanced (Phase D.4)",
        "evaluation_timestamp_utc": "2026-10-19T14:30:00Z",
        "ranking_engine": "scikit-learn TF-IDF Semantic Cosine Engine with Curated Synonym Expansion, Biomedical Morphology, and Concept Conjunction Bonus",
        "enhancement_configuration": {
            "curated_synonyms": True,
            "biomedical_morphology": True,
            "concept_conjunction_bonus": True,
            "conjunction_bonus_weight": conjunction_bonus_weight,
        },
        "network_access": "OFFLINE_FROZEN_BENCHMARK",
        "total_queries": len(query_results),
        "total_pool_sentences": sum(q["pool_total_sentences"] for q in query_results),
        "total_pool_relevant": total_relevant_all_pools,
        "aggregate_metrics": {
            "macro_averages": {
                "mrr": round(macro_mrr, 4),
                "precision_at_5_fixed": round(macro_p5_fixed, 4),
                "precision_at_k_variable": round(macro_p5_variable, 4),
                "recall_at_5": round(macro_recall5, 4),
                "ndcg_at_5_exponential": round(macro_ndcg5_exp, 4),
                "ndcg_at_5_linear": round(macro_ndcg5_lin, 4),
                "zero_overlap_filter_rate": round(macro_zero_overlap, 4),
                "duplicate_return_rate": round(macro_duplicate, 4),
            },
            "micro_averages": {
                "precision_at_5_pooled": round(micro_p5_fixed, 4),
                "recall_at_5_pooled": round(micro_recall5, 4),
            },
        },
        "query_breakdown": query_results,
    }

    # Save to json
    os.makedirs(os.path.dirname(output_report_file), exist_ok=True)
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    return final_report


if __name__ == "__main__":
    report = run_enhanced_evaluation()
    print("=" * 60)
    print("MOLECULEIQ — PHASE D.4 ENHANCED EVALUATION REPORT")
    print("=" * 60)
    print(f"Ranking Engine: {report['ranking_engine']}")
    print(f"Enhancements: {report['enhancement_configuration']}")
    print("-" * 60)
    print("MACRO-AVERAGE METRICS:")
    for k, v in report["aggregate_metrics"]["macro_averages"].items():
        print(f"  {k:28s}: {v}")
    print("-" * 60)
    print("MICRO-AVERAGE METRICS:")
    for k, v in report["aggregate_metrics"]["micro_averages"].items():
        print(f"  {k:28s}: {v}")
    print("-" * 60)
    print("QUERY BREAKDOWN:")
    for q in report["query_breakdown"]:
        m = q["top5_metrics"]
        print(f"[{q['query_id']}] '{q['query_text']}'")
        print(f"  Pool: {q['pool_total_sentences']} sents (Rel: {q['pool_relevant_sentences']}) | Returned: {q['returned_total_sentences']}")
        print(f"  MRR={m['reciprocal_rank']:.4f} | Fixed P@5={m['precision_at_5_fixed']:.4f} | Recall@5={m['recall_at_5']:.4f} | nDCG@5={m['ndcg_at_5_exponential']:.4f}")
        for s in q["top5_sentences"]:
            print(f"    Rank {s['rank']} [Grade {s['gold_grade']} (Score {s['relevance_score']:.4f} = Cos {s['cosine_score']:.4f} + Conj {s['conjunction_bonus']:.4f})]: {s['sentence'][:80]}...")
    print("=" * 60)
