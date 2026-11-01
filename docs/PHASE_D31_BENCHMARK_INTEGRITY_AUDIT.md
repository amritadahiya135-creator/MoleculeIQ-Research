# MoleculeIQ — Phase D.3.1: Independent Benchmark Integrity & Metric Audit Report

**Auditor:** Independent Biomedical Information Retrieval Evaluator & Senior Software QA Engineer<br>
**Audit Date:** 19 October 2026<br>
**Audited Baseline:** Phase D.3 Frozen Literature Benchmark (`pubmed_benchmark_v1`)<br>
**Evaluated Artifacts:**<br>
- `data/benchmarks/literature/pubmed_benchmark_v1.json`<br>
- `data/benchmarks/literature/relevance_annotations_v1.json`<br>
- `data/benchmarks/literature/EVALUATION_PROTOCOL.md`<br>
- `data/benchmarks/literature/baseline_evaluation_results.json`<br>
- `data/benchmarks/literature/checksums.sha256`<br>
- `data/benchmarks/literature/run_baseline_evaluation.py`<br>
**Protected Artifacts Integrity:** 31 / 31 SHA-256 Hashes Verified (100% INTACT)<br>
**Overall Phase D.3.1 Gate Verdict:** **CONDITIONAL PASS (Benchmark Valid & Reproducible for Internal Comparative Evaluation; Single-Annotator Ground Truth Protocol Formally Recorded)**<br>

---

## 1. Executive Summary & Audit Task Verdicts

This independent audit rigorously verified the frozen authentic PubMed benchmark, relevance annotations, evaluation protocol, and baseline ranking metrics established in Phase D.3. The audit was conducted in a strictly read-only manner without altering production code, ranking algorithms, model files, or benchmark fixtures.

### Audit Tasks Summary Table

| Audit Task | Scope | Verdict | Key Finding |
| :--- | :--- | :---: | :--- |
| **Task 1: Frozen Dataset Verification** | Completeness, PMIDs, provenance, checksums | **PASS** | 18 unique PMIDs; 100% match live NCBI records; 198 sentences match abstracts 1:1; checksums verified. |
| **Task 2: Metric Recomputation** | Independent scratch calculation of all metrics | **PASS** | 0 discrepancies found across Macro MRR, Fixed P@5, Variable P@k, Recall@5, nDCG@5, and Micro metrics. |
| **Task 3: Label Independence** | Human review vs scoring bias; rater protocol | **CONDITIONAL PASS** | Ground truth was NOT biased by TF-IDF scores (correctly labeled 0.0000-score APAP findings as Grade 3); single-annotator limitation documented. |
| **Task 4: Evaluation Validity** | Pool structure, filtering, selection bias | **PASS** | Candidate pools accurately mirror live search behavior; scope correctly classified as focused diagnostic benchmark. |
| **Task 5: Phase D.4 Criteria Review** | Target thresholds, trade-offs, regressions | **PASS** | Multi-metric, per-query gate established (Q2 nDCG $\ge 0.8500$, Q3 P@5 $\ge 0.8000$, Q1 no regression). |
| **Task 6: Repository Integrity** | Model checksums, working tree preservation | **PASS** | 31/31 protected artifacts 100% intact; zero production changes; working tree preserved. |

---

## 2. Working-Tree Status Verification

Before and throughout this audit, the Git working tree was inspected and preserved:
- **Tracked Uncommitted Files (from prior phases C.3.3 / D.1)**:
  `app.py`, `docs/KNOWN_LIMITATIONS.md`, `docs/MODEL_CARD_TOX21.md`, `src/literature/pubmed_client.py`, `src/literature/retrieval.py`, `src/models/inference.py`, `src/services/contracts.py`, `src/services/orchestration.py`, `src/ui/batch_screening.py`, `src/ui/components.py`, `src/ui/literature_assistant.py`, `src/ui/model_insights.py`, `src/ui/single_molecule.py`, `tests/e2e/test_batch_workflow.py`.
- **Untracked Directories & Files**:
  `data/benchmarks/`, `docs/PHASE_D3_PUBMED_BENCHMARK_REPORT.md`, `tests/unit/test_pubmed_benchmark.py`, calibration artifacts, diagnostic scripts.
- **Audit Actions**: No destructive commands (`git reset`, `git clean`, `checkout`, `stash`) were executed. All audit scratch scripts were isolated to the conversation scratch directory.

---

## 3. Task 1: Frozen Dataset Verification & Provenance

### 3.1 Checksum & File Byte Integrity
All 5 files listed in `data/benchmarks/literature/checksums.sha256` were independently hashed using SHA-256 and compared:
- `data/benchmarks/literature/EVALUATION_PROTOCOL.md`: `8a395a4051861587a3e46ce32c84a30ece5034f7cb74e936a113e227bc0ff4bb` (**MATCH**)
- `data/benchmarks/literature/baseline_evaluation_results.json`: `b84b6362cd5b9f17709c2a7d79d3e1ab71a7184b1b00d442e83f3c4118342bde` (**MATCH**)
- `data/benchmarks/literature/pubmed_benchmark_v1.json`: `444a730ffefae65a1596a5f291c5d86d1820d28bc19baf42ec3e51f80cf4e405` (**MATCH**)
- `data/benchmarks/literature/relevance_annotations_v1.json`: `b40b4777a9ed295d7b41523944ef8bc41209ef3557befe7b021239c7f7593ef7` (**MATCH**)
- `data/benchmarks/literature/run_baseline_evaluation.py`: `a0f32197dc5d1ddf8c3d26d43c28b044b5051d30ff9651843d6278465070cab7` (**MATCH**)

### 3.2 Dataset Structure & Partitioning
- **Total Articles**: Exactly 18 records across 3 queries (6 records per query).
- **Unique PMIDs**: Exactly 18 unique PMIDs.
- **Cross-Query Isolation**: Intersection across query PMID sets is $\emptyset$ (zero overlap; each article belongs strictly to one query).
- **Article Field Integrity**: 100% of articles possess a valid numeric PMID ($\ge 7$ digits), non-empty title, complete abstract ($>500$ chars, mean 1,480 chars), author string, publication year (2024–2026), and matching PubMed URL (`https://pubmed.ncbi.nlm.nih.gov/<pmid>/`).

### 3.3 Live NCBI E-Utilities Provenance Verification
An online query was executed against the NCBI Entrez E-utilities API (`esummary.fcgi`):
- All 18 PMIDs returned active records from the National Library of Medicine.
- Titles, publication years, and author lists in `pubmed_benchmark_v1.json` matched the official NCBI records with 100% fidelity.

### 3.4 Sentence Extraction Accuracy
The production sentence segmentation function [`split_into_sentences()`](../src/literature/retrieval.py#L63-L74) and canonical normalization function [`normalize_sentence_key()`](../src/literature/retrieval.py#L47-L60) were applied to the raw abstracts:
- Segmented candidate sentences: **198**.
- Annotated entries in `relevance_annotations_v1.json`: **198**.
- Discrepancies / unmapped sentences: **0**. Every segmented sentence maps 1:1 to an audited annotation.

### 3.5 Immutability Finding
The benchmark is "frozen" **by protocol specification, SHA-256 manifest tracking, and regression unit tests**. On the local Windows NTFS filesystem, the JSON files retain standard read-write permissions (`stat.S_IWRITE = True`). While sufficient for internal software engineering governance, researchers should note that filesystem write-protection flags were not explicitly toggled.

---

## 4. Task 2: Independent Metric Recomputation

An independent scratch auditor script (`audit_task2_recompute_metrics.py`) was implemented without importing any functions from `run_baseline_evaluation.py`. The TF-IDF pipeline and evaluation math were re-implemented from first principles using standard `scikit-learn` and `numpy`.

### 4.1 Metric Comparison Table

| Metric | Mathematical Formula | Recomputed Value | Reported Value | Absolute Difference | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Macro MRR** | $\frac{1}{3} \sum_{q=1}^3 \frac{1}{\text{rank}_{\text{first rel}}}$ | **1.0000** | **1.0000** | 0.000000 | **MATCH** |
| **Macro Fixed P@5** | $\frac{1}{3} \sum_{q=1}^3 \left(\frac{1}{5} \sum_{i=1}^5 \mathbb{I}(\text{Grade}_i \ge 2)\right)$ | **0.8667** | **0.8667** | 0.000033 | **MATCH** |
| **Macro Variable P@k** | $\frac{1}{3} \sum_{q=1}^3 \left(\frac{1}{\min(5, \|\text{Ret}\|)} \sum_{i=1}^k \mathbb{I}(\text{Grade}_i \ge 2)\right)$ | **0.8667** | **0.8667** | 0.000033 | **MATCH** |
| **Macro Recall@5** | $\frac{1}{3} \sum_{q=1}^3 \left(\frac{\text{Rel in Top 5}}{\text{Total Pool Rel}}\right)$ | **0.1500** | **0.1500** | 0.000043 | **MATCH** |
| **Macro nDCG@5 (Exp)** | $\frac{1}{3} \sum_{q=1}^3 \frac{\sum_{i=1}^5 (2^{\text{Grade}_i} - 1)/\log_2(i+1)}{\text{IDCG@5}_q}$ | **0.7904** | **0.7904** | 0.000026 | **MATCH** |
| **Macro nDCG@5 (Lin)** | $\frac{1}{3} \sum_{q=1}^3 \frac{\sum_{i=1}^5 \text{Grade}_i/\log_2(i+1)}{\text{IDCG@5}_q}$ | **0.8623** | **0.8624** | 0.000055 | **MATCH** |
| **Zero-Overlap Rate** | $\frac{1}{3} \sum_{q=1}^3 \frac{\|\{s \in \text{Pool}_q : \text{Score} \le 0.0\}\|}{\|\text{Pool}_q\|}$ | **0.8766** | **0.8766** | 0.000026 | **MATCH** |
| **Macro Duplicate Rate** | $\frac{1}{3} \sum_{q=1}^3 \frac{\text{Duplicates in Top 5}}{5}$ | **0.0000** | **0.0000** | 0.000000 | **MATCH** |
| **Micro Precision@5** | $\frac{1}{15} \sum_{j=1}^{15} \mathbb{I}(\text{Grade}_j \ge 2)$ | **0.8667** (13 / 15) | **0.8667** (13 / 15) | 0.000033 | **MATCH** |
| **Micro Recall@5** | $\frac{\text{Total Rel Retrieved in Top 5}}{\text{Total Rel in All Pools}}$ | **0.1313** (13 / 99) | **0.1313** (13 / 99) | 0.000013 | **MATCH** |

*Note: Differences of $<0.00006$ represent standard display rounding to 4 decimal places.*

### 4.2 Recall@5 Mathematical Verification
The audit specifically scrutinized the reported Recall@5 values to ensure numerators and denominators derived from compatible, judged pools:
- **Query 1 (`metformin lactic acidosis`)**:
  - Judged relevant in candidate pool: **42** sentences.
  - Relevant sentences retrieved in Top 5: **5**.
  - Query Recall@5 = $\frac{5}{42} = 0.119048... \approx 0.1190$.
- **Query 2 (`aspirin cardiotoxicity`)**:
  - Judged relevant in candidate pool: **20** sentences.
  - Relevant sentences retrieved in Top 5: **5**.
  - Query Recall@5 = $\frac{5}{20} = 0.250000... = 0.2500$.
- **Query 3 (`paracetamol hepatotoxicity`)**:
  - Judged relevant in candidate pool: **37** sentences.
  - Relevant sentences retrieved in Top 5: **3**.
  - Query Recall@5 = $\frac{3}{37} = 0.081081... \approx 0.0811$.
- **Macro Average**:
  $$\text{Macro Recall@5} = \frac{0.119048 + 0.250000 + 0.081081}{3} = \frac{0.450129}{3} = 0.150043 \approx 0.1500$$
- **Micro (Pooled) Average**:
  $$\text{Micro Recall@5} = \frac{5 + 5 + 3}{42 + 20 + 37} = \frac{13}{99} = 0.131313 \approx 0.1313$$

**Finding**: The numerators and denominators are 100% mathematically consistent and accurately reflect the judged candidate pools.

### 4.3 Top-5 Output Exact Match
Across all 15 top-5 returned slots (3 queries $\times$ 5 ranks), the recomputed outputs matched `baseline_evaluation_results.json` exactly:
- **PMID**: 15 / 15 identical.
- **TF-IDF Relevance Score**: 15 / 15 identical within $10^{-4}$.
- **Gold Grade**: 15 / 15 identical.
- **Normalized Sentence String**: 15 / 15 identical.

---

## 5. Task 3: Relevance-Label Independence Audit

### 5.1 Verification Against Algorithmic Bias
A primary risk in IR benchmarking is circular labeling, where ground-truth labels are assigned based on what the algorithm ranked highly. The audit verified whether the labels were biased by TF-IDF scores:
- **Evidence of Independence**:
  1. In Query 3 (`paracetamol hepatotoxicity`), four core therapeutic/mechanistic statements from PMIDs 42893945, 42883291, and 42850550 were assigned **Grade 3 (Core Finding)** despite receiving a **TF-IDF cosine score of exactly 0.0000** (because they used the synonym "acetaminophen" or "APAP").
  2. In Query 2 (`aspirin cardiotoxicity`), sentence `SENT_0093` ("All patients with CVD were receiving aspirin.") achieved the **#1 highest TF-IDF score (0.0974)**, but was assigned **Grade 2 (Contextual Finding)** rather than Grade 3 because it lacked a direct link to cardiotoxicity.
  3. In Query 3, sentences `SENT_0187` and `SENT_0195` were ranked at **Ranks 4 and 5** by TF-IDF, but were assigned **Grade 1 (Background)** because they were pharmacokinetic/methodology details.
- **Conclusion**: Ground-truth labels were **not** derived from, fitted to, or contaminated by TF-IDF scores.

### 5.2 Rater Protocol Limitation
- The annotations were created by a **single evaluator** under a structured 4-tier rubric.
- There was **no multi-annotator panel** and no inter-annotator agreement score (e.g. Cohen's or Fleiss' kappa).
- **Formal Classification**: The ground truth represents a **Provisional Single-Annotator Ground Truth Protocol**. While scientifically rigorous and consistent, it should not be described as "independently consensus-validated".

---

## 6. Task 4: Evaluation Validity & Scope Classification

### 6.1 Corpus and Candidate Pool Scope
- The benchmark evaluates **query-specific candidate pools** formed by segmenting the top 6 PubMed articles retrieved for each query (55, 67, and 76 sentences).
- This structure **accurately mirrors the production architecture** of MoleculeIQ's `LiteratureAssistantService`, which fetches articles via E-utilities and ranks candidate sentences from those articles.
- The benchmark evaluates **within-abstract sentence extraction, filtering, and cross-sentence semantic ranking**.

### 6.2 Selection Bias & Generalizability Boundaries
- The benchmark represents a **focused diagnostic testbed** (18 articles, 198 sentences across 3 queries), not a global index over the 35+ million records in PubMed.
- High Precision@5 on this benchmark validates sentence ranking quality within query-relevant literature; it does not measure whole-database retrieval recall.

---

## 7. Task 5: Review of Proposed Phase D.4 Acceptance Criteria

The Phase D.3 report proposed the following targets for Phase D.4:
- Macro Fixed P@5 $\ge 0.9333$
- Query 3 Fixed P@5 $\ge 0.8000$
- Query 2 nDCG@5 $\ge 0.8500$
- No Query 1 P@5 regression

### 7.1 Critical Review
1. **Mathematical Justification of 0.9333**:
   - In a 3-query benchmark, Macro P@5 is $\frac{P_1 + P_2 + P_3}{3}$.
   - Baseline: $\frac{1.0 + 1.0 + 0.6}{3} = \frac{2.6}{3} = 0.8667$.
   - If Query 3 improves from 3/5 ($0.60$) to 4/5 ($0.80$) while Query 1 and Query 2 maintain 5/5 ($1.00$), Macro P@5 becomes $\frac{1.0 + 1.0 + 0.8}{3} = \frac{2.8}{3} = 0.93333... \approx 0.9333$.
   - Thus, 0.9333 is **mathematically sound** and corresponds exactly to eliminating at least one false positive in Query 3.
2. **Necessity of nDCG@5 in Query 2**:
   - In Query 2, Fixed P@5 is already $1.0000$ (all 5 sentences are Grade 2 or 3).
   - However, single-term sentences currently outrank dual-term sentences, depressing nDCG@5 to $0.6089$.
   - Precision@5 cannot detect ranking improvements in Query 2. **nDCG@5 is mandatory** to verify that term-conjunction boosting promotes Grade 3 sentences to Ranks 1 and 2.

### 7.2 Formal Phase D.4 Gate Specification
The audit approves the following strict, multi-metric acceptance gate for Phase D.4:

| Metric Target | Baseline | Phase D.4 Target | Justification |
| :--- | :---: | :---: | :--- |
| **Macro Fixed P@5** | 0.8667 | **$\ge 0.9333$** | Requires at least 14 of 15 returned sentences across the benchmark to be relevant. |
| **Query 1 Fixed P@5** | 1.0000 | **1.0000** | Strict zero-regression requirement on core benchmark query. |
| **Query 2 nDCG@5 (Exp)** | 0.6089 | **$\ge 0.8500$** | Validates that conjunction boosting promotes co-occurring Grade 3 findings over single-term Grade 2 sentences. |
| **Query 2 Fixed P@5** | 1.0000 | **1.0000** | Zero-regression requirement on precision. |
| **Query 3 Fixed P@5** | 0.6000 | **$\ge 0.8000$** | Validates that synonym expansion (`paracetamol` $\leftrightarrow$ `acetaminophen`/`APAP`) retrieves true hepatotoxicity findings. |
| **Macro MRR** | 1.0000 | **1.0000** | Must maintain relevant item at Rank 1 for 100% of queries. |
| **Duplicate Return Rate** | 0.00% | **0.00%** | Zero tolerance for redundant statements in top-k output. |

---

## 8. Task 6: Repository Integrity & Safety Verification

### 8.1 Protected Model Artifacts Checksum Audit
All 31 protected model and calibration files in `artifacts/clintox` and `artifacts/tox21` were verified against `artifacts/checksums.sha256`:
- **Result**: **31 / 31 SHA-256 Hashes Matched (100% INTACT)**.
- **Failures / Corruptions**: **0**.

### 8.2 Production Code Modification Audit
- `app.py`: Untouched during Phase D.3 / D.3.1.
- `src/literature/retrieval.py`: Untouched during Phase D.3 / D.3.1.
- `src/literature/pubmed_client.py`: Untouched during Phase D.3 / D.3.1.
- `src/models/inference.py`: Untouched during Phase D.3 / D.3.1.
- All evaluation assets reside in `data/benchmarks/literature/` and `tests/unit/test_pubmed_benchmark.py`.

---

## 9. Final Phase D.3.1 Recommendation

### **RECOMMENDATION: AUTHORIZE PHASE D.4 UNDER STRICT GATE CRITERIA**

1. The Phase D.3 benchmark (`pubmed_benchmark_v1`) is **fully verified, mathematically consistent, reproducible, and free of algorithmic labeling bias**.
2. Its diagnosis of **synonym blindness** and **conjunction neglect** is empirically solid and grounded in genuine clinical literature.
3. Phase D.4 may proceed with targeted, minimal algorithmic improvements:
   - Curated biomedical synonym dictionary mapping (`paracetamol` $\leftrightarrow$ `acetaminophen`, `APAP`; `hepatotoxicity` $\leftrightarrow$ `liver injury`, `DILI`; `cardiotoxicity` $\leftrightarrow$ `cardiovascular toxicity`).
   - Term conjunction co-occurrence bonus (+25% to +50%).
   - Inflectional stem handling for `-toxicity` / `-toxic`.
4. Verification must be executed against the frozen benchmark runner under the multi-metric acceptance gate specified in Section 7.2.
