# MoleculeIQ — Literature Retrieval & Ranking Evaluation Protocol
**Version:** 1.0.0 (Frozen Benchmark `pubmed_benchmark_v1`)<br>
**Scope:** Phase D.3 Real PubMed Baseline Evaluation<br>
**Date Established:** 19 October 2026<br>
**Status:** FROZEN REFERENCE SPECIFICATION<br>

---

## 1. Objective & Scope

This protocol establishes the standardized, reproducible methodology for evaluating MoleculeIQ's literature retrieval and sentence ranking pipeline (Module M09, `src/literature/retrieval.py`).

Unlike synthetic unit-test fixtures, this evaluation operates exclusively on **authentic, peer-reviewed PubMed abstracts** fetched from the NCBI E-utilities API and frozen with cryptographic SHA-256 checksums.

This document defines:
1. Benchmark corpus composition and provenance.
2. Candidate sentence segmentation and deduplication logic.
3. Four-tier graded relevance schema.
4. Formal mathematical definitions for all information-retrieval metrics.
5. Baseline execution rules preventing data leakage or algorithm drift.

---

## 2. Benchmark Corpus Definition

The frozen benchmark dataset resides in `data/benchmarks/literature/pubmed_benchmark_v1.json` and consists of 18 unique, full-length PubMed records across three representative clinical toxicology query families:

| Query ID | Clinical Query Text | Article Count | Unique PMIDs | Sentence Count | Relevant (Grade $\ge$ 2) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q1_METFORMIN_LACTIC_ACIDOSIS` | `metformin lactic acidosis` | 6 | 6 | 55 | 42 (76.4%) |
| `Q2_ASPIRIN_CARDIOTOXICITY` | `aspirin cardiotoxicity` | 6 | 6 | 67 | 20 (29.9%) |
| `Q3_PARACETAMOL_HEPATOTOXICITY` | `paracetamol hepatotoxicity` | 6 | 6 | 76 | 37 (48.7%) |
| **Total** | — | **18** | **18** | **198** | **99 (50.0%)** |

### Provenance Criteria
1. **Verifiable NCBI PMIDs**: Every article exists in the National Library of Medicine PubMed repository and possesses an active URL `https://pubmed.ncbi.nlm.nih.gov/<PMID>/`.
2. **Complete Abstracts**: Only records containing $>500$ characters of structured or unstructured narrative abstract text are eligible.
3. **Clinical/Toxicological Diversity**: Includes pediatric cases, intentional overdoses, drug-drug interactions, multi-organ failure, preclinical organoid models, and epidemiology.

---

## 3. Segmentation & Deduplication Specification

To ensure exact fidelity with MoleculeIQ's production pipeline, sentence extraction and deduplication must strictly replicate `src/literature/retrieval.py`:

### 3.1 Sentence Segmentation (`split_into_sentences`)
1. Normalize whitespace by collapsing consecutive whitespace characters (`\s+`) to a single space.
2. Split on sentence-terminal punctuation (`.`, `!`, `?`) followed by whitespace and an uppercase letter or number:
   ```python
   re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_text)
   ```
3. Exclude fragments shorter than or equal to 25 characters (filters out stray abbreviations, section labels, or malformed punctuation).

### 3.2 Canonical Key Normalization (`normalize_sentence_key`)
Sentence deduplication relies on a canonical lookup key:
1. Strip leading/trailing whitespace and lowercase the text.
2. Normalize typographical quotes and accents: `’`, `‘`, `` ` `` $\to$ `'`, `“`, `”` $\to$ `"`.
3. Normalize em-dashes and en-dashes: `–`, `—` $\to$ `-`.
4. Strip trailing punctuation: `.`, `!`, `?`, `:`, `;`.

### 3.3 Deduplication Enforcement
- **Intra-article**: Applied during candidate sentence pooling from abstract text. Duplicate sentences within the same article are discarded.
- **Inter-article**: Applied during descending score ranking. If two distinct articles contain identical sentences (or duplicate publication of the same statement), only the highest-scoring instance is retained.

---

## 4. Graded Relevance Schema

Relevance annotations are recorded in `data/benchmarks/literature/relevance_annotations_v2.json` using a four-tier graded scale aligned with TREC/Biomedical Information Retrieval standards:

| Grade | Label | Description | Operational Inclusion Rule |
| :---: | :--- | :--- | :--- |
| **3** | **Highly Relevant / Core Finding** | Directly connects the drug entity to the specific adverse endpoint, toxicity mechanism, clinical presentation, or treatment of the named toxicity. | Drug AND Toxicity directly paired with actionable or conclusive claim. |
| **2** | **Moderately Relevant / Contextual Finding** | Discusses the drug or endpoint with strong contextual relevance (e.g. general pharmacology, toxicity biomarker changes, clinical management in related setting). | Substantive scientific link to the clinical topic without full explicit pairing in a single sentence. |
| **1** | **Marginally Relevant / Background** | Mentions general disease epidemiology, baseline patient health, or unrelated assays/pathways without direct claim on query topic. | Peripheral or background medical information. |
| **0** | **Irrelevant / Off-Topic** | Methodological parameters, assay reagents, animal housing, patient demographics, or administrative statements. | Zero clinical or toxicological information regarding the query. |

### Binary Mapping
For binary retrieval evaluation metrics (MRR, P@5, Recall@5):
$$\text{Binary Label} = \begin{cases} 1 & \text{if } \text{Gold Grade} \ge 2 \\ 0 & \text{if } \text{Gold Grade} < 2 \end{cases}$$

---

## 5. Mathematical Metric Definitions

Let $Q$ denote the set of queries ($|Q| = 3$), $R_q$ denote the set of relevant sentences in the pool for query $q$ ($\text{Binary Label} = 1$), and $\hat{S}_q = [s_1, s_2, \dots, s_k]$ denote the ranked sentences returned by the model for query $q$ (typically $k=5$ or $k=8$).

### 5.1 Mean Reciprocal Rank (MRR)
For a query $q$:
$$\text{RR}_q = \begin{cases} \frac{1}{\min \{ i : \text{Binary Label}(s_i) = 1 \}} & \text{if } \exists i \le k \text{ s.t. } \text{Binary Label}(s_i) = 1 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}_q$$

### 5.2 Precision@5 (Fixed Cutoff)
Standard information-retrieval convention with a fixed denominator of 5:
$$\text{Precision@5}_q = \frac{1}{5} \sum_{i=1}^{\min(5, |\hat{S}_q|)} \mathbb{I}(\text{Binary Label}(s_i) = 1)$$
If fewer than 5 sentences are returned, the unreturned slots contribute 0 to the numerator while the denominator remains 5.

### 5.3 Precision@k (Variable Denominator)
Alternative UI-focused metric where the denominator is capped at the number of returned items:
$$\text{Precision@k}_q = \frac{1}{\min(5, |\hat{S}_q|)} \sum_{i=1}^{\min(5, |\hat{S}_q|)} \mathbb{I}(\text{Binary Label}(s_i) = 1)$$
*(Both metrics must be reported simultaneously to eliminate reporting ambiguity).*

### 5.4 Recall@5
The proportion of all relevant sentences in the candidate pool retrieved in the top 5 ranks:
$$\text{Recall@5}_q = \frac{\sum_{i=1}^{\min(5, |\hat{S}_q|)} \mathbb{I}(\text{Binary Label}(s_i) = 1)}{|R_q|}$$

### 5.5 Normalized Discounted Cumulative Gain (nDCG@5)
Evaluates graded ranking quality (grades 0, 1, 2, 3) with logarithmic positional discounting:
$$\text{DCG@5}_q = \sum_{i=1}^{\min(5, |\hat{S}_q|)} \frac{2^{\text{Grade}(s_i)} - 1}{\log_2(i + 1)}$$
Let $\text{IDCG@5}_q$ be the DCG@5 of the ideal ranking obtained by sorting all pool sentences in descending order of gold grade:
$$\text{nDCG@5}_q = \frac{\text{DCG@5}_q}{\text{IDCG@5}_q}$$

### 5.6 Operational Pipeline Metrics
- **Zero-Overlap Filter Rate**:
  $$\text{Filter Rate}_q = \frac{\sum_{s \in \text{Pool}_q} \mathbb{I}(\text{Cosine Score}(s, q) \le 0.0)}{|\text{Pool}_q|}$$
- **Top-k Duplicate Rate**:
  $$\text{Duplicate Rate}_q = 1.0 - \frac{|\{ \text{normalize\_sentence\_key}(s_i) : i \le k \}|}{k}$$

### 5.7 Aggregation Conventions
- **Macro-average**: The unweighted arithmetic mean across all 3 queries: $\text{Macro}(M) = \frac{1}{3} \sum_{q=1}^3 M_q$.
- **Micro-average**: Pooled evaluation treating all returned sentences across all queries as a single combined cohort.

---

## 6. Execution Safeguards & Integrity

1. **Frozen Input Data**: The runner MUST read exclusively from `pubmed_benchmark_v1.json` and `relevance_annotations_v2.json`. Network calls to NCBI E-utilities are strictly prohibited during benchmark execution.
2. **Production Code Fidelity**: The ranking pipeline must execute the production TF-IDF vectorizer configuration: `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)` with `cosine_similarity`.
3. **No Retraining / No Weight Tuning**: Phase D.3 serves as a read-only baseline audit. Zero hyperparameters or ranking weights may be tuned in this phase.
