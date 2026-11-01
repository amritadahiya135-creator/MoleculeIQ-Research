# MoleculeIQ — Phase D.3: Real PubMed Benchmark Construction & Baseline Evaluation Report

**Auditor / Lead:** Senior Biomedical Information Retrieval Researcher & Software Quality Lead<br>
**Evaluation Date:** 19 October 2026<br>
**Benchmark Version:** `pubmed_benchmark_v1` (Frozen)<br>
**Corpus Provenance:** Authentic National Library of Medicine (NCBI) PubMed Abstracts (2024–2026)<br>
**Evaluated Engine:** scikit-learn TF-IDF Semantic Cosine Engine (`ngram_range=(1, 2)`, `sublinear_tf=True`)<br>
**Network State:** OFFLINE / DETERMINISTIC (Frozen Client)<br>
**Protected Artifacts Integrity:** 31 / 31 SHA-256 Hashes Matched (100% INTACT)<br>
**Test Suite Status:** 24 / 24 Tests Passed (100%)<br>
**Final Phase D.3 Verdict:** **PASS (Baseline Established on Authentic Clinical Literature)**<br>

---

## 1. Executive Summary & Gate Verdict

Phase D.3 successfully establishes a **fully reproducible, cryptographically frozen benchmark** constructed exclusively from genuine, peer-reviewed PubMed abstracts. This supersedes the synthetic fixture limitations of Phase D.2 and establishes an empirical baseline for MoleculeIQ's literature ranking pipeline prior to making production algorithmic changes.

### Key Milestones Achieved
1. **Zero Synthetic Records**: Harvested 18 authentic, full-length PubMed articles across 3 core clinical toxicology query families via NCBI Entrez E-utilities. Every record possesses an active PMID, real author list, publication year (2024–2026), and complete abstract text ($>500$ characters).
2. **Four-Tier Graded Relevance Schema**: Decomposed the corpus into 198 deduplicated candidate sentences. Each sentence was individually evaluated and assigned a 4-tier relevance label ($3=\text{Core}$, $2=\text{Contextual}$, $1=\text{Background}$, $0=\text{Irrelevant}$) with clinical rationale. The corpus exhibits an exact 50/50 binary split (99 relevant vs 99 non-relevant).
3. **Deterministic Offline Evaluation**: Implemented a frozen benchmark client and evaluation runner (`data/benchmarks/literature/run_baseline_evaluation.py`) that executes MoleculeIQ's exact production TF-IDF ranking pipeline without external API calls.
4. **Baseline Quantitative Performance**:
   - **Macro MRR**: **1.0000** (Relevant sentence returned at Rank 1 for 100% of queries).
   - **Macro Precision@5 (Fixed Cutoff)**: **0.8667** (Micro: 0.8667, 13/15 relevant).
   - **Macro Precision@k (Variable Denominator)**: **0.8667**.
   - **Macro Recall@5**: **0.1500** (Micro: 0.1313, 13/99 relevant retrieved).
   - **Macro nDCG@5 (Exponential Gain)**: **0.7904** (Linear Gain: 0.8624).
   - **Zero-Overlap Filter Rate**: **87.66%** of irrelevant/orthogonal pool sentences filtered out.
   - **Top-k Duplicate Return Rate**: **0.00%** (100% deduplication integrity).
5. **Clear Algorithmic Diagnosis**: Quantitative failure analysis proved two critical deficiencies in pure TF-IDF cosine ranking: **synonym blindness** (`paracetamol` $\leftrightarrow$ `acetaminophen`/`APAP`) and **term conjunction neglect** (single-term matches outranking co-occurring sentences).
6. **Strict Repository Preservation**: Zero production code modifications, zero Git resets, zero model retrainings, and 100% cryptographic integrity across all 31 protected models.

---

## 2. Benchmark Corpus & Provenance

The frozen benchmark dataset is persisted in `data/benchmarks/literature/pubmed_benchmark_v1.json`.

### 2.1 Query Families & Article Allocation

| Query ID | Clinical Query Text | Article Count | Unique PMIDs | Candidate Sentences | Relevant Sentences ($\text{Grade} \ge 2$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q1_METFORMIN_LACTIC_ACIDOSIS` | `metformin lactic acidosis` | 6 | 6 | 55 | 42 (76.4%) |
| `Q2_ASPIRIN_CARDIOTOXICITY` | `aspirin cardiotoxicity` | 6 | 6 | 67 | 20 (29.9%) |
| `Q3_PARACETAMOL_HEPATOTOXICITY` | `paracetamol hepatotoxicity` | 6 | 6 | 76 | 37 (48.7%) |
| **Total** | — | **18** | **18** | **198** | **99 (50.0%)** |

### 2.2 Complete Provenance Manifest

```
Query 1: "metformin lactic acidosis"
  • PMID 42834623 (2026): "Successful treatment of metformin-associated lactic acidosis without hemodialysis." (1,738 chars)
  • PMID 42789768 (2026): "Concurrent metformin-associated lactic acidosis and euglycemic diabetic ketoacidosis from tirzepatide starvation ketosis." (1,404 chars)
  • PMID 42730466 (2026): "Severe Metformin Intoxication Resulting in Extreme Lactic Acidosis." (1,784 chars)
  • PMID 42696402 (2026): "Metformin-Associated Lactic Acidosis and Relative Euglycemic Ketoacidosis Precipitated by a Glucagon-Like Peptide-1 Receptor Agonist: A Case Report and Literature Review." (1,419 chars)
  • PMID 42545322 (2026): "Metformin's disruption of gluconeogenesis in type 2 diabetes impairments of the cori and alanine cycles as hidden drivers of metabolic waste accumulation, NF-κBHIF-1α-mediated low-grade inflammation, and multisystem dysfunction." (1,424 chars)
  • PMID 42377238 (2026): "Atypical Presentation of Metformin-associated Lactic Acidosis: A Case Report." (1,489 chars)

Query 2: "aspirin cardiotoxicity"
  • PMID 42756119 (2026): "A narrative review of emerging chemoprotective agents for chemotherapy-induced toxicity: nanoformulated natural products, dual COX-2/TNF-α inhibitors, and Nrf2 modulators." (1,939 chars)
  • PMID 42211159 (2026): "Polycythemia Vera and Cardiovascular Disease: A Mini Review." (970 chars)
  • PMID 42131099 (2026): "Case Report: A case analysis of antitumor therapy and cardiovascular safety management in a breast cancer patient with multiple cardiovascular diseases." (1,446 chars)
  • PMID 41040531 (2025): "Cardiovascular comorbidities and cancer-directed therapies in Hispanic breast cancer patients: a two-center analysis from the Southwestern U.S." (1,293 chars)
  • PMID 40364115 (2025): "Risk Factors and Prevention of Cancer and CVDs: A Chicken and Egg Situation." (1,847 chars)
  • PMID 39357328 (2024): "Aspirin reduces Ponatinib-induced cardiovascular toxic phenotypes and death in zebrafish." (1,154 chars)

Query 3: "paracetamol hepatotoxicity"
  • PMID 42893945 (2026): "TMEM121 aggravates acetaminophen-induced pyroptosis in hepatocellular carcinoma cells by activating the NLRP3/caspase-1/GSDMD pathway." (1,570 chars)
  • PMID 42883291 (2026): "Hepatoprotective Effects of Flavonoids Against Acetaminophen Toxicity." (1,061 chars)
  • PMID 42850550 (2026): "An Organoid-Based Assay for Evaluating Oxidative Stress-Reactive Fluorescent Probes Using the CERO 3D Bioreactor in a Mouse Hepatic Oxidative Injury Model." (1,328 chars)
  • PMID 42811846 (2026): "Effect of Moringa Oleifera Leaf Extract on Paracetamol Induced Liver Damage in Wistar Albino Male Rats." (2,168 chars)
  • PMID 42807834 (2026): "Hippocampal NLRP3 Inflammasome Drives Anxiety- and Depressive-Like Behaviors Following Drug-Induced Liver Injury." (1,232 chars)
  • PMID 42781941 (2026): "Age-related patterns in paediatric paracetamol poisoning: a retrospective cohort study, 2015-2025." (1,507 chars)
```

---

## 3. Sentence Segmentation & Relevance Annotation Protocol

Candidate sentences were extracted using `split_into_sentences()` and deduplicated via `normalize_sentence_key()`. The complete annotation dataset is stored in `data/benchmarks/literature/relevance_annotations_v1.json`.

### 3.1 Four-Tier Graded Relevance Definitions

- **Grade 3 (Highly Relevant / Core Finding)**: The sentence directly pairs the target molecule with the adverse effect, clinical toxicity manifestation, biochemical mechanism, incidence, or direct treatment of that specific toxicity.
- **Grade 2 (Moderately Relevant / Contextual Finding)**: The sentence discusses the target molecule or the toxic endpoint with substantive contextual relevance (e.g. general pharmacology, toxicity biomarkers, patient management in a related clinical context).
- **Grade 1 (Marginally Relevant / Background)**: The sentence describes general disease background, baseline patient epidemiology, or unrelated assays without a direct finding on the query topic.
- **Grade 0 (Irrelevant / Off-Topic)**: Methodological parameters (e.g., animal weights, assay kits, statistical software versions), demographics, or unrelated clinical conditions.

### 3.2 Annotation Distribution Breakdown

| Metric | Overall Corpus | Q1 (Metformin) | Q2 (Aspirin) | Q3 (Paracetamol) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Annotated Sentences** | **198** | **55** | **67** | **76** |
| Grade 3 (Core Finding) | 43 (21.7%) | 25 (45.5%) | 5 (7.5%) | 13 (17.1%) |
| Grade 2 (Contextual Finding) | 56 (28.3%) | 17 (30.9%) | 15 (22.4%) | 24 (31.6%) |
| Grade 1 (Background) | 47 (23.7%) | 9 (16.4%) | 21 (31.3%) | 17 (22.4%) |
| Grade 0 (Irrelevant) | 52 (26.3%) | 4 (7.3%) | 26 (38.8%) | 22 (28.9%) |
| **Binary Relevant (Grade $\ge$ 2)** | **99 (50.0%)** | **42 (76.4%)** | **20 (29.9%)** | **37 (48.7%)** |
| **Binary Non-Relevant (Grade $<$ 2)** | **99 (50.0%)** | **13 (23.6%)** | **47 (70.1%)** | **39 (51.3%)** |

The benchmark achieves an exact 50/50 balance between relevant and non-relevant items, avoiding both sparse target pathology and saturated relevance traps.

---

## 4. Frozen Baseline Evaluation Results

The evaluation runner executed `LiteratureAssistantService.search_and_rank()` using the frozen benchmark client and production TF-IDF configuration (`ngram_range=(1,2)`, `sublinear_tf=True`, `cosine_similarity`).

### 4.1 Aggregate Benchmark Performance

| Evaluation Metric | Mathematical Definition | Macro-Average | Micro-Average |
| :--- | :--- | :---: | :---: |
| **Mean Reciprocal Rank (MRR)** | $\frac{1}{\|Q\|} \sum_{q} \frac{1}{\text{rank}_{\text{first rel}}}$ | **1.0000** | — |
| **Precision@5 (Fixed Cutoff)** | $\frac{1}{5} \sum_{i=1}^5 \mathbb{I}(\text{Grade}_i \ge 2)$ | **0.8667** | **0.8667** (13 / 15) |
| **Precision@k (Variable Denominator)** | $\frac{1}{\min(5, \|\text{Ret}\|)} \sum_{i=1}^k \mathbb{I}(\text{Grade}_i \ge 2)$ | **0.8667** | — |
| **Recall@5** | $\frac{\text{Relevant in Top 5}}{\text{Total Relevant in Pool}}$ | **0.1500** | **0.1313** (13 / 99) |
| **nDCG@5 (Exponential Gain)** | $\frac{\text{DCG@5}}{\text{IDCG@5}}, \text{Gain} = 2^{\text{grade}} - 1$ | **0.7904** | — |
| **nDCG@5 (Linear Gain)** | $\frac{\text{DCG@5}}{\text{IDCG@5}}, \text{Gain} = \text{grade}$ | **0.8624** | — |
| **Zero-Overlap Filter Rate** | Fraction of pool with $\text{Score} \le 0.0$ | **0.8766** | — |
| **Top-k Duplicate Rate** | Redundant normalized keys in Top 5 | **0.0000** | **0.0000** |

---

## 5. Query-by-Query Breakdown & Failure Analysis

### 5.1 Query 1: `metformin lactic acidosis`
- **Candidate Pool**: 55 sentences (42 relevant, 76.4% prevalence).
- **Returned Sentences (Score $>$ 0)**: 8 sentences (85.5% of pool filtered by zero-overlap).
- **Query Metrics**:
  - MRR: **1.0000**
  - Precision@5 (Fixed): **1.0000** (5 / 5)
  - Precision@k (Variable): **1.0000** (5 / 5)
  - Recall@5: **0.1190** (5 / 42)
  - nDCG@5 (Exponential): **1.0000** (Perfect: all 5 returned sentences are Grade 3 Core Findings).

#### Top-5 Ranked Sentences:
1. **Rank 1 [Grade 3 | Score 0.2082 | PMID 42834623]**:<br>
   *"Metformin-associated lactic acidosis (MALA) is a rare but potentially life-threatening condition."*
2. **Rank 2 [Grade 3 | Score 0.1859 | PMID 42377238]**:<br>
   *"However, it is rarely associated with life-threatening lactic acidosis."*
3. **Rank 3 [Grade 3 | Score 0.1555 | PMID 42730466]**:<br>
   *"In conclusion, massive metformin overdose may lead to life-threatening lactic acidosis, multi-organ dysfunction, and circulatory collapse."*
4. **Rank 4 [Grade 3 | Score 0.1375 | PMID 42696402]**:<br>
   *"Metformin-associated lactic acidosis (MALA) is an uncommon but often fatal complication of metformin therapy (incidence less than 10 per 100,000 patient-years; with a 30-50% mortality)."*
5. **Rank 5 [Grade 3 | Score 0.1180 | PMID 42789768]**:<br>
   *"We describe a case of concurrent metformin-associated lactic acidosis (MALA) and euglycemic diabetic ketoacidosis (EDKA) in a 64-year-old woman with type 2 diabetes mellitus (T2DM) treated with metformin and tirzepatide who presented with shock, encephalopathy, and severe high anion gap metabolic acidosis after two weeks of markedly reduced oral intake following tirzepatide dose escalation."*

**Query 1 Assessment**: Outstanding performance. The query terms "metformin", "lactic", and "acidosis" align directly with the standardized clinical terminology used in peer-reviewed abstracts. Every top-5 slot contains a definitive Grade 3 statement.

---

### 5.2 Query 2: `aspirin cardiotoxicity`
- **Candidate Pool**: 67 sentences (20 relevant, 29.9% prevalence).
- **Returned Sentences (Score $>$ 0)**: 8 sentences (88.1% of pool filtered by zero-overlap).
- **Query Metrics**:
  - MRR: **1.0000**
  - Precision@5 (Fixed): **1.0000** (5 / 5)
  - Precision@k (Variable): **1.0000** (5 / 5)
  - Recall@5: **0.2500** (5 / 20)
  - nDCG@5 (Exponential): **0.6089** (Grade order: [2, 2, 3, 3, 2]; ideal would be [3, 3, 3, 2, 2]).

#### Top-5 Ranked Sentences:
1. **Rank 1 [Grade 2 | Score 0.0974 | PMID 41040531]**:<br>
   *"All patients with CVD were receiving aspirin."*<br>
   *Failure Mechanism*: Contains "aspirin" in a very short sentence (5 words), producing an inflated TF-IDF weight for "aspirin" alone, despite zero mention of cardiotoxicity.
2. **Rank 2 [Grade 2 | Score 0.0791 | PMID 42131099]**:<br>
   *"Following multidisciplinary team (MDT) consultation, the low-cardiotoxicity PTD regimen (pertuzumab, trastuzumab, docetaxel) was selected."*<br>
   *Failure Mechanism*: Contains "cardiotoxicity" alone, without any mention of aspirin.
3. **Rank 3 [Grade 3 | Score 0.0728 | PMID 39357328]**:<br>
   *"To investigate the effects of Aspirin on Ponatinib-induced cardiovascular toxicity in zebrafish."*<br>
   *Core Finding*: Contains BOTH "Aspirin" and "cardiovascular toxicity". Outranked by single-term sentences at Ranks 1 and 2 due to term conjunction neglect.
4. **Rank 4 [Grade 3 | Score 0.0662 | PMID 39357328]**:<br>
   *"Our findings suggest that Aspirin exhibits the potential to reduce Ponatinib-induced cardiovascular toxicity."*<br>
   *Core Finding*: Contains BOTH "Aspirin" and "cardiovascular toxicity".
5. **Rank 5 [Grade 2 | Score 0.0568 | PMID 42211159]**:<br>
   *"This mini review summarizes the cardiovascular burden of PV, therapeutic strategies to mitigate risk, and emerging perspectives on cardiotoxicity-including a recently reported case of reversible left ventricular dysfunction associated with ropeginterferon α-2b."*

**Query 2 Assessment**: While fixed Precision@5 reached 100% (all 5 were Grade 2 or 3), the **ranking order is suboptimal** (nDCG@5 = 0.6089). Sentences with single-term lexical hits ("aspirin" only or "cardiotoxicity" only) outranked sentences containing the true conjunction of the drug and the adverse endpoint.

---

### 5.3 Query 3: `paracetamol hepatotoxicity`
- **Candidate Pool**: 76 sentences (37 relevant, 48.7% prevalence).
- **Returned Sentences (Score $>$ 0)**: 8 sentences (89.5% of pool filtered by zero-overlap).
- **Query Metrics**:
  - MRR: **1.0000**
  - Precision@5 (Fixed): **0.6000** (3 / 5)
  - Precision@k (Variable): **0.6000** (3 / 5)
  - Recall@5: **0.0811** (3 / 37)
  - nDCG@5 (Exponential): **0.7623** (Grade order: [3, 3, 3, 1, 1]; ideal would be [3, 3, 3, 3, 3]).

#### Top-5 Ranked Sentences:
1. **Rank 1 [Grade 3 | Score 0.1534 | PMID 42811846]**:<br>
   *"Paracetamol is a common analgesics and antipyretic drug but at overdose can cause severe hepatotoxicity."*<br>
   *Core Finding*: Direct hit on both "paracetamol" and "hepatotoxicity".
2. **Rank 2 [Grade 3 | Score 0.0727 | PMID 42807834]**:<br>
   *"Moreover, N-acetylcysteine, an agent against APAP-induced hepatotoxicity, inhibited NLRP3 inflammasome activation and alleviated anxiety- and depressive-like behaviors in DILI mice."*<br>
   *Core Finding*: Exact match on "hepatotoxicity" and antidote for APAP.
3. **Rank 3 [Grade 3 | Score 0.0641 | PMID 42781941]**:<br>
   *"Hepatotoxicity (ALT > 3 × ULN and/or bilirubin > 2 × ULN) occurred in 21.4%, and acute liver failure (according to established paediatric criteria) in 9.2%, without significant differences between age groups."*<br>
   *Core Finding*: Quantitative clinical incidence of paracetamol-induced hepatotoxicity.
4. **Rank 4 [Grade 1 | Score 0.0639 | PMID 42781941]**:<br>
   *"Cases were identified through serum paracetamol measurements and classified by exposure pattern."*<br>
   *Failure Mechanism*: Contains "paracetamol" in a study methodology sentence. Scored higher than Grade 3 sentences because Grade 3 sentences used the synonym "acetaminophen" or "APAP".
5. **Rank 5 [Grade 1 | Score 0.0567 | PMID 42781941]**:<br>
   *"Reported ingested dose showed only weak correlation with serum paracetamol concentrations, particularly in younger children."*<br>
   *Failure Mechanism*: Contains "paracetamol" in a pharmacokinetic correlation statement without toxicity data.

#### False Negatives (Grade 3 sentences scored 0.0000):
- PMID 42893945: *"Acetaminophen (APAP) is a widely used analgesic and antipyretic drug, but overdose causes severe acute liver injury."* (Grade 3, Score = 0.0000)
- PMID 42883291: *"Acetaminophen (APAP) is a common analgesic and antipyretic agent; however, overdose can cause severe liver injury requiring transplantation or resulting in death."* (Grade 3, Score = 0.0000)
- PMID 42883291: *"Currently, N-acetylcysteine (NAC) is the standard first-line treatment for APAP-induced liver injury (AILI)."* (Grade 3, Score = 0.0000)
- PMID 42850550: *"Here we describe a protocol for evaluating carbonyl-reactive fluorescent probes in mouse hepatic organoids grown in suspension in the CERO 3D bioreactor, using acetaminophen (APAP) overdose as a well-established model of acute oxidative liver injury."* (Grade 3, Score = 0.0000)

**Query 3 Assessment**: This query demonstrates the **primary scientific failure mode of unaugmented TF-IDF**: **synonym blindness**. In biomedical literature, "acetaminophen" and "APAP" are ubiquitous synonyms for "paracetamol". Because TF-IDF relies strictly on surface n-grams, articles using standard US/scientific terminology were completely invisible to the retrieval engine, allowing peripheral Grade 1 sentences containing "paracetamol" to capture Ranks 4 and 5.

---

## 6. Root-Cause Analysis of Current TF-IDF Ranking Deficiencies

The baseline evaluation isolates three concrete algorithmic limitations in MoleculeIQ's current ranking implementation:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      TF-IDF RANKING DEFICIENCY ROOT-CAUSE MAP                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
     ┌───────────────────────────────────┼──────────────────────────────────┐
     ▼                                   ▼                                  ▼
1. SYNONYM BLINDNESS              2. CONJUNCTION NEGLECT             3. MORPHOLOGY MISMATCH
- "paracetamol" missed            - Single-term sentence             - "cardiotoxicity" vs
  "acetaminophen" & "APAP"          Rank 1: "aspirin" only             "cardiotoxic" vs
- "hepatotoxicity" missed         - Co-occurring sentence              "cardiovascular toxicity"
  "acute liver injury" & "DILI"     Rank 3: both terms               - "acidosis" vs "acidemia"
- Causes 0.0000 score on          - Cosine similarity treats         - Unstemmed TF-IDF splits
  definitive Grade 3 claims         isolated hits similarly to          related variants into
  (e.g., in PMID 42893945)          full co-occurrences                 independent n-grams
```

1. **Biomedical Synonym Blindness**:
   - The query `paracetamol` fails to retrieve abstracts using `acetaminophen` or `APAP`.
   - The query `hepatotoxicity` fails to match `liver injury`, `hepatic damage`, `DILI`, or `AILI`.
   - The query `lactic acidosis` misses `hyperlactatemia` or `acidemia`.
   - *Impact*: Low Recall@5 (0.0811 in Q3) and false-positive promotion of peripheral sentences that happen to contain the exact queried word.

2. **Term Conjunction Neglect**:
   - Cosine similarity on sublinear TF-IDF weights computes inner products of document vectors without rewarding the simultaneous presence of both the drug entity and the adverse outcome.
   - A short sentence containing only "aspirin" can achieve a higher cosine similarity than a longer sentence containing both "aspirin" and "cardiovascular toxicity".
   - *Impact*: Inverted ranking order in Q2 (nDCG@5 dropped to 0.6089).

3. **Morphological / Subword Blindness**:
   - Surface string matching cannot recognize that "cardiotoxic" and "cardiotoxicity" are inflectional variants of the same root.
   - *Impact*: Incomplete feature representation in `TfidfVectorizer(ngram_range=(1,2))`.

---

## 7. Recommendations & Scope for Phase D.4

Phase D.3 successfully establishes the empirical baseline. In accordance with the stop condition, **no ranking algorithm changes have been made in Phase D.3**.

### Prioritized Roadmap for Phase D.4:
1. **Biomedical Synonym Expansion (Primary Intervention)**:
   - Introduce a curated, deterministic clinical synonym dictionary for core screening molecules (e.g. `paracetamol` $\leftrightarrow$ `acetaminophen`, `APAP`; `aspirin` $\leftrightarrow$ `acetylsalicylic acid`, `ASA`) and core toxicity endpoints (`hepatotoxicity` $\leftrightarrow$ `liver injury`, `hepatic failure`; `cardiotoxicity` $\leftrightarrow$ `cardiovascular toxicity`, `cardiac damage`).
   - Query expansion should apply controlled synonym mapping prior to TF-IDF vectorization without introducing nondeterministic external models.
2. **Conjunction Co-Occurrence Bonus (Secondary Intervention)**:
   - Introduce an explicit co-occurrence multiplier $\beta$ (e.g. $+25\%$ to $+50\%$) for sentences containing at least one term from the drug entity set AND at least one term from the adverse outcome set.
   - This directly resolves the inverted ranking in Q2 where single-term sentences outranked dual-term findings.
3. **Lightweight Biomedical Lemmatization / Stemming**:
   - Add stemming/lemmatization (e.g., Snowball or simple suffix rules) to collapse `cardiotoxicity` / `cardiotoxic` and `acidosis` / `acidemic`.
4. **Verification Gate**:
   - Re-evaluate on `pubmed_benchmark_v1.json`.
   - Gate criteria: Macro P@5 must increase from 0.8667 to $\ge 0.9333$, Q3 P@5 must increase from 0.6000 to $\ge 0.8000$, and Q2 nDCG@5 must increase from 0.6089 to $\ge 0.8500$, with zero regression on Q1.

---

## 8. Repository Safety, Artifact Integrity & Test Verification

### 8.1 Cryptographic Verification of Protected Artifacts
All 31 protected model files across `artifacts/clintox` and `artifacts/tox21` were verified against `artifacts/checksums.sha256`:
- **Verification Result**: **31 / 31 SHA-256 Hashes Matched (100% INTACT)**.
- **Corrupted / Modified Artifacts**: **0**.

### 8.2 Unit Test Execution
The test suite was executed via `pytest`:
```powershell
pytest tests/unit/test_pubmed_benchmark.py tests/unit/test_literature_retrieval.py -v
============================= 24 passed in 13.26s =============================
```
- Benchmark schema and integrity tests: **4 passed**.
- Mathematical metric tests: **2 passed**.
- Deterministic baseline evaluation tests: **1 passed**.
- Literature retrieval and deduplication tests: **17 passed**.

### 8.3 Git Repository Status
- Zero production application files (`app.py`, `src/literature/retrieval.py`, `src/literature/pubmed_client.py`) were modified in Phase D.3.
- All evaluation assets reside in isolated benchmark directories (`data/benchmarks/literature/` and `tests/unit/test_pubmed_benchmark.py`).
- No Git reset, clean, stash, or branch checkout operations were executed.

---

## 9. Final Phase D.3 Verdict

### **VERDICT: PASS (BASELINE AUDIT COMPLETE & REPRODUCIBLE)**

Phase D.3 has delivered an authentic, peer-reviewed PubMed benchmark with complete graded annotations, established an empirical baseline for the production TF-IDF ranking engine, isolated root-cause ranking failure modes, verified 100% artifact and test integrity, and outlined a focused, evidence-backed implementation plan for Phase D.4.
