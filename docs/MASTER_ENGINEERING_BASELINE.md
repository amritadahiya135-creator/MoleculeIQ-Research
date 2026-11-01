# MoleculeIQ — Master Engineering Baseline Report

**Document Identifier:** `MEB-V5.0-2026-01`<br>
**Date:** October 4, 2026<br>
**Auditor / Engineering Role:** Master Forensic Engineering Team (Principal Software Architect, ML Systems Engineer, Computational Chemist, Security & DevOps Engineer)<br>
**Target Repository:** MoleculeIQ<br>
**Baseline Git Commit:** `9934604`<br>

---

## 1. Forensic Repository Inventory

### 1.1 Source Code Architecture
The repository is structured into modular Python packages anchored by [`src/config.py`](../src/config.py):

| Package / Module | Primary File | Primary Responsibilities | Current Operational Status |
| :--- | :--- | :--- | :---: |
| **Chemistry Engine** | [`src/chemistry/validation.py`](../src/chemistry/validation.py)<br>[`src/chemistry/descriptors.py`](../src/chemistry/descriptors.py)<br>[`src/chemistry/fingerprints.py`](../src/chemistry/fingerprints.py)<br>[`src/chemistry/scaffolds.py`](../src/chemistry/scaffolds.py) | RDKit structure parsing, Lipinski & QED descriptors, 2048-bit Morgan ECFP4 generation, Bemis–Murcko scaffold decomposition. | **VERIFIED & OPERATIONAL** |
| **Model Pipelines** | [`src/models/clintox.py`](../src/models/clintox.py)<br>[`src/models/tox21.py`](../src/models/tox21.py)<br>[`src/models/inference.py`](../src/models/inference.py) | ClinTox deduplication (`conservative_toxic`), Platt probability calibration, Tox21 12-endpoint multi-models, dynamic applicability domain. | **VERIFIED & OPERATIONAL** |
| **Explainability** | [`src/explainability/shap_analysis.py`](../src/explainability/shap_analysis.py) | TreeSHAP feature attributions on base tree ensemble, atom environment mapping, 2D SVG substructure highlights. | **VERIFIED & OPERATIONAL** |
| **Similarity Search** | [`src/similarity/chembl_cache.py`](../src/similarity/chembl_cache.py)<br>[`src/similarity/index.py`](../src/similarity/index.py) | In-memory Tanimoto similarity search over 20 approved reference therapeutics. | **VERIFIED & OPERATIONAL** |
| **Ranking Engine** | [`src/ranking/scoring.py`](../src/ranking/scoring.py) | Multi-objective linear composite prioritization heuristic ($R = 0.5S + 0.3D + 0.2M$). | **VERIFIED & OPERATIONAL** |
| **Literature Intelligence** | [`src/literature/pubmed_client.py`](../src/literature/pubmed_client.py)<br>[`src/literature/retrieval.py`](../src/literature/retrieval.py) | PubMed E-utilities client with native Windows TLS root certificates and fallback, abstract parsing, cosine semantic ranking. | **VERIFIED & OPERATIONAL** |
| **Application Services** | [`src/services/contracts.py`](../src/services/contracts.py)<br>[`src/services/orchestration.py`](../src/services/orchestration.py) | Strongly-typed DTOs, fault-tolerant workflow orchestration, batch fault isolation. | **VERIFIED & OPERATIONAL** |
| **Presentation Layer** | [`app.py`](../app.py)<br>[`src/ui/`](../src/ui) | Streamlit multipage application: Single Molecule, Batch Screening, Model Insights, Literature Assistant. | **VERIFIED & OPERATIONAL** |

---

### 1.2 Model Artifacts and Data Provenance

| Artifact / Dataset | Location | Properties | Verification Status |
| :--- | :--- | :--- | :---: |
| **ClinTox Calibrated Model** | [`artifacts/clintox/model.joblib`](../artifacts/clintox/model.joblib) | `CalibratedClassifierCV` (sigmoid, 5-fold CV) | **VALID (Brier 0.0522, ECE 0.1604)** |
| **ClinTox Base RF Model** | [`artifacts/clintox/uncalibrated_model.joblib`](../artifacts/clintox/uncalibrated_model.joblib) | `RandomForestClassifier` (300 trees, `balanced`) | **VALID (TreeSHAP compatible)** |
| **ClinTox Training Library** | [`artifacts/clintox/training_fingerprints.npz`](../artifacts/clintox/training_fingerprints.npz) | 1,168 packed 2048-bit Morgan vectors + SMILES | **VALID (Applicability Domain ready)** |
| **Tox21 12-Endpoint Models** | [`artifacts/tox21/*_model.joblib`](../artifacts/tox21) | 12 individual binary classifiers | **VALID (All 12 endpoints load)** |
| **Reference Compounds** | [`data/reference/approved_drugs.csv`](../data/reference/approved_drugs.csv) | 20 FDA-approved benchmark drugs | **VALID (Fingerprint index cached)** |
| **Raw ClinTox Dataset** | [`data/raw/clintox.csv`](../data/raw/clintox.csv) | 1,484 raw rows; 19 conflicting pairs | **PRESERVED & AUDITED** |
| **Raw Tox21 Dataset** | [`data/raw/tox21.csv`](../data/raw/tox21.csv) | 7,831 raw rows; 12 bioassay targets | **PRESERVED & AUDITED** |

---

## 2. Test Suite Baseline

* **Test Framework:** `pytest` 9.1.1 on Python 3.12.6 (Windows 11).
* **Test Count:** **66 collected tests, 66 passing (100% pass rate in 169.60s)**.
* **Coverage Breakdown:**
  - `tests/unit/`: 49 tests (chemistry, config, error handling, ranking, scaffolds, similarity, ssl, clintox_data, calibration, applicability_domain).
  - `tests/integration/`: 15 tests (explainability, literature, model inference).
  - `tests/e2e/`: 2 tests (single-molecule workflow, batch screening workflow).

---

## 3. Forensic Risk and Technical Debt Analysis

### 3.1 Scientific & Machine Learning
* **Morgan Fingerprint Deprecation:** RDKit generates deprecation warnings (`GetMorganFingerprintAsBitVect -> rdFingerprintGenerator.GetMorganGenerator`). Upgrading to modern `MorganGenerator` ensures future RDKit 2025/2026 compatibility while preserving 100% bit-exact parity.
* **Tox21 Model Reporting:** While ClinTox is calibrated and deduplicated, Tox21's 12 endpoints have differing sample sizes and missingness ($7.2\% - 25.8\%$). Model cards and UI must continue highlighting endpoint-specific confidence intervals.

### 3.2 Security, Privacy & Input Hardening
* **CSV Formula Injection:** User-supplied batch CSV files or exported CSV reports should be sanitized against spreadsheet formula injection (`=`, `+`, `-`, `@`).
* **Batch Input Size Safeguards:** Explicit row-count limits (e.g. max 5,000 compounds per batch) should be enforced to protect against denial-of-service through resource exhaustion.
* **Model Deserialization Integrity:** Checksums (SHA-256) should be computed and verified for model artifacts to detect corruption or tampering.

### 3.3 Release Engineering & DevOps
* **Missing Dockerfile:** No container specification currently exists for reproducible cross-platform deployment.
* **Missing CI Pipeline:** No GitHub Actions workflow exists to automate test execution on push/PR.
* **Requirements Pinning:** `requirements.txt` uses minimal `>=` constraints; generating an exact pinned manifest ensures reproducible installations across platforms.

---

## 4. Prioritized V5.0 Action Plan

```
[Phase 0: Baseline Forensics] (COMPLETED)
            |
            v
[Phase 1: Scientific & ML Revalidation Report] --> [docs/FINAL_SCIENTIFIC_VALIDATION_REPORT.md]
            |
            v
[Phase 2: Core Cheminformatics Modernization] --> [Modernize MorganGenerator with fallback + edge tests]
            |
            v
[Phase 3: Model Artifacts, Checksums & MLOps] --> [SHA-256 manifests, integrity checks]
            |
            v
[Phase 4-5: Application, UX & Candidate Ranking] --> [Input size limits, CSV injection defense, UI polish]
            |
            v
[Phase 6-8: Security, Reliability & Benchmarking] --> [Latency benchmarks, performance telemetry]
            |
            v
[Phase 9-10: Release Engineering, Docker & CI] --> [Dockerfile, .github/workflows/ci.yml, release docs]
            |
            v
[Phase 11-12: End-to-End Demo & Release Audit] --> [Walkthrough guide, docs/INDEPENDENT_RELEASE_AUDIT.md]
```
