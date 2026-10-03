# MoleculeIQ — Project State

**Current Status:** STAGE 1 PRE-AUDIT HARDENING COMPLETE (Verified & Passing)  
**Directive Version:** 4.0  
**PRD / SRD Version:** 1.0  
**Audit Identifier:** `AUDIT-REV-2026-01` (Pass With Conditions)  
**Environment:** Windows, Python 3.12.6, Git initialized  

---

## Completed Milestones
- [x] Initial workspace audit conducted and documented.
- [x] Engineering Control Plane established (`.agent/`).
- [x] Complete technical and scientific documentation suite established (`docs/`).
- [x] Module M01: Molecular Input and Structure Identity Engine (`src/chemistry/validation.py`).
- [x] Module M02: Molecular Descriptor Calculation Engine (`src/chemistry/descriptors.py`).
- [x] Module M03: Molecular Fingerprint Generator (`src/chemistry/fingerprints.py`).
- [x] Module M04: ClinTox Ingestion, Preprocessing & Random Forest Baseline (`src/models/clintox.py`).
- [x] Module M05: Tox21 12-Endpoint Ingestion, Multi-Model RF Classifiers (`src/models/tox21.py`).
- [x] Module M06: Chemical Similarity Search Engine & Approved Reference Cache (`src/similarity/`).
- [x] Module M07: Candidate Prioritization & Ranking Engine (`src/ranking/`).
- [x] Module M08: Explainability Engine via SHAP TreeExplainer & Substructure Mapping (`src/explainability/`).
- [x] Module M09: PubMed Literature Retrieval & Semantic Cosine Ranking (`src/literature/`).
- [x] Application Service Layer & DTO Data Contracts (`src/services/`).
- [x] Module M10: Interactive Streamlit Scientific Workspace (`app.py`, `src/ui/`).
- [x] Independent Architecture Review Completed (`docs/INDEPENDENT_ARCHITECTURE_REVIEW.md` - `PASS WITH CONDITIONS`).
- [x] **Stage 1 Remediation Tasks 1–4 Completed:**
  - [x] Task 1: Strict SSL verification adapter with native Windows root certificate store integration and explicit environment fallback (`src/literature/pubmed_client.py`, `tests/unit/test_pubmed_ssl.py`).
  - [x] Task 2: Centralized dynamic path configuration (`src/config.py`, `tests/unit/test_config.py`).
  - [x] Task 3: Bemis-Murcko scaffold extraction & zero-leakage scaffold split benchmark for ClinTox and Tox21 (`src/chemistry/scaffolds.py`, `artifacts/*/scaffold_benchmark.json`, `docs/MODEL_CARD_*.md`, `tests/unit/test_scaffolds.py`).
  - [x] Task 4: Clarified RDKit Lipinski HBA definition excluding amide nitrogens across UI caption, code comments, and scientific documentation (`src/chemistry/descriptors.py`, `src/ui/single_molecule.py`, `docs/SCIENTIFIC_ASSUMPTIONS.md`).
- [x] Automated Test Suite Expanded — 52/52 passing (100% pass rate in 158.28s).

## Verified Workflows
- **Workflow A (Single Molecule Analysis):** End-to-end verified with Aspirin, Caffeine, Paracetamol, Atorvastatin, Metformin.
- **Workflow B (Batch Screening):** End-to-end verified with mixed valid/invalid CSV inputs, row-level fault isolation, ranking, and enriched export.
- **Workflow C (Model Insights):** Verified with held-out metrics (ROC-AUC, PR-AUC, Balanced Accuracy, confusion matrices) and dual random/scaffold benchmark comparisons.
- **Workflow D (Literature Intelligence):** Verified with secure TLS verification against NCBI PubMed search, abstract segmentation, and semantic cosine relevance ranking.
- **Workflow E (Candidate Prioritization):** Verified with formula $R = 0.5S + 0.3D + 0.2M$ and weight sensitivity analysis.

## Quality Gates Status
- Gate 1 (Cheminformatics): MET
- Gate 2 (Machine Learning Models): MET
- Gate 3 (Similarity & Prioritization): MET
- Gate 4 (Explainability & Literature): MET
- Gate 5 (Application & End-to-End Integration): MET
- Gate 6 (Stage 1 Pre-Audit Hardening): MET
