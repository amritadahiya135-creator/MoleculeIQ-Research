# Implementation Roadmap

**Directive Version:** 4.0  
**Methodology:** Incremental, Test-Driven, Research-Aware  

---

## Phase 1: Core Foundation & Chemistry Engine (Tasks 01)
- [x] Engineering Control Plane (`.agent/`) & Project Audit.
- [x] Baseline Documentation Suite (`docs/`).
- [ ] Module M01: Molecular Input Validation (`src/chemistry/validation.py`).
- [ ] Module M02: Molecular Descriptors Engine (`src/chemistry/descriptors.py`).
- [ ] Module M03: Molecular Fingerprint Generator (`src/chemistry/fingerprints.py`).
- [ ] Comprehensive Chemistry Unit Test Suite (`tests/unit/test_chemistry.py`).

## Phase 2: Data Governance & Predictive Modeling (Task 02)
- [ ] Dataset ingestion & verification scripts for ClinTox and Tox21 (`data/raw/`).
- [ ] Module M04: ClinTox Random Forest training pipeline, stratified split, evaluation, and artifact serialization (`src/models/clintox.py`).
- [ ] Module M05: Tox21 12-endpoint training pipeline with missing-label isolation (`src/models/tox21.py`).
- [ ] Model inference service with explicit class probability mapping (`src/models/inference.py`).
- [ ] Model card generation and reproducibility evaluation suite (`tests/integration/test_models.py`).

## Phase 3: Reference Similarity & Prioritization (Tasks 03 & 04)
- [ ] Module M06: ChEMBL reference approved small-molecule cache (`src/similarity/chembl_cache.py`).
- [ ] In-memory Tanimoto similarity search index (`src/similarity/index.py`).
- [ ] Module M07: Candidate Prioritization and Ranking Engine (`src/ranking/scoring.py`).
- [ ] Sensitivity analysis and ranking stability tests (`tests/unit/test_ranking.py`).

## Phase 4: Explainability & Literature Intelligence (Tasks 05 & 06)
- [ ] Module M08: SHAP TreeExplainer integration for ClinTox (`src/explainability/shap_analysis.py`).
- [ ] Mapping active Morgan fingerprint bits to molecular atom environments and SVG/PNG highlights.
- [ ] Module M09: NCBI E-utilities PubMed client with rate-limiting & caching (`src/literature/pubmed_client.py`).
- [ ] Sentence segmentation, semantic embedding ranking, and citation link preservation (`src/literature/retrieval.py`).

## Phase 5: Application Service Layer & Streamlit UI (Tasks 07 & 08)
- [ ] Data contracts & screening orchestrator service (`src/services/contracts.py`, `src/services/orchestration.py`).
- [ ] Streamlit interface:
  - Overview & Scientific Boundaries (`app.py`).
  - Single Molecule Analysis (`src/ui/single_molecule.py`).
  - Batch Screening with CSV upload & download (`src/ui/batch_screening.py`).
  - Model Insights & Metrics (`src/ui/model_insights.py`).
  - Literature Assistant (`src/ui/literature_assistant.py`).
  - Reusable chemical cards & disclaimers (`src/ui/components.py`).

## Phase 6: System Verification, Documentation & Release (Task 09)
- [ ] Full automated test suite execution (Unit, Integration, E2E, Regression).
- [ ] Final model cards and documentation synchronization.
- [ ] Release demonstration verification across Workflows A through E.
