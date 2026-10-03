# Release Notes: MoleculeIQ v1.0.0

**Release Date:** 2026-10-03  
**Status:** Verification Complete — Demonstrable Research Prototype  
**Baseline Directive:** Directive Version 4.0  
**PRD / SRD:** PRD v1.0, SRD v1.0  

---

## Highlights of Release v1.0.0

### 1. Robust Cheminformatics Engine (Modules M01, M02, M03)
- Molecular syntax validation, SMILES canonicalization, formula generation, and heavy atom counting.
- Full physicochemical descriptors calculation: Molecular Weight, LogP, H-Bond Donors, H-Bond Acceptors, Rotatable Bonds, TPSA.
- Lipinski Rule of 5 evaluation with transparent violation annotations.
- Standardized 2048-bit Morgan circular fingerprints (radius=2) with atom environment bit mapping for XAI.

### 2. Machine Learning Predictive Toxicology (Modules M04, M05)
- **ClinTox Baseline:** Random Forest classifier trained on MoleculeNet ClinTox dataset (`CT_TOX` target) achieving held-out ROC-AUC of **0.8517** and PR-AUC of **0.2972**.
- **Tox21 12-Endpoint Suite:** 12 independent Random Forest classifiers trained with strict missing-assay isolation, achieving ROC-AUCs up to **0.9092** (`NR-AhR`) and **0.9083** (`SR-ATAD5`).
- Dynamic positive class mapping verifying `model.classes_` to eliminate probability inversion risks.

### 3. Chemical Similarity & Prioritization (Modules M06, M07)
- In-memory Tanimoto similarity search on 2048-bit Morgan fingerprints against a curated library of approved small molecules.
- Deterministic, multi-criteria prioritization formula: $R = 0.5S + 0.3D + 0.2M$ ($S$: Safety, $D$: Lipinski, $M$: Reference Similarity).
- Interactive weight sensitivity analysis evaluating ranking stability across candidate perturbations.

### 4. Explainable AI & Substructure Mapping (Module M08)
- Integration with SHAP `TreeExplainer` on ClinTox Random Forest model.
- Extraction of active fingerprint bits with direction of influence (risk increasing vs. decreasing).
- Topological mapping of active bits to molecular atoms with 2D SVG vector highlights.

### 5. Literature Intelligence Assistant (Module M09)
- Programmatic retrieval of peer-reviewed articles from NCBI PubMed via E-utilities.
- Sentence-level segmentation and semantic cosine relevance ranking with source article provenance (PMID, title, authors, year, direct links).

### 6. Interactive Scientific Workspace (Module M10)
- Streamlit application supporting:
  - Overview and Scientific Boundaries.
  - Single Molecule Analysis (Workflow A).
  - Batch Molecular Screening with CSV upload/download and fault isolation (Workflow B).
  - Model Insights & Metrics (Workflow C).
  - Literature Intelligence (Workflow D).
- Clean scientific styling, transparent limitation disclosures, and data tables.

### 7. Multi-Layer Testing & Verification
- 37 automated tests across Unit, Integration, and End-to-End suites with 100% pass rate.
