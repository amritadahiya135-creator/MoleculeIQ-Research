# MoleculeIQ

**An AI-Assisted Molecular Screening and Literature Intelligence Platform**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/cheminformatics-RDKit-green.svg)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/ui-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-37%20passed%20(100%25)-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-demonstration%20ready-blue.svg)]()

---

## Overview

**MoleculeIQ** is a modular, research-oriented computational software system designed for early-stage molecular screening, computational toxicology triage, structural similarity exploration, and scientific literature retrieval.

Built on an auditable, layered monolithic architecture, MoleculeIQ unifies RDKit cheminformatics, benchmark Random Forest models trained on ClinTox and Tox21, SHAP TreeExplainer feature attributions, and NCBI PubMed semantic sentence retrieval into an accessible Streamlit interface.

> **SCIENTIFIC & REGULATORY NOTICE:** MoleculeIQ is an in silico screening and educational triage prototype. It does not provide regulatory toxicology filings, clinical diagnostic evaluations, or guarantees of drug safety or efficacy. All computational estimates must be verified through empirical biological assays.

---

## Key Capabilities

1. **Molecular Input & Structure Processing (Module M01)**
   - Strict SMILES validation, sanitization, canonicalization, and formula generation.
   - Preserves original input representation for complete data lineage.

2. **Physicochemical Descriptors Engine (Module M02)**
   - Calculates Molecular Weight, Crippen LogP, H-Bond Donors, H-Bond Acceptors, Rotatable Bonds, and TPSA.
   - Evaluates Lipinski Rule of 5 oral bioavailability heuristic with transparent violation notes.

3. **Predictive Toxicology Classifiers (Modules M04 & M05)**
   - **ClinTox:** Predicts clinical trial termination probability attributed to toxicity ($ROC\text{-}AUC = 0.852$).
   - **Tox21:** Predicts pathway activation across 12 independent nuclear receptor and stress-response assays ($ROC\text{-}AUC$ up to $0.909$).
   - Dynamic positive class mapping verifying `model.classes_` to eliminate probability inversion risks.

4. **Chemical Similarity & Reference Library (Module M06)**
   - High-throughput in-memory Tanimoto similarity search on 2048-bit Morgan circular fingerprints (radius=2) against a curated reference library of approved small molecules.

5. **Multi-Objective Candidate Prioritization (Module M07)**
   - Transparent, deterministic triage formula:
     $$S = (1 - P_{\text{tox}}) \times 100$$
     $$D = 100 \text{ if Lipinski pass else } 40$$
     $$M = 100 \times T_{\max}$$
     $$R = 0.5 \times S + 0.3 \times D + 0.2 \times M$$
   - Multi-parameter sensitivity analysis testing ranking stability.

6. **Explainable AI via SHAP (Module M08)**
   - TreeExplainer calculates mathematical feature contributions for active fingerprint bits.
   - 2D SVG vector depiction highlighting molecular atoms corresponding to influential bits.

7. **PubMed Literature Intelligence (Module M09)**
   - Queries NCBI Entrez E-utilities for scientific abstracts matching chemical or adverse event queries.
   - Decomposes abstracts into individual statements and computes semantic cosine relevance.
   - Preserves complete article citation provenance (PMID, title, authors, year, direct PubMed URL).

8. **Batch Screening with Fault Isolation (Workflow B)**
   - Upload compound libraries via CSV with automatic column detection.
   - Row-level fault isolation: malformed molecules are caught and reported without terminating batch execution.
   - Enriched dataset downloadable in CSV format.

---

## Architecture

The codebase adheres strictly to layered dependency principles:

```text
Presentation Layer (Streamlit UI in app.py & src/ui/)
       │
       ▼
Application Service Layer (src/services/orchestration.py)
       │
       ▼
Domain & Scientific Engines (src/chemistry, src/models, src/similarity, src/ranking, src/explainability, src/literature)
       │
       ▼
Data & Artifacts Storage (artifacts/, data/reference/, data/raw/)
```

*Domain modules never import Streamlit, ensuring 100% testability in headless CLI and CI environments.*

---

## Quickstart

### 1. Installation
```bash
git clone <repo-url>
cd moleculeiq
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```
*Expected: 37 tests passing (100% pass rate).*

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Detailed Documentation

Comprehensive technical specifications and governance documents are available in the [docs/](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/) directory:
- [Architecture Specification](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/ARCHITECTURE.md)
- [Requirements Traceability Matrix](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/REQUIREMENTS_TRACEABILITY.md)
- [Engineering Decisions (ADRs)](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/ENGINEERING_DECISIONS.md)
- [ClinTox Model Card](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/MODEL_CARD_CLINTOX.md)
- [Tox21 Model Card](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/MODEL_CARD_TOX21.md)
- [Experiment Records](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/EXPERIMENTS.md)
- [Testing Report](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/TESTING.md)
- [Known Limitations](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/KNOWN_LIMITATIONS.md)
- [Deployment Guide](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/DEPLOYMENT.md)
- [Release Notes](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/RELEASE_NOTES.md)
