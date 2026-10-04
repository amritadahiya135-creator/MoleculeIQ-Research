# MoleculeIQ

**An AI-Assisted Molecular Screening and Literature Intelligence Platform**

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/cheminformatics-RDKit%202024-green.svg)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/ui-Streamlit-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/Dockerfile)
[![Status](https://img.shields.io/badge/status-Research%20Prototype%20Release%20Candidate%20(v5.0)-brightgreen.svg)]()

---

## Overview

**MoleculeIQ** is a modular, research-oriented computational software system designed for early-stage molecular screening, calibrated computational toxicology triage, structural similarity exploration, and scientific literature retrieval.

Built on an auditable, layered monolithic architecture, MoleculeIQ unifies RDKit cheminformatics, Platt-calibrated Random Forest models trained on ClinTox and Tox21, dynamic chemical applicability domain evaluation, SHAP TreeExplainer feature attributions, and NCBI PubMed semantic sentence retrieval into an accessible, reactive Streamlit interface.

> **SCIENTIFIC & REGULATORY NOTICE:** MoleculeIQ is an in silico screening and research triage prototype. It does not provide regulatory toxicology filings, clinical diagnostic evaluations, or guarantees of drug safety or efficacy. All computational estimates must be verified through empirical biological assays.

---

## Key Capabilities

1. **Molecular Input & Chemical Correctness (Module M01)**
   - Strict SMILES validation, sanitization, canonicalization, and formula generation.
   - 4,000-character input ceiling protecting C++ parser against recursion/stack overflow.
   - Validated on zwitterions, salts, stereoisomers, macrocycles, and unprintable characters.

2. **Physicochemical Descriptors Engine (Module M02)**
   - Calculates Molecular Weight, Crippen LogP, H-Bond Donors, H-Bond Acceptors, Rotatable Bonds, and TPSA.
   - Evaluates Lipinski Rule of 5 oral bioavailability heuristic with transparent violation notes.
   - Physiological HBA definition accounting for non-basic delocalized nitrogens.

3. **Predictive Toxicology Classifiers & Calibration (Modules M04 & M05)**
   - **ClinTox:** Deduplicated training set (1,461 compounds, zero cross-split leakage). Calibrated via Platt scaling achieving Brier reliability score of **0.0522**, ECE of **0.1604**, and test PR-AUC of **0.5099** ($ROC\text{-}AUC = 0.852$).
   - **Dynamic Applicability Domain:** Continuous Tanimoto distance computation against 1,461 training fingerprints categorizes query compounds into `WITHIN_DOMAIN`, `NEAR_BOUNDARY`, and `OUTSIDE_DOMAIN`.
   - **Tox21:** Predicts pathway activation across 12 independent nuclear receptor and stress-response assays ($ROC\text{-}AUC$ up to $0.909$).

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
   - Queries NCBI Entrez E-utilities with Windows certificate store integration for secure TLS.
   - Decomposes abstracts into individual statements and computes semantic cosine relevance.
   - Preserves complete citation provenance (PMID, title, authors, year, direct PubMed URL).

8. **Batch Screening & Security Guardrails (Workflow B)**
   - Enforces 5,000 compound maximum batch size guardrail.
   - CSV Formula Injection defense (CWE-1236) neutralizing Excel/Sheets macro triggers (`=`, `+`, `-`, `@`).
   - Row-level fault isolation: malformed molecules are isolated without terminating batch processing.

---

## Quickstart

### 1. Local Native Installation
```bash
git clone https://github.com/moleculeiq/moleculeiq.git
cd moleculeiq
python -m venv .venv
# Activate: source .venv/bin/activate (Linux/macOS) or .venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 2. Verify Cryptographic Integrity (SHA-256)
```bash
python -c "from src.utils.integrity import verify_artifact_checksums; res = verify_artifact_checksums(); print(res); assert res.is_valid"
```

### 3. Run Automated Tests
```bash
python -m pytest tests/unit/ tests/integration/ tests/e2e/ -v
```

### 4. Launch Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Docker Deployment
```bash
docker build -t moleculeiq:latest .
docker run -d -p 8501:8501 --name moleculeiq-app moleculeiq:latest
```

---

## Comprehensive Documentation

* [Installation & Deployment Guide](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/INSTALLATION.md)
* [Demonstration & Evaluation Walkthrough](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/DEMO_GUIDE.md)
* [Final Scientific & ML Validation Report](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/FINAL_SCIENTIFIC_VALIDATION_REPORT.md)
* [Master Engineering Baseline](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/MASTER_ENGINEERING_BASELINE.md)
* [Release Readiness Report (v5.0)](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/RELEASE_READINESS_REPORT.md)
* [Independent Release Audit](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/INDEPENDENT_RELEASE_AUDIT.md)
* [Architecture Specification](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/ARCHITECTURE.md)
* [ClinTox Model Card](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/MODEL_CARD_CLINTOX.md)
* [Tox21 Model Card](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/MODEL_CARD_TOX21.md)
* [Release Notes](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/RELEASE_NOTES.md)
