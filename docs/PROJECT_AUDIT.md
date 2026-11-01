# Project Audit: MoleculeIQ

**Audit Date:** 2026-10-03  
**Auditor:** Principal Software Architect & Lead Engineering Intelligence  
**Directive Version:** 4.0  
**PRD / SRD Reference:** PRD v1.0, SRD v1.0  

---

## 1. Executive Summary

A comprehensive initial inspection was conducted on the designated project workspace at `<repository-root>`.

Key findings:
1. **Initial Repository State:** The workspace was completely uninitialized and empty (0 files, 0 folders, no Git repository).
2. **Execution Environment:** 
   - Operating System: Windows 11 (PowerShell environment)
   - Python Version: Python 3.12.6 (`python.exe`)
   - Pip Version: 24.2
3. **Core Dependencies Availability:**
   - Cheminformatics: `rdkit` (Available and functional)
   - Machine Learning & Statistics: `scikit-learn`, `numpy`, `pandas`, `joblib` (Available)
   - Explainable AI: `shap` (Available)
   - Web Presentation: `streamlit` (Available)
   - Networking & I/O: `requests` (Available)
   - NLP & Sentence Embeddings: `torch`, `sentence-transformers` (Currently Not Installed)
4. **Scope & Roadmap Alignment:** 
   - A greenfield engineering build is required, adhering strictly to the modular monolithic design outlined in the SRD.
   - All modules M01 through M10 must be built with rigorous data contracts, type annotations, unit tests, and scientific qualification disclaimers.

---

## 2. Inventory and Component Analysis

| Component / Subsystem | Current Status | Codebase File(s) | Verification Evidence |
|-----------------------|----------------|------------------|-----------------------|
| Control Plane | Initialized | `.agent/*` | Files present, tracking state |
| Documentation Base | Initialized | `docs/*` | Requirements and architecture documented |
| Chemistry Validation (M01) | Planned | `src/chemistry/validation.py` | To be implemented in Task 01 |
| Descriptor Engine (M02) | Planned | `src/chemistry/descriptors.py` | To be implemented in Task 01 |
| Fingerprint Generator (M03) | Planned | `src/chemistry/fingerprints.py` | To be implemented in Task 01 |
| ClinTox Classifier (M04) | Planned | `src/models/clintox.py` | To be implemented in Task 02 |
| Tox21 Multi-Classifier (M05) | Planned | `src/models/tox21.py` | To be implemented in Task 02 |
| Similarity Engine (M06) | Planned | `src/similarity/*` | To be implemented in Task 03 |
| Prioritization Engine (M07) | Planned | `src/ranking/scoring.py` | To be implemented in Task 04 |
| Explainability Engine (M08) | Planned | `src/explainability/*` | To be implemented in Task 05 |
| Literature Assistant (M09) | Planned | `src/literature/*` | To be implemented in Task 06 |
| Streamlit Application (M10) | Planned | `app.py`, `src/ui/*` | To be implemented in Task 08 |

---

## 3. Architectural and Technical Risks

1. **Dependency Direction Violations:**
   - *Risk:* Presentation logic (Streamlit) leaking into core scientific calculations.
   - *Mitigation:* Explicit architecture rule: Domain modules must not import `streamlit`.
2. **Missing Deep Learning / NLP Stack:**
   - *Risk:* `sentence-transformers` requires `torch`, which is large (~2GB) and may fail compilation or take excessive bandwidth.
   - *Mitigation:* Implement modular adapter for literature similarity that can use `sentence-transformers` if available, and provide a robust lexical/TF-IDF semantic cosine fallback to guarantee system resilience.
3. **Serialization Vulnerabilities:**
   - *Risk:* Insecure deserialization via joblib/pickle.
   - *Mitigation:* Only load artifacts from known local application artifact directory with accompanying cryptographic checksums and metadata validation.

---

## 4. Scientific Risks

1. **Class Imbalance in Clinical Datasets:**
   - ClinTox and Tox21 exhibit severe class imbalance (minority toxic cases). Simple accuracy is deceptive.
   - *Mitigation:* Mandatory evaluation using Balanced Accuracy, PR-AUC, and ROC-AUC with explicit confusion matrices.
2. **Over-interpretation of Similarity and Ranking:**
   - Structural similarity does not imply bioactivity equivalence. Lipinski rule compliance is an ADME heuristic, not a therapeutic score.
   - *Mitigation:* Prominent user-facing scientific disclaimers in every UI card and export file.
