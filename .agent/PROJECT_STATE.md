# MoleculeIQ — Project State

**Current Status:** INDEPENDENT SCIENTIFIC & ML AUDIT COMPLETED (Pass With Conditions)  
**Directive Version:** 2.0 (Scientific Validation) / 4.0 (Autonomous Engineering)  
**PRD / SRD Version:** 1.0  
**Architecture Audit Identifier:** `AUDIT-REV-2026-01` (Pass With Conditions)  
**Scientific & ML Audit Identifier:** `AUDIT-SCI-ML-2026-01` (Pass With Conditions / Scientifically Sound with Methodological Caveats)  
**Audit Completion Date:** October 4, 2026  
**Environment:** Windows 11 Enterprise, Python 3.12.6, Git initialized  

---

## 1. Completed Milestones
- [x] Initial workspace audit conducted and documented.
- [x] Engineering Control Plane established (`.agent/`).
- [x] Complete technical and scientific documentation suite established (`docs/`).
- [x] Module M01: Molecular Input and Structure Identity Engine ([`src/chemistry/validation.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/chemistry/validation.py)).
- [x] Module M02: Molecular Descriptor Calculation Engine ([`src/chemistry/descriptors.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/chemistry/descriptors.py)).
- [x] Module M03: Molecular Fingerprint Generator ([`src/chemistry/fingerprints.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/chemistry/fingerprints.py)).
- [x] Module M04: ClinTox Ingestion, Preprocessing & Random Forest Baseline ([`src/models/clintox.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/models/clintox.py)).
- [x] Module M05: Tox21 12-Endpoint Ingestion, Multi-Model RF Classifiers ([`src/models/tox21.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/models/tox21.py)).
- [x] Module M06: Chemical Similarity Search Engine & Approved Reference Cache ([`src/similarity/`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/similarity/)).
- [x] Module M07: Candidate Prioritization & Ranking Engine ([`src/ranking/`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/ranking/)).
- [x] Module M08: Explainability Engine via SHAP TreeExplainer & Substructure Mapping ([`src/explainability/`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/explainability/)).
- [x] Module M09: PubMed Literature Retrieval & Semantic Cosine Ranking ([`src/literature/`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/literature/)).
- [x] Application Service Layer & DTO Data Contracts ([`src/services/`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/services/)).
- [x] Module M10: Interactive Streamlit Scientific Workspace ([`app.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/app.py), [`src/ui/`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/ui/)).
- [x] Independent Architecture Review Completed ([`docs/INDEPENDENT_ARCHITECTURE_REVIEW.md`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/INDEPENDENT_ARCHITECTURE_REVIEW.md)).
- [x] Stage 1 Remediation Tasks 1–4 Completed and Verified (Commits `33e7304`, `f44148b`).
- [x] Automated Test Suite Expanded — 52/52 passing (100% pass rate in 158.28s).
- [x] **Independent Scientific and Machine Learning Audit Completed** ([`docs/INDEPENDENT_SCIENTIFIC_ML_AUDIT.md`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/INDEPENDENT_SCIENTIFIC_ML_AUDIT.md)).

---

## 2. Scientific & ML Audit Scope Inspected
- Raw datasets: [`data/raw/clintox.csv`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/data/raw/clintox.csv) and [`data/raw/tox21.csv`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/data/raw/tox21.csv).
- Preprocessing and feature extraction pipelines: RDKit sanitization, 2D physicochemical descriptors, 2048-bit Morgan ECFP4 fingerprints.
- Splitting methodologies: Stratified random split vs. Bemis–Murcko scaffold split.
- Model architecture & serialization: Random Forest classifiers, class weighting, deterministic serialization.
- Inference consistency and DTO validation: [`src/models/inference.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/models/inference.py).
- Explainability: TreeSHAP bit attributions and atomic environment mappings.
- Candidate ranking & similarity: Multi-objective linear composite scoring, Tanimoto similarity against 20 reference drugs.
- Model cards & documentation: [`docs/MODEL_CARD_CLINTOX.md`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/MODEL_CARD_CLINTOX.md), [`docs/MODEL_CARD_TOX21.md`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/MODEL_CARD_TOX21.md), [`docs/SCIENTIFIC_ASSUMPTIONS.md`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/SCIENTIFIC_ASSUMPTIONS.md), [`docs/KNOWN_LIMITATIONS.md`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/KNOWN_LIMITATIONS.md).

---

## 3. Empirical Experiments Actually Executed
1. **EXP-01 (Dataset Provenance & Duplicate Profiling):** Analyzed 1,484 ClinTox and 7,831 Tox21 rows. Detected 19 duplicate SMILES pairs with opposite labels in ClinTox; 0 duplicates in Tox21.
2. **EXP-02 (Data Leakage & Split Contamination):** Verified that 9 identical molecules leak across train/test in random split. Verified 0 identical molecules leak in scaffold split.
3. **EXP-03 (Multi-Seed Stability & Variance):** Evaluated ClinTox across seeds 42, 123, 456, 789, 999. Measured Random ROC-AUC ($0.8306 \pm 0.0510$) vs. Scaffold ROC-AUC ($0.7789 \pm 0.0125$).
4. **EXP-04 (Calibration & Probability Bias):** Calculated Brier score loss ($0.0789$) and 5-bin calibration curve on ClinTox test set. Detected risk score inflation caused by `class_weight='balanced'`.
5. **EXP-05 (Tox21 12-Endpoint Performance):** Evaluated missingness ($7.2\% - 25.8\%$), positive rates ($2.9\% - 16.2\%$), and individual endpoint scaffold drops ($0.05 - 0.12$).
6. **EXP-06 (Ranking Weight Sensitivity):** Evaluated 10 candidates across 5 weight regimes; confirmed top candidate invariance and high Spearman rank correlation ($\rho \ge 0.9152$).
7. **EXP-07 (Pipeline Parity & Ground Truth Check):** Verified bit-exact output match ($|\Delta P| < 10^{-4}$) between `ToxicityInferenceService` and direct models. Confirmed Aspirin row 846 ground truth ($CT\_TOX=1$).
8. **EXP-08 (Applicability Domain on OOD Molecules):** Submitted Cisplatin and $\text{XeF}_2$. Identified that inorganic structures pass RDKit sanitization without triggering dynamic applicability domain warnings.

---

## 4. Audit Findings Summary

| Severity | Count | Finding IDs | Short Description |
| :--- | :---: | :--- | :--- |
| **Critical** | 0 | — | No critical defects invalidating core architecture |
| **High** | 3 | `AUDIT-DATA-01` | Contradictory duplicate labels in raw ClinTox dataset |
| | | `AUDIT-ML-01` | High test metric variance across random split seeds |
| | | `AUDIT-ML-02` | Uncalibrated probability inflation from balanced weighting |
| **Medium** | 2 | `AUDIT-APP-01` | Missing distance-based applicability warning for OOD molecules |
| | | `AUDIT-DATA-02` | Ground-truth label noise for established therapeutics (Aspirin) |
| **Low** | 1 | `AUDIT-RANK-01` | Linear composite candidate ranking is an uncalibrated triage heuristic |

---

## 5. Unresolved Questions & Outstanding Validation Work
- **Deduplication Policy:** Whether to resolve contradictory duplicates in ClinTox via conservative labeling ($CT\_TOX=1$) or drop conflicting pairs entirely during preprocessing.
- **Calibration Method Selection:** Empirical comparison between isotonic regression and Platt scaling (`CalibratedClassifierCV`) on the imbalanced ClinTox dataset.
- **Applicability Domain Metric:** Determining the optimal distance metric (minimum training Tanimoto distance vs. PCA/k-NN embedding radius) for triggering out-of-domain warnings in `ToxicityInferenceService`.

---

## 6. Recommended Next Phase
- **Phase:** **Stage 2 Remediation: Scientific and Machine Learning Hardening**
- **Action:** Await user review and authorization before initiating remediation tasks.
