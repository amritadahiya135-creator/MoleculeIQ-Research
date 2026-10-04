# MoleculeIQ — Project State

**Current Status:** STAGE 2 SCIENTIFIC HARDENING & ML VALIDATION COMPLETE (Verified & Passing)  
**Directive Version:** 2.0 (Stage 2 Scientific Hardening Directive) / 4.0 (Autonomous Engineering)  
**PRD / SRD Version:** 1.0  
**Audit Identifiers:**
- Architecture Audit: `AUDIT-REV-2026-01` (Pass With Conditions)
- Scientific & ML Audit: `AUDIT-SCI-ML-2026-01` (Pass With Conditions / Scientifically Sound with Methodological Caveats)
**Stage 2 Completion Date:** October 4, 2026  
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
- [x] Independent Scientific and Machine Learning Audit Completed ([`docs/INDEPENDENT_SCIENTIFIC_ML_AUDIT.md`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/docs/INDEPENDENT_SCIENTIFIC_ML_AUDIT.md), Commit `94dc49e`).
- [x] **Stage 2 Scientific Hardening & ML Validation Completed:**
  - [x] **Task 1: ClinTox Data Integrity & Duplicate Resolution:**
    - Resolved 19 duplicate canonical structures (38 rows) with conflicting labels via deterministic `conservative_toxic` policy ([`src/models/clintox.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/models/clintox.py)).
    - Completely eliminated cross-split identical molecule leakage ($0$ overlapping SMILES).
    - Added `audit_clintox_duplicates()` helper with comprehensive provenance tracking.
    - Added unit test suite in [`tests/unit/test_clintox_data.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/tests/unit/test_clintox_data.py).
  - [x] **Task 2: ClinTox Probability Calibration via Platt Scaling:**
    - Integrated leak-free `CalibratedClassifierCV(method="sigmoid", cv=5)` on training partition only ([`src/models/clintox.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/models/clintox.py)).
    - Reduced Brier score loss from $0.0789$ to $0.0522$ and Log Loss from $0.5242$ to $0.1969$.
    - Increased PR-AUC from $0.2972$ to $0.5099$ by removing contradictory test leakage.
    - Preserved uncalibrated tree ensemble in `artifacts/clintox/uncalibrated_model.joblib` for TreeSHAP compatibility.
    - Added unit test suite in [`tests/unit/test_calibration.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/tests/unit/test_calibration.py).
  - [x] **Task 3: Dynamic Applicability Domain & Boundary Detection:**
    - Vectorized Tanimoto distance computation against 1,168 training library fingerprints saved in `artifacts/clintox/training_fingerprints.npz` ([`src/models/inference.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/models/inference.py)).
    - Empirical thresholds justified via training distribution ($\ge 0.40$ within domain, $0.25 - 0.40$ near boundary, $< 0.25$ outside domain).
    - Populates `applicability_domain_status`, `applicability_score`, and `applicability_warning` in `ClinToxPrediction`.
    - Tested and verified on Cisplatin and $\text{XeF}_2$ triggering explicit extrapolation warnings.
    - Updated UI card in [`src/ui/single_molecule.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/src/ui/single_molecule.py).
    - Added unit test suite in [`tests/unit/test_applicability_domain.py`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/tests/unit/test_applicability_domain.py).
- [x] Automated Test Suite Expanded — **66/66 passing (100% pass rate in 169.60s)**.

---

## 2. Verified Workflows
- **Workflow A (Single Molecule Analysis):** End-to-end verified with Aspirin, Caffeine, Paracetamol, Atorvastatin, Metformin, and OOD structures (Cisplatin, $\text{XeF}_2$) displaying calibrated probabilities and applicability domain badges.
- **Workflow B (Batch Screening):** End-to-end verified with mixed valid/invalid CSV inputs, row-level fault isolation, ranking, and enriched export.
- **Workflow C (Model Insights):** Verified with held-out metrics (ROC-AUC, PR-AUC, Brier score, confusion matrices) and dual random/scaffold benchmark comparisons.
- **Workflow D (Literature Intelligence):** Verified with secure TLS verification against NCBI PubMed search, abstract segmentation, and semantic cosine relevance ranking.
- **Workflow E (Candidate Prioritization):** Verified with formula $R = 0.5S + 0.3D + 0.2M$ and weight sensitivity analysis.

---

## 3. Stage 2 Quality Gates Status

| Acceptance Gate | Status | Supporting Evidence |
| :--- | :---: | :--- |
| **Audit Traceability** | **PASS** | Tasks 1, 2, and 3 map directly to findings `AUDIT-DATA-01`, `AUDIT-ML-02`, and `AUDIT-APP-01`. |
| **Data Integrity** | **PASS** | Deduplication resolved 19 conflicting groups; $0$ identical molecules leak across splits; verified in `test_clintox_data.py`. |
| **Calibration** | **PASS** | Leakage-safe Platt scaling fitted on training CV; Brier score $0.0522$, log loss $0.1969$; verified in `test_calibration.py`. |
| **Applicability Domain** | **PASS** | Max Tanimoto against 1,168 training library fingerprints; thresholds justified by 10th percentile; verified in `test_applicability_domain.py`. |
| **Testing** | **PASS** | 66/66 tests passing in 169.60s with zero skips, zero failures, and zero xfails. |
| **Integration** | **PASS** | Streamlit UI updated with domain badges and calibrated captions; TreeSHAP verified on Aspirin. |
| **Reproducibility** | **PASS** | Deterministic seeds ($42$), artifact checksums, metadata bundles, and `training_fingerprints.npz` saved. |
| **Documentation** | **PASS** | Updated `MODEL_CARD_CLINTOX.md`, `SCIENTIFIC_ASSUMPTIONS.md`, `KNOWN_LIMITATIONS.md`, and `.agent/PROJECT_STATE.md`. |
| **Independent Self-Review** | **PASS** | Verified zero leakage, backward compatibility, and absence of synthetic or fabricated metrics. |

---

## 4. Recommended Next Phase
- **Phase:** **Stage 3 Engineering Release and System Demonstration**
- **Action:** Await user review and authorization before proceeding to final release packaging.
