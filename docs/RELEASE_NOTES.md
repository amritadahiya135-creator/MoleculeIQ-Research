# MoleculeIQ Release Notes

## Version 5.0 (Research Prototype Release Candidate)

**Release Date:** 2026-10-04  
**Release Verdict:** RESEARCH PROTOTYPE RELEASE CANDIDATE  
**Directive Version:** Master Directive V5.0 (Autonomous Scientific Verification, Advanced Engineering & Release)  

---

### Highlights of Release 5.0

#### 1. Scientific & Methodological Rigor
* **ClinTox Deduplication:** Canonicalized and deduplicated the ClinTox dataset down to 1,461 unique chemical structures, eliminating duplicate SMILES and cross-split stereoisomer leakage.
* **Platt Probability Calibration:** Implemented cross-validated Platt scaling on the ClinTox classifier, achieving Brier reliability score of **0.0522**, Expected Calibration Error (ECE) of **0.1604**, and test PR-AUC of **0.5099**.
* **Bemis-Murcko Scaffold Benchmarking:** Documented realistic generalization drop between random splits and scaffold splits (ClinTox ROC-AUC drops from 0.89 to 0.771 on unseen molecular scaffolds; Tox21 scaffold benchmarks established across 12 targets).
* **Dynamic Applicability Domain:** Integrated training-library Tanimoto boundary evaluation that transparently flags compounds operating `WITHIN_DOMAIN`, `NEAR_BOUNDARY`, or `OUTSIDE_DOMAIN`.

#### 2. Core Cheminformatics Modernization
* **Modern RDKit Morgan Generator:** Refactored fingerprint generation to utilize `rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator` with bit-for-bit verified fallback to `AllChem.GetMorganFingerprintAsBitVect`, eliminating deprecation warnings.
* **C++ Parser Protection:** Added strict input length boundaries (max 4,000 characters) in SMILES validation to protect against C++ stack overflow crashes on pathologically large inputs.
* **Edge Case Verification:** Verified correct handling of zwitterions, charged species, inorganic salts, disconnected fragments, macrocycles, and stereoisomers.

#### 3. Cryptographic Artifact Integrity & MLOps
* **SHA-256 Manifest:** Generated cryptographic checksums manifest (`artifacts/checksums.sha256`) covering all 31 model `.joblib`, metadata `.json`, and reference `.npz` files.
* **Automated Tamper Detection:** Implemented `src/utils/integrity.py` with automated verification running during CI and test execution to detect tampering or silent bit rot.

#### 4. Application Security & Defensive Guardrails
* **CSV Formula Injection Defense:** Implemented cell sanitization in `src/utils/security.py` neutralizing characters that trigger spreadsheet formula/DDE execution (`=`, `+`, `-`, `@`, `\t`, `\r`) with leading single quotes.
* **Batch Screening Guardrails:** Enforced explicit limit of 5,000 compounds per batch submission with immediate, friendly user advisory.
* **Path Traversal Guards:** Implemented `safe_join_path` preventing directory traversal in file resolution.

#### 5. Release Engineering & Containerization
* **Hardened Production Dockerfile:** Multi-stage image running Python 3.12-slim, non-root `moleculeiq` application user, RDKit X11/cairo libraries, and container health checks.
* **GitHub Actions CI Workflow:** Multi-version test matrix (Python 3.11, 3.12) validating artifact hashes, unit tests, integration workflows, and performance smoke tests.

---

### Previous Releases
* **v1.0.0 (2026-10-03):** Initial demonstrable prototype with baseline Random Forest models, basic Streamlit UI, and literature assistant.
