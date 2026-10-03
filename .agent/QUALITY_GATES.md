# Quality Gates

A phase or task cannot be marked as **VERIFIED** or approved for integration until the following verifiable conditions are met:

### Gate 1: Core Cheminformatics Engine (M01, M02, M03)
- [ ] 100% test pass rate on unit tests for SMILES validation, canonicalization, and syntax checking.
- [ ] Known benchmark molecules (Ethanol, Aspirin, Caffeine, Atorvastatin) match validated physicochemical descriptors within standard floating-point tolerances ($10^{-3}$).
- [ ] Morgan fingerprint generator outputs deterministic 2048-bit bit vectors with radius=2.
- [ ] Malformed, empty, and non-chemical string inputs fail safely with explicit, informative error objects.

### Gate 2: Machine Learning Models (M04, M05)
- [ ] Datasets (ClinTox, Tox21) verified for label integrity, missing values, and class distribution.
- [ ] Stratified 80/20 train/test split executed with seed=42; test sets strictly isolated from training and hyperparameter choices.
- [ ] Random Forest baseline models trained with 300 estimators and balanced class weighting.
- [ ] Full evaluation metrics computed on held-out test data (ROC-AUC, PR-AUC, Balanced Accuracy, Precision, Recall, F1, Confusion Matrix).
- [ ] Serialized model artifacts saved alongside complete JSON metadata (dataset checksum, hyperparams, feature dim, evaluation metrics, timestamp).
- [ ] Positive class index verified explicitly against `model.classes_`.

### Gate 3: Similarity & Prioritization (M06, M07)
- [ ] Reference dataset cached locally with pre-calculated fingerprints.
- [ ] Tanimoto similarity function satisfies identity property ($T(A,A)=1$), symmetry ($T(A,B)=T(B,A)$), and bounds ($0 \le T(A,B) \le 1$).
- [ ] Ranking formula $R = 0.5S + 0.3D + 0.2M$ verified numerically against test vectors (e.g., $P_{\mathrm{tox}}=0.20$, Lipinski=Pass, $M=0.75 \implies R=85.0$).
- [ ] Edge cases (missing toxicity, unparseable molecule) handled with explicit incomplete status rather than imputed fake scores.

### Gate 4: Explainability & Literature (M08, M09)
- [ ] SHAP TreeExplainer computes feature contributions on Morgan fingerprint bits without runtime exceptions.
- [ ] Top active bits mapped to Morgan atom environments with highlighted molecular substructures.
- [ ] Literature module queries PubMed via NCBI E-utilities, handles network failures gracefully, and ranks retrieved sentences with citation metadata preserved.

### Gate 5: Application & End-to-End Integration (M10, Workflows A-E)
- [ ] All 4 application modes accessible via Streamlit without crashes.
- [ ] Single-molecule workflow executes end-to-end within 5 seconds for cached models.
- [ ] Batch screening processes multi-row CSV, isolates bad rows without terminating, and produces complete CSV export.
- [ ] All scientific claims accompanied by explicit limitation disclaimers.
