# Model Card: ClinTox Random Forest Classifier

**Model Identifier:** `clintox_rf_baseline`  
**Model Family:** Supervised Random Forest Classifier  
**Status:** APPROVED_FOR_DEMO  
**Date of Training:** 2026-10-03  
**Artifact Location:** `artifacts/clintox/model.joblib`, `artifacts/clintox/metadata.json`  

---

## 1. Intended Use
- **Primary Objective:** In silico estimation of whether a small drug-like molecule has structural features associated with clinical trial termination attributed to toxicity.
- **Intended Audience:** Academic researchers, cheminformaticians, and computational chemists performing early-stage triage.
- **Explicit Non-Goals:** Regulatory safety filings, clinical diagnostics, animal toxicology replacement, or absolute determination of human toxicity.

## 2. Dataset & Target Semantics
- **Benchmark Source:** MoleculeNet ClinTox Dataset.
- **Target Column:** `CT_TOX` (1 = Clinical trial failure attributed to toxicity; 0 = FDA-approved / passed clinical trial).
- **Training Population:** 1,480 valid chemical structures after RDKit syntax validation and canonicalization.
- **Class Distribution:** 112 positive (toxic) vs. 1,368 negative (non-toxic). Class positive fraction: ~7.57%.

## 3. Architecture & Features
- **Algorithm:** `sklearn.ensemble.RandomForestClassifier`
- **Estimators:** 300
- **Class Weighting:** `balanced` (mitigates minority class penalty)
- **Random Seed:** 42 (deterministic training)
- **Features:** Morgan Circular Fingerprints (ECFP4 equivalent)
  - Radius: 2
  - Bit length: 2048
  - Type: Binary bit-vector (`int8`)

## 4. Evaluation Protocol & Results
Evaluated on a held-out stratified test set (20% of data, $N=296$ test samples, 22 positive, 274 negative):

| Metric | Measured Value |
|--------|----------------|
| ROC-AUC | **0.8517** |
| PR-AUC (Average Precision) | **0.2972** |
| Accuracy | **0.9155** |
| Balanced Accuracy | **0.6617** |
| Precision | **0.4211** |
| Recall | **0.3636** |
| F1-Score | **0.3902** |

### Confusion Matrix (Test Split)
```text
               Predicted Non-Toxic (0)    Predicted Toxic (1)
Actual Non-Toxic (0)         263                       11
Actual Toxic (1)              14                        8
```

## 5. Scientific Limitations & Caveats
1. **Severe Imbalance:** The positive class represents ~7.5% of samples. Balanced accuracy (0.6617) and PR-AUC (0.2972) reflect the inherent difficulty of minority-class toxicity detection.
2. **Clinical Trial Termination:** Trials fail for multiple commercial, pharmacokinetic, or clinical reasons. While `CT_TOX` annotates toxicity-related failures, label noise exists in historical trial records.
3. **Applicability Domain:** Generalizes best to small organic molecules within the Lipinski chemical space ($MW < 600$ Da). Novel scaffolds, polymers, and metallo-compounds fall outside reliable feature distributions.
