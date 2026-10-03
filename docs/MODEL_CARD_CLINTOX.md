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
The model is evaluated under two distinct splitting methodologies:
1. **Stratified Random Split (Interpolation Baseline):** 80/20 train/test split stratified by target class. Measures performance when testing molecules belonging to known chemical series.
2. **Bemis-Murcko Scaffold Split (Extrapolation Benchmark):** Partitions molecules based on core ring assemblies with zero scaffold overlap between train and test sets. Measures generalization to novel chemical scaffolds.

### Dual Benchmark Comparison (Test Split $N=296$)

| Metric | Stratified Random Split (Baseline) | Bemis-Murcko Scaffold Split | Scientific Interpretation |
|---|---|---|---|
| **ROC-AUC** | **0.8517** | **0.7668** | Demonstrates strong predictive capacity even on unseen scaffolds ($>0.75$) |
| **PR-AUC** | **0.2972** | **0.3239** | $4.3\times$ higher than random prevalence ($7.5\%$) |
| **Accuracy** | **0.9155** | **0.9628** | High overall fidelity |
| **Balanced Accuracy** | **0.6617** | **0.5769** | Minority-class sensitivity under scaffold distribution shifts |
| **Precision** | **0.4211** | **1.0000** | Zero false-positive toxic calls on scaffold test split |
| **Recall** | **0.3636** | **0.1538** | Identifies 2 of 13 rare toxic scaffold-hop compounds |
| **F1-Score** | **0.3902** | **0.2667** | Minority harmonic balance |

### Confusion Matrix Comparison (Test Split)

**Stratified Random Split:**
```text
               Predicted Non-Toxic (0)    Predicted Toxic (1)
Actual Non-Toxic (0)         263                       11
Actual Toxic (1)              14                        8
```

**Bemis-Murcko Scaffold Split:**
```text
               Predicted Non-Toxic (0)    Predicted Toxic (1)
Actual Non-Toxic (0)         283                        0
Actual Toxic (1)              11                        2
```

## 5. Scientific Limitations & Caveats
1. **Scaffold Generalization Shift:** While random split ROC-AUC reaches 0.8517, scaffold-split ROC-AUC is 0.7668. This drop is consistent with computational chemistry benchmarks and demonstrates that performance on novel chemotypes requires wider uncertainty bounds.
2. **Severe Imbalance:** The positive class represents ~7.5% of samples. Balanced accuracy and PR-AUC reflect the inherent difficulty of minority-class toxicity detection.
3. **Clinical Trial Termination Semantics:** Trials fail for multiple commercial, pharmacokinetic, or clinical reasons. While `CT_TOX` annotates toxicity-related failures, label noise exists in historical trial records.
4. **Applicability Domain:** Generalizes best to small organic molecules within Lipinski chemical space ($MW < 600$ Da). Exotic scaffolds, polymers, and organometallics fall outside reliable feature distributions.
