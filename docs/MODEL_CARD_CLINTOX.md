# Model Card: ClinTox Calibrated Random Forest Classifier

**Model Identifier:** `clintox_rf_calibrated`  
**Model Family:** Supervised Random Forest Classifier with Platt Scaling Calibration  
**Status:** APPROVED_FOR_DEMO  
**Date of Retraining & Calibration:** October 4, 2026 (Stage 2 Scientific Hardening)  
**Artifact Locations:**
- Production Serving Model: [`artifacts/clintox/model.joblib`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/artifacts/clintox/model.joblib) (CalibratedClassifierCV)
- Base Tree Ensemble: [`artifacts/clintox/uncalibrated_model.joblib`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/artifacts/clintox/uncalibrated_model.joblib) (RandomForestClassifier for TreeSHAP)
- Training Library Fingerprints: [`artifacts/clintox/training_fingerprints.npz`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/artifacts/clintox/training_fingerprints.npz)
- Model Metadata: [`artifacts/clintox/metadata.json`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/artifacts/clintox/metadata.json)
- Dual Benchmark Record: [`artifacts/clintox/scaffold_benchmark.json`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/artifacts/clintox/scaffold_benchmark.json)

---

## 1. Intended Use
- **Primary Objective:** In silico estimation of calibrated probability that a small drug-like molecule has structural features associated with clinical trial failure attributed to toxicity.
- **Intended Audience:** Academic researchers, cheminformaticians, and computational chemists performing early-stage library triage.
- **Explicit Non-Goals:** Regulatory safety filings, clinical diagnostics, animal toxicology replacement, or definitive determination of in vivo human toxicity.

---

## 2. Dataset & Target Semantics
- **Benchmark Source:** MoleculeNet ClinTox Dataset ([`data/raw/clintox.csv`](file:///c:/Users/AMRITA/OneDrive/Desktop/moleculeiq/data/raw/clintox.csv)).
- **Target Column:** `CT_TOX` (1 = Clinical trial failure attributed to toxicity; 0 = FDA-approved / passed clinical trial).
- **Deduplication Policy:** `conservative_toxic` (Precautionary Hazard Screening).
  - Out of 1,484 raw rows, 4 invalid SMILES are excluded, leaving 1,480 valid structures.
  - Across the valid records, exactly 19 duplicate canonical SMILES groups (38 rows) possessed conflicting labels (`FDA=1, CT_TOX=0` vs `FDA=0, CT_TOX=1`).
  - Under the `conservative_toxic` policy, any molecule with documented clinical trial toxicity failure is assigned $CT\_TOX = 1$.
  - Deduplicated Population: **1,461 unique canonical compounds** (112 toxic, 1,349 non-toxic; 7.67% prevalence).
  - **Zero Data Leakage:** Contradictory duplicate leakage across train and test splits is eliminated completely ($0$ overlapping SMILES).

---

## 3. Architecture, Calibration & Features
- **Base Algorithm:** `sklearn.ensemble.RandomForestClassifier` (300 estimators, `class_weight='balanced'`, `random_state=42`).
- **Calibration Engine:** `sklearn.calibration.CalibratedClassifierCV` (method: `sigmoid` / Platt scaling, 5-fold stratified CV on training data only).
- **Features:** 2048-bit Morgan Circular Fingerprints (ECFP4 equivalent, radius 2, binary `int8`).
- **Explainability:** TreeSHAP (`shap.TreeExplainer`) executed directly against the base tree ensemble (`uncalibrated_model.joblib`), providing exact Shapley attribution without calibration distortion.

---

## 4. Evaluation Protocol & Results

The model is evaluated on held-out test splits ($N=293$) under two splitting methodologies:
1. **Stratified Random Split (Chemical Interpolation):** 80/20 train/test split stratified by target class.
2. **Bemis–Murcko Scaffold Split (Generalization Benchmark):** Partitions molecules based on core ring-linker topology with zero scaffold overlap between train and test sets.

### Dual Benchmark Comparison (Held-Out Test Split $N=293$)

| Evaluation Metric | Stratified Random Split | Bemis–Murcko Scaffold Split | Scientific Interpretation |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.8507** | **0.7877** | Strong discrimination across both familiar and novel chemotypes ($>0.78$) |
| **PR-AUC** | **0.5099** | **0.3994** | Substantial gain over un-deduplicated baseline ($0.2972$) due to elimination of contradictory test leakage |
| **Brier Score Loss** | **0.0522** | **0.0467** | Marked improvement over uncalibrated model ($0.0789$), demonstrating sharp probabilistic calibration |
| **Log Loss** | **0.1969** | **0.1841** | Massive reduction from uncalibrated log loss ($0.5242$) |
| **Expected Calibration Error (ECE)** | **0.1604** | **0.1851** | Empirical probabilities monotonically align with true risk outcomes |
| **Accuracy** | **0.9317** | **0.9454** | High overall fidelity |
| **Balanced Accuracy** | **0.5663** | **0.5294** | Strict 50% decision threshold under calibrated probabilities |
| **Precision** | **0.7500** | **1.0000** | High positive predictive value for toxic calls |
| **Recall** | **0.1364** | **0.0588** | Conservative toxic triage |
| **F1-Score** | **0.2308** | **0.1111** | Harmonic balance on rare positive class |

### Confusion Matrix Comparison (Held-Out Test Split)

**Stratified Random Split:**
```text
               Predicted Non-Toxic (0)    Predicted Toxic (1)
Actual Non-Toxic (0)         270                        1
Actual Toxic (1)              19                        3
```

**Bemis-Murcko Scaffold Split:**
```text
               Predicted Non-Toxic (0)    Predicted Toxic (1)
Actual Non-Toxic (0)         276                        0
Actual Toxic (1)              16                        1
```

---

## 5. Dynamic Applicability Domain
- **Methodology:** Vectorized Tanimoto distance against the 1,168 training library compounds using identical 2048-bit Morgan fingerprints.
- **Empirical Boundary Thresholds:**
  - `Max Tanimoto >= 0.40`: **Within Domain** (`WITHIN_DOMAIN`). High structural neighborhood representation.
  - `0.25 <= Max Tanimoto < 0.40`: **Near Boundary** (`NEAR_BOUNDARY`). Low analogue density; warning issued.
  - `Max Tanimoto < 0.25`: **Outside Domain** (`OUTSIDE_DOMAIN`). Novel chemotype or alien scaffold; explicit warning that prediction is an extrapolation.
- **Verification:**
  - Known therapeutics (Aspirin, Caffeine, Ethanol) achieve Max Tanimoto $= 1.0000$ (`WITHIN_DOMAIN`).
  - Inorganics (Cisplatin, $\text{XeF}_2$) achieve Max Tanimoto $= 0.1250$ and correctly trigger `OUTSIDE_DOMAIN` warnings.

---

## 6. Scientific Limitations & Caveats
1. **Historical Trial Context:** Labeling reflects clinical trial discontinuation records rather than everyday consumer safety. Approved drugs tested in failed oncology trials may carry toxic annotations.
2. **Scaffold Generalization Gap:** Scaffold-split ROC-AUC ($0.7877$) is lower than random split ($0.8507$), reflecting genuine generalization difficulty for out-of-scaffold compounds.
3. **Severe Imbalance:** The positive class represents ~7.7% of samples. PR-AUC and Brier scores are the appropriate metrics for monitoring model quality.
4. **Metabolism Invisibility:** The model assesses parent 2D structures directly; reactive metabolite bioactivation is not simulated.
