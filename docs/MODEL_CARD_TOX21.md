# Model Card: Tox21 12-Endpoint Multi-Assay Classifiers

**Model Identifier:** `tox21_rf_12endpoints`  
**Model Family:** Independent Binary Random Forest Classifiers (12 Models)  
**Status:** APPROVED_FOR_DEMO  
**Date of Training:** 2026-10-03  
**Artifact Location:** `artifacts/tox21/{endpoint}_model.joblib`, `artifacts/tox21/summary.json`  

---

## 1. Intended Use
In vitro bioassay pathway activation screening across 7 Nuclear Receptor (NR) pathways and 5 Stress Response (SR) pathways from the US Toxicology in the 21st Century (Tox21) data challenge.

## 2. Dataset & Target Governance
- **Source:** Tox21 Data Challenge (NIH NCATS / EPA / NTP / FDA).
- **Assay Targets:**
  - `NR-AR`: Androgen Receptor full length
  - `NR-AR-LBD`: Androgen Receptor Ligand-Binding Domain
  - `NR-AhR`: Aryl Hydrocarbon Receptor
  - `NR-Aromatase`: Aromatase enzyme inhibition
  - `NR-ER`: Estrogen Receptor alpha full length
  - `NR-ER-LBD`: Estrogen Receptor alpha LBD
  - `NR-PPAR-gamma`: Peroxisome Proliferator-Activated Receptor gamma
  - `SR-ARE`: Nrf2 antioxidant response element
  - `SR-ATAD5`: Genotoxicity-induced DNA damage response
  - `SR-HSE`: Heat Shock Response factor
  - `SR-MMP`: Mitochondrial Membrane Potential disruption
  - `SR-p53`: p53 Tumor Suppressor pathway activation
- **Missing Data Governance:** Each endpoint was modeled independently. Records with null/unassayed labels for a given assay were strictly excluded from that model's dataset. Missing labels were **never** imputed as inactive.

## 3. Training & Featurization Protocol
- **Algorithm:** Independent `RandomForestClassifier` per endpoint.
- **Parameters:** 300 estimators, `class_weight='balanced'`, `random_state=42`.
- **Features:** 2048-bit Morgan circular fingerprints (radius=2).
- **Split Strategy:** 80/20 Stratified Random Split per endpoint.

## 4. Evaluated Test Metrics Summary

### Primary Stratified Random Split (Baseline)

| Endpoint | Total Usable Samples | Test ROC-AUC | Test PR-AUC | Balanced Accuracy | Test Positive Fraction |
|----------|----------------------|--------------|-------------|-------------------|------------------------|
| `NR-AR` | 7,258 | **0.7485** | 0.3979 | 0.6948 | 4.2% |
| `NR-AR-LBD` | 6,751 | **0.9051** | 0.5662 | 0.8248 | 3.5% |
| `NR-AhR` | 6,542 | **0.9092** | 0.6248 | 0.7712 | 11.7% |
| `NR-Aromatase` | 5,815 | **0.8546** | 0.5332 | 0.6774 | 5.1% |
| `NR-ER` | 6,186 | **0.7064** | 0.3527 | 0.6423 | 12.8% |
| `NR-ER-LBD` | 6,948 | **0.8188** | 0.4247 | 0.7259 | 5.0% |
| `NR-PPAR-gamma` | 6,443 | **0.8116** | 0.2417 | 0.5779 | 2.9% |
| `SR-ARE` | 5,825 | **0.8183** | 0.5249 | 0.6979 | 16.1% |
| `SR-ATAD5` | 7,065 | **0.9083** | 0.3892 | 0.6092 | 3.7% |
| `SR-HSE` | 6,460 | **0.8113** | 0.3081 | 0.6099 | 5.7% |
| `SR-MMP` | 5,804 | **0.8619** | 0.6092 | 0.7369 | 15.8% |
| `SR-p53` | 6,767 | **0.8510** | 0.3601 | 0.6055 | 6.2% |

### Dual Benchmark Comparison: Stratified Random vs. Bemis-Murcko Scaffold Split

Evaluated on key representative nuclear receptor and stress response pathways (stored in `artifacts/tox21/scaffold_benchmark.json`):

| Endpoint | Biological Target | Random Split ROC-AUC | Scaffold Split ROC-AUC | Random Split PR-AUC | Scaffold Split PR-AUC |
|---|---|---|---|---|---|
| `NR-AR-LBD` | Androgen Receptor (LBD) | **0.9051** | **0.7836** | 0.5662 | 0.2818 |
| `NR-AhR` | Aryl Hydrocarbon Receptor | **0.9092** | **0.8839** | 0.6248 | 0.4631 |
| `SR-MMP` | Mitochondrial Membrane Disruption | **0.8619** | **0.8488** | 0.6092 | 0.4476 |
| `SR-p53` | p53 DNA Damage Response | **0.8510** | **0.7719** | 0.3601 | 0.2641 |

## 5. Scientific Limitations & Scaffold Generalization Notice
1. **Out-of-Scaffold Performance:** In vitro pathway classification shows modest reduction on novel chemical scaffolds (e.g. `NR-AR-LBD` ROC-AUC shifts from 0.9051 to 0.7836), yet models maintain strong discriminatory power above random across all tested pathways ($ROC\text{-}AUC \ge 0.77$).
2. **In vitro cell-based reporter assay signal:** Indicates specific luciferase or enzyme perturbation, not physiological organ failure.
3. **Cytotoxicity interference:** High compound concentrations can disrupt cell viability, causing false-positive or false-negative readouts in reporter assays.
4. **Assay Noise:** Class imbalances and assay cross-reactivities in high-throughput chemical screening libraries introduce non-systematic variance.
