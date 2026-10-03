# Experiment Tracking and Benchmark Reproducibility

**Directive Version:** 4.0  
**PRD / SRD Source:** PRD v1.0 & SRD v1.0  

---

## Experiment EXP-001: ClinTox Baseline Binary Classification

- **Target:** `CT_TOX` (1 = Clinical trial failure due to toxicity; 0 = FDA-approved)
- **Dataset:** MoleculeNet ClinTox (`data/raw/clintox.csv`)
- **Total Valid Molecules:** 1,480
- **Class Distribution:** 112 positive (7.57%), 1,368 negative (92.43%)
- **Feature Configuration:** Morgan Circular Fingerprints, radius=2, length=2048 bits
- **Algorithm:** `RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)`
- **Splitting Strategy:** Stratified random split (80% train / 20% test, `random_state=42`)
- **Evaluation on Held-Out Test Set (N=296):**
  - **ROC-AUC:** 0.8517
  - **PR-AUC (Average Precision):** 0.2972
  - **Accuracy:** 0.9155
  - **Balanced Accuracy:** 0.6617
  - **Precision:** 0.4211
  - **Recall:** 0.3636
  - **F1-Score:** 0.3902
  - **Confusion Matrix:** `[[263, 11], [14, 8]]`
- **Artifacts:** `artifacts/clintox/model.joblib`, `artifacts/clintox/metadata.json`

---

## Experiment EXP-002: Tox21 12-Endpoint In Vitro Screening Suite

- **Dataset:** US Tox21 Data Challenge (`data/raw/tox21.csv`)
- **Total Valid Molecules:** 7,823
- **Algorithm:** 12 independent `RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)`
- **Splitting Strategy:** Stratified random split per endpoint (80% train / 20% test, `random_state=42`)
- **Missing Label Handling:** Strict endpoint-level filtering; records without assay measurements were completely excluded from that endpoint's model.

### Summary Results Across All 12 Assays

| Endpoint | Pathway Category | Usable Samples | Held-Out ROC-AUC | Held-Out PR-AUC | Balanced Accuracy |
|----------|------------------|----------------|------------------|-----------------|-------------------|
| `NR-AR` | Nuclear Receptor | 7,258 | 0.7485 | 0.3979 | 0.6948 |
| `NR-AR-LBD` | Nuclear Receptor | 6,751 | 0.9051 | 0.5662 | 0.8248 |
| `NR-AhR` | Nuclear Receptor | 6,542 | 0.9092 | 0.6248 | 0.7712 |
| `NR-Aromatase` | Nuclear Receptor | 5,815 | 0.8546 | 0.5332 | 0.6774 |
| `NR-ER` | Nuclear Receptor | 6,186 | 0.7064 | 0.3527 | 0.6423 |
| `NR-ER-LBD` | Nuclear Receptor | 6,948 | 0.8188 | 0.4247 | 0.7259 |
| `NR-PPAR-gamma` | Nuclear Receptor | 6,443 | 0.8116 | 0.2417 | 0.5779 |
| `SR-ARE` | Stress Response | 5,825 | 0.8183 | 0.5249 | 0.6979 |
| `SR-ATAD5` | Stress Response | 7,065 | 0.9083 | 0.3892 | 0.6092 |
| `SR-HSE` | Stress Response | 6,460 | 0.8113 | 0.3081 | 0.6099 |
| `SR-MMP` | Stress Response | 5,804 | 0.8619 | 0.6092 | 0.7369 |
| `SR-p53` | Stress Response | 6,767 | 0.8510 | 0.3601 | 0.6055 |

- **Artifacts:** `artifacts/tox21/{endpoint}_model.joblib`, `artifacts/tox21/summary.json`
