# Model Card: Tox21 12-Endpoint Multi-Assay Classifiers

**Model Identifier:** `tox21_rf_12endpoints`<br>
**Model Family:** Independent Binary Random Forest Classifiers (12 Models)<br>
**Status:** APPROVED_FOR_DEMO<br>
**Date of Training:** 2026-10-03<br>
**Artifact Location:** `artifacts/tox21/{endpoint}_model.joblib`, `artifacts/tox21/summary.json`<br>

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

---

## 6. Assay-Specific Applicability Domain Evaluation (Phase A.2)

### 6.1 Reference Corpus Derivation & Partition Integrity
- **Corpus Source:** Extracted deterministically from `data/raw/tox21.csv` and persisted in `artifacts/tox21/training_fingerprints.npz`.
- **Missing-Label Isolation:** Each of the 12 endpoints is assessed against only those compounds that entered that specific endpoint's model training partition. Compounds with null/unassayed (`NaN`) records are excluded from that endpoint's reference corpus.
- **Zero-Leakage Guarantee:** The held-out evaluation/test split (20% stratified random test partition, `random_state=42`) is strictly excluded from all endpoint reference corpora.

### 6.2 Featurization & Distance Metric
- **Fingerprint Representation:** 2048-bit Morgan circular fingerprints (radius $r=2$), identical to model featurization.
- **Similarity Metric:** Maximum Tanimoto similarity to the endpoint's training partition:
  $$\text{AD\_score}_e(q) = \max_{j \in \text{train}_e} \frac{q \cdot x_{e, j}}{\|q\|_1 + \|x_{e, j}\|_1 - q \cdot x_{e, j}}$$

### 6.3 Status & Threshold Governance
- **Unvalidated Threshold State:** Categorical decision boundaries (such as ClinTox's 0.40/0.25 thresholds) are **not** transferred to Tox21. In-vitro pathway perturbation assays operate under diverse receptor geometries and binding-site topologies that lack empirical validation for uniform categorical cutoffs.
- **Domain Status:** Reported as `UNVALIDATED` for valid structures, and `UNAVAILABLE` for invalid or missing structures.
- **Mandatory Scientific Disclaimer:** A high Tanimoto similarity indicates structural representation in the assay training corpus; it does **not** establish biological inactivity, in-vitro assay safety, or clinical harmlessness.

---

## 7. Post-Hoc Probability Calibration & Reliability Benchmark (Phase B.1 — Schema v1.1.0)

### 7.1 Calibration Architecture, 3-Way Disjoint Method Selection & Zero-Leakage Protocol
- **Motivation:** Because the 12 deployed Tox21 Random Forest classifiers use `class_weight='balanced'` on skewed assays ($2.89\%\text{–}16.17\%$ positive prevalence), raw tree-vote probabilities systematically overestimate empirical positive base rates (macro-average 10-bin weighted Expected Calibration Error $\text{ECE} = 0.0618$).
- **Leakage-Safe 3-Way Disjoint Method Selection Protocol (`5_fold_3way_disjoint_base60_cal20_val20`):**
  1. For each endpoint, the exact 80/20 stratified train/test split (`random_state=42`) used during model training is reconstructed from `data/raw/tox21.csv`. Held-out 20% test samples ($X_{\text{test}}, y_{\text{test}}$) are **never** accessed during method selection or calibrator fitting.
  2. To avoid the cross-fold label coupling that arises if a single precomputed OOF vector is split a second time for method selection, the 80% training partition $D_{\text{train}}$ is partitioned into 5 deterministic stratified folds $F_0, \dots, F_4$. In each rotation $k \in \{0..4\}$, three **pairwise disjoint** index subsets are formed:
     - `base_rf_idx` ($60\%$ of $D_{\text{train}}$, 3 folds): trains fold Random Forest $M_{\text{base}}^{(k)}$.
     - `cal_fit_idx` ($20\%$ of $D_{\text{train}}$, 1 fold): receives out-of-sample raw predictions from $M_{\text{base}}^{(k)}$ and fits candidate Sigmoid and Isotonic calibrators ($C_{\text{sig}}^{(k)}, C_{\text{iso}}^{(k)}$) using `y_train[cal_fit_idx]`.
     - `val_idx` ($20\%$ of $D_{\text{train}}$, 1 fold): receives out-of-sample raw predictions from $M_{\text{base}}^{(k)}$ and evaluates $C_{\text{sig}}^{(k)}$ and $C_{\text{iso}}^{(k)}$ against untouched labels `y_train[val_idx]`.
  3. **Base-Estimator Sample-Size Nuance & Conservative Sigmoid Fallback:** In 3-way disjoint selection, $M_{\text{base}}^{(k)}$ is trained on $60\%$ of $D_{\text{train}}$ (compared to $80\%$ for final OOF fitting and $100\%$ for the deployed model; Pearson correlation between $60\%$ and $80\%$ out-of-sample training probabilities is $r = 0.8885\text{–}0.9707$, mean absolute difference $0.0177\text{–}0.0473$), and each disjoint `cal_fit_idx` fold contains $29\text{–}150$ positive examples ($< 200$ minimum per-fold positives required for stable non-parametric step-function estimation). Across all 12 endpoints on the 3-way disjoint validation benchmark, Sigmoid achieved superior Log Loss over Isotonic (`12/12` endpoints) and superior or tied Brier score on `11/12` endpoints, selecting **`sigmoid` across all 12 endpoints**. (An earlier exploratory v1.0.0 run that cross-validated a precomputed OOF vector has been superseded by v1.1.0.)
- **Final Calibrator Fitting (`5_fold_stratified_out_of_fold_posthoc`):**
  - Final persisted calibrator parameters (`calibrator_parameters.sigmoid`, $P(y=1|f) = [1 + \exp(a f + b)]^{-1}$ with $a < 0$, and exploratory `calibrator_parameters.isotonic`) are fitted on 5-fold stratified OOF predictions across $100\%$ of $D_{\text{train}}$ and evaluated once on the locked 20% test partition $D_{\text{test}}$.
- **Runtime Artifact Hardening:** `Tox21CalibrationService` enforces exact schema version matching (`schema_version == "1.1.0"`, returning `UNSUPPORTED_SCHEMA` otherwise), endpoint identity, valid 64-hex `dataset_sha256` and `base_model_sha256` provenance, cryptographic SHA-256 verification against the deployed `{endpoint}_model.joblib` (`MODEL_HASH_MISMATCH` on mismatch), and strict finiteness/monotonicity checks (`CORRUPT_ARTIFACT` / `ERROR` on `NaN`, `±inf`, non-negative Sigmoid slope $a \ge 0$, or non-monotonic Isotonic thresholds), always preserving valid raw predictions on fallback.

### 7.2 Aggregate Held-Out Test Reliability Benchmark (12 Endpoints, $N_{\text{test, total}} = 15,577$)

*Metrics definitions:* 10-bin sample-weighted ECE uses 10 equal-width bins across $[0.0, 1.0]$ (`[0.0, 0.1), ..., [0.9, 1.0]`). PR-AUC is computed via `sklearn.metrics.average_precision_score`. Paired 95% bootstrap confidence intervals resample individual test-compound observations with replacement ($B = 500$, `random_seed=42`).

| Aggregation | Method | Brier Score ($\downarrow$) | Log Loss ($\downarrow$) | 10-Bin Weighted ECE ($\downarrow$) | ROC-AUC ($\uparrow$) | PR-AUC ($\uparrow$) |
|---|---|---|---|---|---|---|
| **Macro-Average** | **Uncalibrated RF (Raw)** | 0.0598 | 0.2461 | 0.0618 | **0.8337** | **0.4444** |
| **Macro-Average** | **Sigmoid (Selected — 5-Fold OOF)** | **0.0523** (-12.5%) | **0.1916** (-22.1%) | **0.0134** (-78.3%) | **0.8337** | **0.4444** |
| **Macro-Average** | **Isotonic (Exploratory — 5-Fold OOF)** | 0.0523 | 0.1926 | 0.0121 | 0.8322 | 0.4218 |
| **Sample-Weighted** | **Uncalibrated RF (Raw)** | 0.0582 | 0.2422 | 0.0606 | **0.8339** | **0.4413** |
| **Sample-Weighted** | **Sigmoid (Selected — 5-Fold OOF)** | **0.0509** (-12.5%) | **0.1874** (-22.6%) | **0.0131** (-78.4%) | **0.8339** | **0.4413** |
| **Sample-Weighted** | **Isotonic (Exploratory — 5-Fold OOF)** | 0.0509 | 0.1881 | 0.0118 | 0.8324 | 0.4192 |

### 7.3 Per-Endpoint 3-Way Selection Validation & Locked Held-Out Test Results

| Endpoint | $N_{\text{train}}$ ($\text{Pos}$) | $N_{\text{test}}$ ($\text{Pos}$) | Selected Method | 3-Way Val Brier (Raw $\rightarrow$ Sig / Iso) | 3-Way Val LogLoss (Raw $\rightarrow$ Sig / Iso) | Locked Test Brier: Raw $\rightarrow$ Sig ($\Delta$, 95% Paired Bootstrap CI) | Test LogLoss: Raw $\rightarrow$ Sig | Test ECE: Raw $\rightarrow$ Sig | Test ROC-AUC / PR-AUC (Preserved) |
|---|---|---|---|---|---|---|---|---|---|
| `NR-AR` | 5,806 (246) | 1,452 (62) | `sigmoid` | 0.0326 $\rightarrow$ **0.0269** / 0.0268 | 0.2061 $\rightarrow$ **0.1207** / 0.1824 | 0.0375 $\rightarrow$ **0.0299** (-0.0076, [-0.0101, -0.0052])\* | 0.2630 $\rightarrow$ **0.1333** | 0.0382 $\rightarrow$ **0.0102** | 0.7485 / 0.3979 |
| `NR-AR-LBD` | 5,400 (190) | 1,351 (47) | `sigmoid` | 0.0227 $\rightarrow$ **0.0195** / 0.0200 | 0.1557 $\rightarrow$ **0.0898** / 0.1424 | 0.0206 $\rightarrow$ **0.0175** (-0.0031, [-0.0048, -0.0011])\* | 0.1400 $\rightarrow$ **0.0782** | 0.0328 $\rightarrow$ **0.0081** | 0.9051 / 0.5662 |
| `NR-AhR` | 5,233 (614) | 1,309 (154) | `sigmoid` | 0.0793 $\rightarrow$ **0.0697** / 0.0706 | 0.2721 $\rightarrow$ **0.2387** / 0.2696 | 0.0769 $\rightarrow$ **0.0653** (-0.0117, [-0.0156, -0.0077])\* | 0.2621 $\rightarrow$ **0.2216** | 0.0883 $\rightarrow$ **0.0208** | 0.9092 / 0.6248 |
| `NR-Aromatase` | 4,652 (240) | 1,163 (60) | `sigmoid` | 0.0468 $\rightarrow$ **0.0411** / 0.0414 | 0.2198 $\rightarrow$ **0.1656** / 0.2125 | 0.0386 $\rightarrow$ **0.0340** (-0.0046, [-0.0079, -0.0012])\* | 0.2138 $\rightarrow$ **0.1391** | 0.0547 $\rightarrow$ **0.0203** | 0.8546 / 0.5332 |
| `NR-ER` | 4,948 (633) | 1,238 (158) | `sigmoid` | 0.1149 $\rightarrow$ **0.0967** / 0.0984 | 0.4041 $\rightarrow$ **0.3347** / 0.4028 | 0.1240 $\rightarrow$ **0.0996** (-0.0245, [-0.0311, -0.0181])\* | 0.4125 $\rightarrow$ **0.3422** | 0.1183 $\rightarrow$ **0.0155** | 0.7064 / 0.3527 |
| `NR-ER-LBD` | 5,558 (279) | 1,390 (70) | `sigmoid` | 0.0433 $\rightarrow$ **0.0368** / 0.0381 | 0.2022 $\rightarrow$ **0.1512** / 0.2019 | 0.0409 $\rightarrow$ **0.0349** (-0.0060, [-0.0094, -0.0028])\* | 0.2124 $\rightarrow$ **0.1433** | 0.0471 $\rightarrow$ **0.0080** | 0.8188 / 0.4247 |
| `NR-PPAR-gamma` | 5,154 (149) | 1,289 (37) | `sigmoid` | 0.0271 $\rightarrow$ **0.0250** / 0.0253 | 0.1291 $\rightarrow$ **0.1093** / 0.1241 | 0.0265 $\rightarrow$ **0.0251** (-0.0014, [-0.0033, +0.0009])† | 0.1393 $\rightarrow$ **0.1073** | 0.0295 $\rightarrow$ **0.0072** | 0.8116 / 0.2417 |
| `SR-ARE` | 4,660 (754) | 1,165 (188) | `sigmoid` | 0.1232 $\rightarrow$ **0.1125** / 0.1139 | 0.4053 $\rightarrow$ **0.3682** / 0.4322 | 0.1188 $\rightarrow$ **0.1035** (-0.0152, [-0.0204, -0.0101])\* | 0.4110 $\rightarrow$ **0.3428** | 0.1082 $\rightarrow$ **0.0199** | 0.8183 / 0.5249 |
| `SR-ATAD5` | 5,652 (211) | 1,413 (53) | `sigmoid` | 0.0355 $\rightarrow$ **0.0325** / 0.0326 | 0.1541 $\rightarrow$ **0.1354** / 0.1741 | 0.0303 $\rightarrow$ **0.0296** (-0.0007, [-0.0030, +0.0017])† | 0.1246 $\rightarrow$ **0.1195** | 0.0418 $\rightarrow$ **0.0102** | 0.9083 / 0.3892 |
| `SR-HSE` | 5,168 (298) | 1,292 (74) | `sigmoid` | 0.0551 $\rightarrow$ **0.0504** / 0.0505 | 0.2576 $\rightarrow$ **0.1957** / 0.2138 | 0.0504 $\rightarrow$ **0.0464** (-0.0040, [-0.0069, -0.0008])\* | 0.2455 $\rightarrow$ **0.1812** | 0.0523 $\rightarrow$ **0.0089** | 0.8113 / 0.3081 |
| `SR-MMP` | 4,643 (734) | 1,161 (184) | `sigmoid` | 0.0973 $\rightarrow$ **0.0874** / 0.0886 | 0.3402 $\rightarrow$ **0.2941** / 0.3424 | 0.0998 $\rightarrow$ **0.0919** (-0.0079, [-0.0121, -0.0032])\* | 0.3314 $\rightarrow$ **0.3046** | 0.0772 $\rightarrow$ **0.0219** | 0.8619 / 0.6092 |
| `SR-p53` | 5,413 (338) | 1,354 (85) | `sigmoid` | 0.0545 $\rightarrow$ **0.0503** / 0.0503 | 0.2121 $\rightarrow$ **0.1899** / 0.2458 | 0.0533 $\rightarrow$ **0.0496** (-0.0037, [-0.0070, -0.0007])\* | 0.1975 $\rightarrow$ **0.1857** | 0.0534 $\rightarrow$ **0.0094** | 0.8510 / 0.3601 |

*\*Statistically supported Brier improvement (95% paired bootstrap CI strictly below 0.0, observed on 10/12 endpoints).*<br>
*†Point-estimate Brier, LogLoss, and ECE improvement observed, but 95% paired bootstrap CI for $\Delta\text{Brier}$ includes 0.0 due to low test-set positive counts (`NR-PPAR-gamma`: 37 test positives; `SR-ATAD5`: 53 test positives).*

### 7.4 Production Release Policy & Screening Threshold Governance
- **Default Production Behavior Unchanged:** Raw uncalibrated Random Forest probabilities (`probability`), binary activity flags (`predicted_class`, `is_active` at $\ge 0.5$), `active_endpoints_count`, and composite candidate prioritization scores ($R = 0.5S + 0.3D + 0.2M$) remain **100% unchanged by default**.
- **Distinction Between Probability Calibration and Binary Classification Thresholds:** Calibrated probabilities align with the empirical assay base rate ($2.9\%\text{–}16.2\%$). Applying an unadjusted $0.50$ cutoff to calibrated probabilities lowers binary sensitivity on rare-positive endpoints. Therefore, raw `class_weight='balanced'` scores remain the default binary alerting signal (`is_active`), while calibrated probabilities (`calibrated_probability`, `calibration_method`, `calibration_status`) are exposed as opt-in empirical likelihood estimates.
- **Scientific Boundaries:** Probability calibration aligns model scores with historical Tox21 assay frequencies within the training distribution; it does **not** establish clinical safety, causal mechanism, prospective wet-lab performance, or reliable predictions outside an endpoint's chemical applicability domain.
