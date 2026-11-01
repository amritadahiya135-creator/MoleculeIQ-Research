# Model Card: ClinTox Calibrated Random Forest Classifier

**Model Identifier:** `clintox_rf_calibrated`  
**Model Family:** Supervised Random Forest Classifier with Platt Scaling Calibration  
**Status:** APPROVED_FOR_DEMO  
**Date of Retraining & Calibration:** October 4, 2026 (Stage 2 Scientific Hardening)  
**Artifact Locations:**
- Production Serving Model: [`artifacts/clintox/model.joblib`](../artifacts/clintox/model.joblib) (CalibratedClassifierCV)
- Base Tree Ensemble: [`artifacts/clintox/uncalibrated_model.joblib`](../artifacts/clintox/uncalibrated_model.joblib) (RandomForestClassifier for TreeSHAP)
- Training Library Fingerprints: [`artifacts/clintox/training_fingerprints.npz`](../artifacts/clintox/training_fingerprints.npz)
- Model Metadata: [`artifacts/clintox/metadata.json`](../artifacts/clintox/metadata.json)
- Dual Benchmark Record: [`artifacts/clintox/scaffold_benchmark.json`](../artifacts/clintox/scaffold_benchmark.json)

---

## 1. Intended Use
- **Primary Objective:** In silico estimation of calibrated probability that a small drug-like molecule has structural features associated with clinical trial failure attributed to toxicity.
- **Intended Audience:** Academic researchers, cheminformaticians, and computational chemists performing early-stage library triage.
- **Explicit Non-Goals:** Regulatory safety filings, clinical diagnostics, animal toxicology replacement, or definitive determination of in vivo human toxicity.

---

## 2. Dataset & Target Semantics
- **Benchmark Source:** MoleculeNet ClinTox Dataset ([`data/raw/clintox.csv`](../data/raw/clintox.csv)).
- **Target Column:** `CT_TOX` (1 = Clinical trial failure attributed to toxicity; 0 = FDA-approved / passed clinical trial).
- **Deduplication Policy:** `conservative_toxic` (Precautionary Hazard Screening).
  - Out of 1,484 raw rows, 4 invalid SMILES are excluded, leaving 1,480 valid structures.
  - Across the valid records, exactly 19 duplicate canonical SMILES groups (38 rows) possessed conflicting labels (`FDA=1, CT_TOX=0` vs `FDA=0, CT_TOX=1`).
  - Under the `conservative_toxic` policy, any molecule with documented clinical trial toxicity failure is assigned $CT\_TOX = 1$.
  - Deduplicated Population: **1,461 unique canonical compounds** (112 toxic, 1,349 non-toxic; 7.67% prevalence).
  - **Cohort Composition & Scaffold Topology:** Out of the 1,461 unique canonical compounds, **1,332 are cyclic compounds** (possessing Bemis–Murcko core ring-linker scaffolds) and **129 are acyclic compounds**. Under the deterministic generic scaffold grouping strategy (`STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC`), the 129 acyclic compounds are **not** collapsed into a single shared group; rather, they are partitioned by their generic carbon skeletons into **90 distinct generic acyclic groups** (`acyc_skeleton:<SMILES>`), comprising 71 singletons, 13 groups of size 2, 2 groups of size 3, and individual groups of sizes 4, 6, 7, and 9 (max acyclic group size: 9). Across the entire cohort, there are **873 total scaffold groups** (91 positive groups, 782 negative groups; 708 singletons, representing $81.1\%$ singleton groups).
  - **Zero Data Leakage:** Contradictory duplicate leakage across train and test splits is eliminated completely ($0$ overlapping SMILES).

---

## 3. Architecture, Calibration & Features
- **Base Algorithm:** `sklearn.ensemble.RandomForestClassifier` (300 estimators, `class_weight='balanced'`, `random_state=42`).
- **Calibration Engine:** `sklearn.calibration.CalibratedClassifierCV` (method: `sigmoid` / Platt scaling, 5-fold stratified CV on training data only).
- **Features:** 2048-bit Morgan Circular Fingerprints (ECFP4 equivalent, radius 2, binary `int8`).
- **Explainability:** TreeSHAP (`shap.TreeExplainer`) executed directly against the base tree ensemble (`uncalibrated_model.joblib`), providing exact Shapley attribution without calibration distortion.

---

## 4. Evaluation Protocol & Results

To evaluate performance across different chemical series, the ClinTox classification architecture was benchmarked under two distinct experimental protocols:

1. **Stratified Random Split (Production Serving Model Baseline):**
   - **Training Cohort:** 80% stratified random partition ($N = 1,168$ compounds; 90 toxic, 1,078 non-toxic; prevalence: $7.71\%$).
   - **Held-Out Test Cohort:** 20% stratified random partition ($N = 293$ compounds; 22 toxic, 271 non-toxic; prevalence: $7.51\%$).
   - **Model Role:** This exact model is serialized as the production serving artifact ([`artifacts/clintox/model.joblib`](../artifacts/clintox/model.joblib)). It measures **chemical series interpolation** within known chemotypic families.

2. **Bemis–Murcko Scaffold Split (Generalization Comparator):**
   - **Training Cohort:** 80% scaffold partition ($N = 1,168$ compounds; 95 toxic, 1,073 non-toxic; prevalence: $8.13\%$) grouped by Bemis–Murcko core ring-linker topologies.
   - **Held-Out Test Cohort:** 20% scaffold partition ($N = 293$ compounds; 17 toxic, 276 non-toxic; prevalence: $5.80\%$).
   - **Model Role:** **This is an independently trained comparator model evaluated on a completely different test set.** It is **not** the production serving model evaluated on novel scaffolds. It tests whether an identical Random Forest architecture can generalize when evaluated exclusively on held-out scaffold topologies with zero train/test scaffold overlap.

> [!WARNING]
> **Comparability Notice:** The random-split and scaffold-split benchmarks represent two separate models evaluated on two different test cohorts with differing positive base rates ($7.51\%$ vs. $5.80\%$). They assess distinct operational regimes and must not be conflated as interchangeable scores of a single static model.

### Dual Benchmark Comparison (Held-Out Test Partitions, $N=293$)

| Evaluation Metric | Stratified Random Split (Production Model) | Bemis–Murcko Scaffold Split (Comparator Model) | Metric Interpretation & Complementarity |
| :--- | :---: | :---: | :--- |
| **Cohort Size ($N$)** | **293** (22 pos / 271 neg; $7.51\%$) | **293** (17 pos / 276 neg; $5.80\%$) | Scaffold clustering shifts positive prevalence from $7.51\%$ to $5.80\%$ in the test set. |
| **ROC-AUC** | **0.8507** | **0.7877** | Measures global ranking across all decision thresholds; scaffold gap ($\Delta = 0.0630$) indicates harder out-of-distribution ranking. |
| **Approx. 95% CI (Hanley–McNeil)** | **[0.7484, 0.9530]** ($SE = 0.0522$) | **[0.6569, 0.9185]** ($SE = 0.0667$) | Approximate biostatistical uncertainty intervals overlap substantially; differences between point estimates should not be overinterpreted. |
| **PR-AUC (Average Precision)** | **0.5099** | **0.3994** | Evaluates precision across all recall levels; highly sensitive to minority base rate ($7.51\%$ vs. $5.80\%$). |
| **Standard Accuracy ($\tau = 0.5$)** | **0.9317** | **0.9454** | Misleadingly high due to the $>92\%$ majority non-toxic class; uninformative without sensitivity metrics. |
| **Balanced Accuracy ($\tau = 0.5$)** | **0.5663** | **0.5294** | Arithmetic mean of sensitivity and specificity at default $0.5$ threshold; close to chance on novel chemotypes. |
| **Precision ($\tau = 0.5$)** | **0.7500** (3 TP, 1 FP) | **1.0000** (1 TP, 0 FP) | Zero false alarms on novel chemotypes, but achieved only by extreme triage conservatism. |
| **Recall ($\tau = 0.5$)** | **0.1364** (3 TP, 19 FN) | **0.0588** (1 TP, 16 FN) | **Critical Limitation:** At default threshold, the model misses $94.1\%$ ($16 / 17$) of toxic compounds on novel scaffolds. |
| **F1-Score ($\tau = 0.5$)** | **0.2308** | **0.1111** | Harmonic mean severely depressed by low recall on the rare toxic class. |
| **Brier Score Loss** | **0.0522** | **0.0467** | Mean squared calibration error; lower under scaffold split due to lower positive base rate. |
| **Log Loss** | **0.1969** | **0.1841** | Cross-entropy under Platt sigmoid calibration. |
| **Expected Calibration Error (ECE)** | **0.1604** | **0.1851** | Mean absolute calibration gap across reliability bins. |

### Confusion Matrix Comparison (Held-Out Test Partitions, $\tau = 0.5$)

**Stratified Random Split (Production Serving Model on Random Test Split, $N=293$):**
```text
               Predicted Non-Toxic (0)    Predicted Toxic (1)    Total Actual
Actual Non-Toxic (0)         270                        1             271 (Specificity: 99.63%)
Actual Toxic (1)              19                        3              22 (Recall: 13.64%)
Total Predicted              289                        4             293 (Precision: 75.00%)
```

**Bemis–Murcko Scaffold Split (Comparator Model on Scaffold Test Split, $N=293$):**
```text
               Predicted Non-Toxic (0)    Predicted Toxic (1)    Total Actual
Actual Non-Toxic (0)         276                        0             276 (Specificity: 100.00%)
Actual Toxic (1)              16                        1              17 (Recall: 5.88%)
Total Predicted              292                        1             293 (Precision: 100.00%)
```

### Understanding Performance Metrics & Methodological Tradeoffs

1. **Why ROC-AUC Does Not Guarantee Safety:** A respectable ROC-AUC ($0.8507$ random, $0.7877$ scaffold) indicates that a randomly chosen toxic molecule is assigned a higher predicted probability than a randomly chosen non-toxic molecule ~79–85% of the time. However, it does not mean the model reliably identifies toxicity at any specific operational cutoff.
2. **The Out-of-Scaffold False-Negative Reality:** At the standard probability threshold of $0.5$, the scaffold-trained model identified only **1 out of 17 toxic compounds** (**$5.88\%$ recall**), failing to detect **$16$ out of $17$ toxic molecules** ($94.1\%$ false negatives). In exchange, it produced zero false positives ($100\%$ precision). This demonstrates extreme conservatism on novel scaffolds, which is hazardous if users interpret a "Non-Toxic" prediction as a confirmation of safety.
3. **Small-Sample Uncertainty in Benchmark Partitions:** With only $22$ positive cases in the random test split and $17$ in the scaffold test split, the approximate Hanley–McNeil 95% confidence intervals are wide ($[0.7484, 0.9530]$ and $[0.6569, 0.9185]$) and overlap by over $0.20$ AUC units. The observed point difference ($\Delta = 0.0630$) falls within approximately one standard error ($SE \approx 0.067$). Point estimates must not be overinterpreted as definitive proof of massive performance divergence.
4. **Metric Complementarity:**
   - **Accuracy ($>93\%$)** is inflated by the non-toxic majority and must not be used alone.
   - **Balanced Accuracy ($52.9\%$)** and **Recall ($5.9\%$)** expose severe sensitivity deficits.
   - **PR-AUC ($0.3994$)** correctly penalizes low precision-recall tradeoffs under class imbalance.

### Multi-Seed Cross-Validation & Generalization Evaluation (Priority 2D)

To quantify performance variability across different partitionings and avoid single-split artifacts, an exploratory repeated cross-validation was conducted across three prespecified seeds (`42`, `1042`, `2042`) using 5-fold cross-validation under two matched regimes on the identical deduplicated cohort ($N=1,461$, 112 positives, 1,349 negatives):

1. **Scaffold-Grouped Regime:** 5-fold cross-validation partitioned strictly by generic Bemis–Murcko scaffold clusters (`murcko_cyclic_acyclic_generic`, 873 total groups, including 90 generic acyclic skeleton groups) with verified $0\%$ cross-split group leakage.
2. **Stratified Random Reference Regime:** Matched 5-fold stratified random partitions on the identical 1,461 compounds.

Across the 3 seeds and 2 regimes, **30 model fits** were evaluated using `RandomForestClassifier(n_estimators=300, class_weight='balanced')`.

#### Aggregate Performance Summary (Across Seeds 42, 1042, 2042)

| Evaluation Metric | Scaffold-Grouped (Mean $\pm$ SD) | Stratified Random Reference (Mean $\pm$ SD) | Generalization Delta ($\Delta_{\text{scaffold} - \text{random}}$) |
| :--- | :---: | :---: | :---: |
| **Pooled Out-of-Fold ROC-AUC** | **$0.8473 \pm 0.0093$** | **$0.8410 \pm 0.0159$** | $+0.0063$ |
| **Pooled Out-of-Fold PR-AUC (Average Precision)** | **$0.3762 \pm 0.0184$** | **$0.3776 \pm 0.0334$** | $-0.0014$ |
| **PR-AUC Lift over Baseline** | $+0.2995 \pm 0.0184$ | $+0.3009 \pm 0.0334$ | $-0.0014$ |
| **Brier Score** | $0.0651 \pm 0.0009$ | $0.0662 \pm 0.0020$ | $-0.0011$ |
| **Log Loss** | $0.2644 \pm 0.0024$ | $0.2884 \pm 0.0061$ | $-0.0240$ |

*Per-seed pooled out-of-fold ROC-AUC:*
- Scaffold-grouped: Seed 42 = $0.8372$, Seed 1042 = $0.8495$, Seed 2042 = $0.8553$.
- Stratified random: Seed 42 = $0.8514$, Seed 1042 = $0.8227$, Seed 2042 = $0.8490$.

*Per-seed pooled out-of-fold PR-AUC:*
- Scaffold-grouped: Seed 42 = $0.3557$, Seed 1042 = $0.3817$, Seed 2042 = $0.3912$.
- Stratified random: Seed 42 = $0.3850$, Seed 1042 = $0.3411$, Seed 2042 = $0.4065$.

#### Scientific Interpretation of the Comparison
- **No Meaningful Advantage:** The observed small differences ($\Delta \text{ROC-AUC} = +0.0063$, $\Delta \text{PR-AUC} = -0.0014$) do **not** establish a meaningful advantage for either splitting regime, nor do they prove that scaffold generalization has been solved or is absent.
- **Sparsity Confounder:** In ClinTox, $81.1\%$ of scaffold groups are singletons ($708 / 873$). Consequently, even an unconstrained random split places $>80\%$ of test molecules on scaffolds unobserved during training. This structural sparsity dilutes the contrast between scaffold-grouped and random partitioning.

#### Uncertainty Accounting & Bootstrap Methodology Boundaries
- **Fixed-Score Cluster Block Bootstrap:** A 500-replicate scaffold-cluster block bootstrap resampling the 873 clusters with replacement was performed on the pooled out-of-fold predictions:
  - Scaffold-Grouped ROC-AUC 95% CIs: Seed 42 $[0.7916, 0.8761]$; Seed 1042 $[0.8034, 0.8875]$; Seed 2042 $[0.8176, 0.8909]$.
  - Stratified Random ROC-AUC 95% CIs: Seed 42 $[0.8048, 0.8906]$; Seed 1042 $[0.7755, 0.8627]$; Seed 2042 $[0.8015, 0.8896]$.
- **Critical Methodological Boundary:** In this protocol, the cluster bootstrap resamples fixed out-of-fold prediction scores; models are **not** refitted within each bootstrap replicate. Therefore, these confidence intervals capture test-sample finite-cluster uncertainty on fixed prediction scores, but do **not** capture training-set variability or model parameter estimation uncertainty across refits. Overlapping confidence intervals must **not** be claimed as proof of statistical equivalence or insignificance. Furthermore, folds and seeds reuse the same 112 positive compounds and are not independent biological validation cohorts.

#### Threshold Sensitivity Limitations & Operational Trade-offs
- **Standard Default Threshold ($\tau = 0.50$):**
  - Across the 15 scaffold-grouped folds, the mean sensitivity / recall is only **$19.33\%$** (mean specificity: $98.59\%$, mean precision: $53.74\%$, mean balanced accuracy: $58.97\%$).
  - At $\tau = 0.50$, the model misses over $80\%$ of toxic molecules on unseen scaffolds.
- **Exploratory Training-Prevalence Threshold ($\tau = \pi_{\text{train}} \approx 0.077$):**
  - Across the 15 scaffold-grouped folds, mean sensitivity / recall rises to **$95.53\%$**, but specificity falls to **$32.76\%$**, precision falls to **$10.60\%$**, and balanced accuracy is $64.15\%$.
  - The threshold $\tau = \pi_{\text{train}}$ is computed strictly per training fold from training labels ($y_{\text{train}}$) without test-label leakage.
  - While adjusting the threshold recovers sensitivity, precision falls to $10.60\%$ (approximately 9 false alarms for every true positive).
- **Explicit Non-Recommendation:** The training-prevalence threshold is **strictly an exploratory sensitivity diagnostic**, not a validated operating threshold. Neither $\tau = 0.50$ nor $\tau = \pi_{\text{train}}$ is recommended for real-world toxicology decisions or safety determinations.

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

## 6. Scientific Limitations & Governance Caveats

1. **Exploratory Research & Educational Scope:** MoleculeIQ models are designed strictly for early-stage hypothesis generation, prioritization, and educational demonstration. They are **NOT** validated or approved for clinical decision-making, patient triage, clinical trial design, or regulatory safety filings. Predictions are not proof of compound safety, non-toxicity, therapeutic efficacy, or clinical suitability.
2. **False-Negative Risk and Sensitivity Deficits on Novel Scaffolds:** A negative prediction ("Non-Toxic") does **not** indicate a validated determination of biological safety. In multi-seed scaffold evaluations, the model achieved only **$19.33\%$ mean sensitivity** at the standard $0.50$ threshold, failing to detect over $80\%$ of toxic molecules on unseen scaffolds (and missing $16$ of $17$ toxic benchmark molecules in single-split comparator testing). High global ROC-AUC (~0.85) reflects rank-ordering ability, but must not be confused with dependable operational sensitivity or human safety assurance.
3. **Historical Trial Context:** Labeling reflects clinical trial discontinuation records rather than everyday consumer safety. Approved drugs tested in failed oncology trials may carry toxic annotations.
4. **Severe Class Imbalance & Small Test Positive Counts:** Toxic compounds comprise only ~7.7% of ClinTox ($N_{\text{pos}} = 112$ total). Small test positive counts ($N=22$ and $N=17$) result in wide statistical uncertainty. PR-AUC, Balanced Accuracy, and calibrated Brier scores are mandatory alongside ROC-AUC.
5. **Parent Structure vs. Active Metabolites:** The model evaluates 2D molecular graphs of parent molecules directly; hepatic cytochrome P450 biotransformation, reactive intermediate formation, and phase II conjugation are not simulated.
