# MoleculeIQ — Phase B.3 Stage 2: Scaffold Generalization Diagnostic Report

> **Audit Status:** COMPLETED | **Stage 2 Verdict:** PASS
> **Generated:** 2026-10-13T15:46:33Z | **Execution Runtime:** 3212.3s (53.54 min)
> **Platform:** Windows-11-10.0.22631-SP0 | **Python:** 3.12.6

---

## 1. Executive Summary & Scientific Scope

This report provides an independent, reproducible evaluation of out-of-scaffold generalization and robustness for **MoleculeIQ's** core predictive engines (ClinTox clinical toxicity and 12 Tox21 assays).

### Key Architectural Safeguards Enforced
1. **Zero Production Mutation:** Deployed production models (`artifacts/clintox/model.joblib`, `artifacts/tox21/*.joblib`), probability calibrators, existing benchmark manifests, and checksums (`artifacts/checksums.sha256`) remained 100% untouched.
2. **Strict Missing-Label Isolation:** Missing Tox21 assay labels (`NaN`) were isolated per endpoint without imputation.
3. **Zero Split Leakage:** Every fold partition was audited with `validate_split_integrity`, confirming 0.0% group overlap, 0.0% exact canonical SMILES overlap, and 100% index disjointness.
4. **Scientific Claim Boundary:** This internal scaffold holdout benchmark quantifies algorithmic structural interpolation vs. extrapolation; it does **NOT** establish external dataset validity, clinical safety, or regulatory readiness.

---

## 2. Master Results Table: Random vs. Scaffold Cross-Validation

The table below compares 5-fold Random Stratified CV with 5-fold Scaffold Grouped CV (`murcko_cyclic_acyclic_generic`). Differences ($\Delta = \text{Scaffold} - \text{Random}$) and 95% scaffold-cluster bootstrap intervals are reported on matched out-of-fold predictions.

| Endpoint / Dataset | N | Positives (%) | Random ROC-AUC | Scaffold ROC-AUC | Δ ROC-AUC [95% CI] | Random PR-AUC | Scaffold PR-AUC | Δ PR-AUC [95% CI] | Random Brier | Scaffold Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ClinTox (CT_TOX)** | 1461 | 112 (7.7%) | 0.8481 | 0.8272 | **-0.0208** [-0.0566, +0.0146] | 0.3804 | 0.3472 | **-0.0333** [-0.1423, +0.0572] | 0.0658 | 0.0668 |
| **Tox21 NR-AR** | 7258 | 308 (4.2%) | 0.7929 | 0.7595 | **-0.0333** [-0.0622, -0.0106] | 0.4678 | 0.4408 | **-0.0270** [-0.0836, +0.0274] | 0.0329 | 0.0322 |
| **Tox21 NR-AR-LBD** | 6751 | 237 (3.5%) | 0.8729 | 0.8377 | **-0.0352** [-0.0629, -0.0132] | 0.6158 | 0.5340 | **-0.0818** [-0.1550, -0.0130] | 0.0211 | 0.0229 |
| **Tox21 NR-AhR** | 6542 | 768 (11.7%) | 0.8920 | 0.8548 | **-0.0372** [-0.0576, -0.0108] | 0.6033 | 0.4791 | **-0.1243** [-0.1589, -0.0477] | 0.0776 | 0.0954 |
| **Tox21 NR-Aromatase** | 5815 | 300 (5.2%) | 0.8215 | 0.7382 | **-0.0833** [-0.1340, -0.0246] | 0.3749 | 0.2097 | **-0.1652** [-0.2471, -0.0623] | 0.0437 | 0.0539 |
| **Tox21 NR-ER** | 6186 | 791 (12.8%) | 0.7234 | 0.7009 | **-0.0225** [-0.0418, +0.0024] | 0.3651 | 0.3046 | **-0.0605** [-0.0937, -0.0019] | 0.1164 | 0.1260 |
| **Tox21 NR-ER-LBD** | 6948 | 349 (5.0%) | 0.8166 | 0.7817 | **-0.0349** [-0.0551, -0.0059] | 0.4300 | 0.3327 | **-0.0973** [-0.1531, +0.0035] | 0.0414 | 0.0477 |
| **Tox21 NR-PPAR-gamma** | 6443 | 186 (2.9%) | 0.8278 | 0.8091 | **-0.0186** [-0.0358, +0.0045] | 0.3078 | 0.2492 | **-0.0586** [-0.1163, +0.0119] | 0.0252 | 0.0275 |
| **Tox21 SR-ARE** | 5825 | 942 (16.2%) | 0.8069 | 0.7527 | **-0.0542** [-0.0873, -0.0136] | 0.5143 | 0.3832 | **-0.1310** [-0.1879, -0.0374] | 0.1161 | 0.1385 |
| **Tox21 SR-ATAD5** | 7065 | 264 (3.7%) | 0.8563 | 0.8030 | **-0.0533** [-0.0869, -0.0145] | 0.2939 | 0.2026 | **-0.0913** [-0.1527, -0.0133] | 0.0335 | 0.0383 |
| **Tox21 SR-HSE** | 6460 | 372 (5.8%) | 0.7813 | 0.7261 | **-0.0553** [-0.0805, -0.0186] | 0.2616 | 0.2180 | **-0.0436** [-0.0821, +0.0228] | 0.0535 | 0.0599 |
| **Tox21 SR-MMP** | 5804 | 918 (15.8%) | 0.8787 | 0.8068 | **-0.0719** [-0.1123, -0.0165] | 0.6489 | 0.4295 | **-0.2194** [-0.2861, -0.0437] | 0.0937 | 0.1295 |
| **Tox21 SR-p53** | 6767 | 423 (6.3%) | 0.8533 | 0.7893 | **-0.0640** [-0.1030, -0.0225] | 0.4043 | 0.2463 | **-0.1581** [-0.2249, -0.0558] | 0.0513 | 0.0637 |

---

## 3. ClinTox Multi-Seed Repeat Evaluation (Seeds 42, 1042, 2042)

To measure partition-assignment variance, ClinTox was evaluated across three prespecified random seeds. Repeated predictions are reported separately rather than pooled as artificially independent samples.

| Regime | Metric | Seed 42 | Seed 1042 | Seed 2042 | Mean ± SD | Range [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scaffold Grouped** | ROC_AUC | 0.8272 | 0.8495 | 0.8510 | **0.8426 ± 0.0133** | [0.8272, 0.8510] |
| **Random Stratified** | ROC_AUC | 0.8481 | 0.8250 | 0.8469 | **0.8400 ± 0.0130** | [0.8250, 0.8481] |
| **Scaffold Grouped** | PR_AUC | 0.3472 | 0.3824 | 0.3811 | **0.3702 ± 0.0200** | [0.3472, 0.3824] |
| **Random Stratified** | PR_AUC | 0.3804 | 0.3409 | 0.4000 | **0.3738 ± 0.0301** | [0.3409, 0.4000] |
| **Scaffold Grouped** | PR_AUC_LIFT | 0.2705 | 0.3058 | 0.3044 | **0.2936 ± 0.0200** | [0.2705, 0.3058] |
| **Random Stratified** | PR_AUC_LIFT | 0.3038 | 0.2642 | 0.3233 | **0.2971 ± 0.0301** | [0.2642, 0.3233] |
| **Scaffold Grouped** | BRIER_SCORE | 0.0668 | 0.0645 | 0.0646 | **0.0653 ± 0.0013** | [0.0645, 0.0668] |
| **Random Stratified** | BRIER_SCORE | 0.0658 | 0.0686 | 0.0657 | **0.0667 ± 0.0017** | [0.0657, 0.0686] |
| **Scaffold Grouped** | LOG_LOSS | 0.2693 | 0.2627 | 0.2637 | **0.2653 ± 0.0036** | [0.2627, 0.2693] |
| **Random Stratified** | LOG_LOSS | 0.2870 | 0.3159 | 0.2852 | **0.2960 ± 0.0172** | [0.2852, 0.3159] |

---

## 4. Applicability Domain Stratification Findings

For each fold in scaffold cross-validation, test compounds were stratified by maximum Morgan-fingerprint Tanimoto similarity to that fold's training set. This directly validates the relationship between chemical novelty and predictive degradation.

| Endpoint | High Sim (≥0.70) ROC-AUC [N] | Moderate Sim (0.40–0.70) ROC-AUC [N] | Low Sim (<0.40) ROC-AUC [N] | Observed Trend |
| :--- | :--- | :--- | :--- | :--- |
| **ClinTox (CT_TOX)** | 0.8632 (N=107) | 0.8812 (N=615) | 0.7753 (N=739) | Steep drop at low similarity |
| **Tox21 NR-AR** | 0.7528 (N=859) | 0.8055 (N=4282) | 0.6020 (N=2117) | Steep drop at low similarity |
| **Tox21 NR-AhR** | 0.9667 (N=757) | 0.8664 (N=3780) | 0.7809 (N=2005) | Steep drop at low similarity |
| **Tox21 SR-MMP** | 0.9077 (N=634) | 0.8018 (N=3289) | 0.7651 (N=1881) | Steep drop at low similarity |
| **Tox21 SR-p53** | 0.8870 (N=775) | 0.8134 (N=3942) | 0.7018 (N=2050) | Steep drop at low similarity |

---

## 5. Group Distribution & Fold Feasibility Diagnostics

- **Primary Grouping Strategy (`murcko_cyclic_acyclic_generic`):**
  - ClinTox: 489 unique groups, largest group is benzene (cyc_murcko:c1ccccc1) with 86 molecules (5.89% < 20.0%). Zero oversized groups.
  - Tox21 (10 of 12 endpoints): Max group size is <20.0% of cohort.
  - Tox21 SR-HSE (20.56%) and SR-ARE (21.01%): Benzene slightly exceeds the nominal 20.0% boundary. However, `StratifiedGroupKFold` successfully assigned all groups with 0.0% leakage and balanced positive class counts across folds.
- **Secondary Grouping Sensitivity (`generic_murcko_framework`):**
  - ClinTox: 379 unique groups, max group (C1CCCCC1) is 185 molecules (12.66% < 20.0%). 5-fold CV feasible.
  - Tox21: In Tox21, generic framework collapses all single-ring aromatics into cyclohexane (`C1CCCCC1`), creating an oversized group comprising 23.8%–26.4% of each assay cohort (>20.0%). This confirms that generic framework grouping is structurally infeasible for strict 5-fold scaffold-disjoint cross-validation on Tox21 without group splitting.

---

## 6. Scientific Limitations & Interpretation Boundary

1. **Internal Holdout vs. External Validation:** Cross-validation holdout on scaffolds assesses model behavior when evaluating molecules whose core cyclic or acyclic framework was held out during training. It does **not** evaluate domain shifts arising from divergent synthetic libraries, alternative assay platforms, or true prospective screening.
2. **Rare Positive Assays:** On assays with low positive counts (e.g. SR-HSE, SR-p53), positive compounds are concentrated in few scaffolds. Cluster bootstrap resamples show wide confidence intervals, indicating that performance on unseen scaffolds carries substantial uncertainty.
3. **Non-Clinical Disclaimer:** MoleculeIQ is an early-stage in-silico triage tool. Negative predictions on out-of-scaffold compounds must never be interpreted as guarantees of non-toxicity or clinical safety.

---

**Stage 2 Audit Complete.** Awaiting user authorization before Phase B.3 Stage 3 (UI integration / Model Insights).