# Scientific Assumptions and Boundaries

**Directive Version:** 4.0 / Stage 2 Scientific Hardening  
**Domain:** Cheminformatics, Computational Toxicology, and In Silico Screening  

---

## 1. Cheminformatics & Molecular Representations
1. **2D Graph Representations vs. 3D Conformations:**
   - Input structures are processed as 2D constitutional graphs via SMILES strings.
   - Morgan circular fingerprints (radius=2, nBits=2048) capture topological connectivity up to 4 bonds away from each central atom.
   - *Limitation:* 3D spatial conformation, conformational stereodynamics, atropisomerism, and protein-ligand binding pockets are not modeled directly.
2. **Lipinski Rule of 5 & Heteroatom Definitions:**
   - Evaluates: $MW \le 500$, $LogP \le 5$, $HBD \le 5$, $HBA \le 10$.
   - *H-Bond Acceptor (HBA) Definition:* Evaluated via RDKit `Lipinski.NumHAcceptors(mol)`. RDKit adheres strictly to Lipinski's physiological definition, counting basic nitrogen and oxygen atoms while explicitly **excluding amide nitrogens** whose lone pairs participate in resonance delocalization with adjacent carbonyl groups. For example, Caffeine ($C_8H_{10}N_4O_2$) has 3 physiological acceptors (2 carbonyl oxygens + 1 basic imidazole nitrogen), distinct from naive elemental heteroatom counting ($N+O=6$).
   - *Boundary:* Compliance correlates with oral bioavailability for certain small-molecule drugs. It is strictly a physicochemical heuristic and does NOT indicate drug efficacy, lack of toxicity, or non-oral therapeutic viability.

---

## 2. Computational Toxicology & Dataset Semantics
1. **ClinTox Endpoint Semantics (`CT_TOX`):**
   - The ClinTox benchmark measures whether a drug candidate failed in human clinical trials due to toxicity or was approved by the FDA.
   - *Boundary:* A positive label (`CT_TOX = 1`) indicates clinical trial failure attributed to toxicity. A negative label (`CT_TOX = 0`) indicates FDA approval or lack of documented clinical failure. It does NOT establish absolute cellular safety or absence of toxic mechanisms in specific organs.
2. **Deterministic Deduplication & Precautionary Conflict Resolution:**
   - Inspection of raw ClinTox revealed 19 duplicate canonical structures (38 rows) entered twice with conflicting labels (`FDA=1, CT_TOX=0` vs `FDA=0, CT_TOX=1`).
   - Under the `conservative_toxic` policy, any molecule with documented clinical trial toxicity failure is resolved to $CT\_TOX = 1$. This adheres to the precautionary hazard screening principle in regulatory toxicology: an approved drug with a trial termination event due to systemic toxicity represents an active safety risk.
   - Deduplication completely eliminates train/test cross-split contamination.
3. **Probability Calibration via Platt Scaling:**
   - Uncalibrated Random Forest probabilities with `class_weight='balanced'` artificially shift tree priors to $\approx 50\%$, causing raw voting fractions to overstate true failure probabilities.
   - Platt scaling (sigmoid calibration) fitted strictly via 5-fold cross-validation on training data scales outputs monotonically to align with the true empirical base-rate ($7.67\%$).
   - Brier score loss drops from $0.0789$ to $0.0522$ and Expected Calibration Error is reduced to $0.1604$.
4. **Tox21 12-Endpoint Semantics:**
   - Measures in vitro luciferase reporter gene assay activity across 12 specific stress-response pathways (SR) and nuclear receptor pathways (NR):
     - NR-AR, NR-AR-LBD: Androgen receptor activation / ligand-binding domain.
     - NR-AhR: Aryl hydrocarbon receptor.
     - NR-Aromatase: Inhibition of aromatase enzyme.
     - NR-ER, NR-ER-LBD: Estrogen receptor activation.
     - NR-PPAR-gamma: Peroxisome proliferator-activated receptor gamma.
     - SR-ARE: Nrf2 antioxidant response element.
     - SR-ATAD5: Genotoxicity-induced DNA damage response.
     - SR-HSE: Heat shock response.
     - SR-MMP: Mitochondrial membrane potential disruption.
     - SR-p53: Tumor suppressor p53 pathway activation.
   - *Boundary:* Assays reflect cell-based in vitro luminescence perturbation. Positive assay response indicates pathway perturbation under assay conditions, not clinical organ toxicity or in vivo carcinogenicity. Missing labels signify unassayed compound concentrations or assay artifact exclusion, not negative activity.

---

## 3. Dynamic Applicability Domain & Chemical Space Boundaries
1. **Distance-to-Training-Library Metric:**
   - Evaluated as maximum Tanimoto similarity against the 1,168 unique training set Morgan fingerprints.
   - Empirical leave-one-out distribution on training molecules demonstrates that the 10th percentile nearest-neighbor similarity is $0.2568$ (median $0.5472$).
   - **Thresholds:**
     - $\ge 0.40$: Within Applicability Domain (`WITHIN_DOMAIN`). High structural neighborhood representation.
     - $0.25 - 0.40$: Near Applicability Domain Boundary (`NEAR_BOUNDARY`). Low analogue density; warning issued.
     - $< 0.25$: Outside Applicability Domain (`OUTSIDE_DOMAIN`). Novel chemotype or alien scaffold; explicit warning that prediction is an extrapolation.
   - *Boundary:* Belonging to the applicability domain indicates that the molecule shares topological substructures with training data; it does NOT guarantee prediction correctness.

---

## 4. Explainable AI (SHAP) Interpretation
1. **TreeExplainer Feature Attributions:**
   - SHAP values quantify the contribution of individual Morgan fingerprint bits to the base Random Forest model's log-odds.
   - Executed against `uncalibrated_model.joblib` to preserve exact tree attribution dynamics without calibration distortion.
   - *Boundary:* SHAP attributes model variance to specific mathematical bit patterns. It does NOT prove biological causality, toxicophore pharmacodynamics, or metabolic bioactivation.

---

## 5. Candidate Prioritization Score
1. **Composite Ranking Formula:**
   - $R = 0.5 \times (1 - P_{tox}) \times 100 + 0.3 \times D + 0.2 \times (100 \times T_{max})$
   - *Boundary:* This formula is a transparent multi-criteria triage heuristic. It is NOT an empirically validated probability of clinical success.
