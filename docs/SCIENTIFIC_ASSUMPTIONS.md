# Scientific Assumptions and Boundaries

**Directive Version:** 4.0  
**Domain:** Cheminformatics, Computational Toxicology, and In Silico Screening  

---

## 1. Cheminformatics & Molecular Representations
1. **2D Graph Representations vs. 3D Conformations:**
   - Input structures are processed as 2D constitutional graphs via SMILES strings.
   - Morgan circular fingerprints (radius=2, nBits=2048) capture topological connectivity up to 4 bonds away from each central atom.
   - *Limitation:* 3D spatial conformation, conformational stereodynamics, atropisomerism, and protein-ligand binding pockets are not modeled directly.
2. **Lipinski Rule of 5:**
   - Evaluates: $MW \le 500$, $LogP \le 5$, $HBD \le 5$, $HBA \le 10$.
   - *Boundary:* Compliance correlates with oral bioavailability for certain small-molecule drugs. It is strictly a physicochemical heuristic and does NOT indicate drug efficacy, lack of toxicity, or non-oral therapeutic viability.

## 2. Computational Toxicology & Dataset Semantics
1. **ClinTox Endpoint Semantics (`CT_TOX`):**
   - The ClinTox benchmark measures whether a drug candidate failed in human clinical trials due to toxicity or was approved by the FDA.
   - *Boundary:* A positive label (`CT_TOX = 1`) indicates clinical trial failure attributed to toxicity. A negative label (`CT_TOX = 0`) indicates FDA approval or lack of documented clinical failure. It does NOT establish absolute cellular safety or absence of toxic mechanisms in specific organs.
2. **Tox21 12-Endpoint Semantics:**
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

## 3. Structural Similarity and Chemical Space
1. **Tanimoto Similarity:**
   - Computed as the Jaccard index between 2048-bit Morgan bit-vectors.
   - *Boundary:* The "Molecular Similarity Principle" states that similar molecules often exert similar biological effects. However, "activity cliffs" frequently occur where minor chemical modifications completely ablate binding or provoke toxicity. Structural similarity alone must never be used to confirm therapeutic equivalence.

## 4. Explainable AI (SHAP) Interpretation
1. **TreeExplainer Feature Attributions:**
   - SHAP values quantify the contribution of individual Morgan fingerprint bits to the Random Forest model's log-odds or predicted probability.
   - *Boundary:* SHAP attributes model variance to specific mathematical bit patterns. It does NOT prove biological causality, toxicophore pharmacodynamics, or metabolic bioactivation.

## 5. Candidate Prioritization Score
1. **Composite Ranking Formula:**
   - $R = 0.5 \times (1 - P_{tox}) \times 100 + 0.3 \times D + 0.2 \times (100 \times T_{max})$
   - *Boundary:* This formula is a transparent multi-criteria triage heuristic. It is NOT an empirically validated probability of clinical success.
