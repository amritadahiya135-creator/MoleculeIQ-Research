# Risk Register

**Project:** MoleculeIQ  
**Directive Version:** 4.0  

---

| Risk ID | Category | Description | Likelihood | Impact | Detectability | Mitigation Strategy | Residual Risk |
|---------|----------|-------------|------------|--------|---------------|---------------------|---------------|
| R-01 | Scientific | Severe class imbalance in ClinTox leads to uninformative model if standard accuracy is used | High | High | High | Train with balanced class weights (`class_weight='balanced'`); evaluate via PR-AUC, ROC-AUC, Balanced Accuracy, and Confusion Matrices. | Low |
| R-02 | Scientific | Toxic activity cliff: structurally similar molecule exhibits radically different toxicity profile | Medium | High | Low | Clearly disclose in UI that Tanimoto similarity is a structural heuristic, not biological proof. | Medium (Documented) |
| R-03 | Architectural | Leaking Streamlit UI code into scientific modeling or chemistry modules | Medium | High | High | Architectural boundary rules enforced by tests; domain modules must never import `streamlit`. | Low |
| R-04 | Technical | Missing PyTorch in runtime prevents heavy sentence transformer execution | High | Medium | High | Dual-engine literature ranker: primary neural transformer with fallback to TF-IDF cosine semantic engine. | Low |
| R-05 | Operational | NCBI E-utilities or ChEMBL REST endpoints throttling or timing out during evaluation | Medium | Medium | High | Bundle offline curated reference library; add timeouts, retries, and informative user status messages. | Low |
| R-06 | Security | Malicious or corrupted CSV file uploaded during batch screening | Medium | Medium | High | Strict schema validation, file size bounds, sanitized parsing, and row-level fault isolation. | Low |
| R-07 | Cheminformatics | Stereochemical information lost or corrupted during SMILES canonicalization | Medium | Medium | Medium | Retain original SMILES, use `isomericSmiles=True` during canonicalization, and flag unrecognized stereocenters. | Low |
| R-08 | Modeling | Inverted prediction probability due to class index ordering mismatch | Low | Critical | High | Explicitly verify and save `model.classes_` in artifact metadata; dynamically look up index of class `1`. | Very Low |
