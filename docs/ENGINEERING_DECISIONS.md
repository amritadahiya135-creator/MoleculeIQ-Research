# Engineering Decisions & ADRs

**Project:** MoleculeIQ  
**Format:** Architectural Decision Record (ADR)  

---

## ADR-001: Modular Monolithic Architecture over Microservices
- **Context:** MoleculeIQ is an academic and research-oriented molecular screening prototype requiring interactive responsiveness, rapid iteration, and easy local demonstration without complex DevOps orchestration.
- **Decision:** Build as a single Python codebase structured into strict logical modules (chemistry, models, similarity, ranking, literature, presentation).
- **Consequences:** 
  - Zero deployment friction (runs locally via `streamlit run app.py` or standard virtual environments).
  - High performance via in-memory data structures without network serialization latency.
  - Clear boundaries maintained by linting and architectural rules.

## ADR-002: Direct RDKit Morgan Circular Fingerprints (Radius 2, 2048 Bits)
- **Context:** Feature representation for both machine learning models (ClinTox, Tox21) and structural similarity calculation.
- **Decision:** Standardize globally on Morgan circular fingerprints with radius=2 (equivalent to ECFP4) and bit vector length 2048.
- **Rationale:** ECFP4 is the standard benchmark representation across literature for small-molecule property prediction, exhibits balanced resolution and bit collision frequency, and aligns with Tox21/ClinTox benchmark literature.

## ADR-003: Model Family Selection: Random Forest Classifiers
- **Context:** ClinTox and Tox21 exhibit severe class imbalance, small-to-moderate sample sizes (1,478 molecules in ClinTox; ~8,000 in Tox21), and binary sparse features (2048 bits).
- **Decision:** Use scikit-learn `RandomForestClassifier` with 300 trees and `class_weight='balanced'`.
- **Rationale:** Random Forests are robust against overfitting on sparse high-dimensional Morgan bit-vectors, naturally support tree-based SHAP explainability via `TreeExplainer`, require minimal hyperparameter tuning compared to deep neural networks, and execute deterministically with fixed random seeds.

## ADR-004: Explicit Positive Class Mapping in Model Inference
- **Context:** In binary classification, `predict_proba[:, 1]` assumes class 1 is the positive toxicity class. If target labels are encoded differently or classes are ordered non-standardly, prediction probabilities can be inverted.
- **Decision:** Explicitly inspect `model.classes_` during training and persist the positive class index in model metadata (`metadata.json`). Inference services must query the explicit positive index.

## ADR-005: Literature Module Semantic Ranking Resilience
- **Context:** Full neural sentence transformers (`sentence-transformers` + `torch`) require large download sizes (~2GB) and may encounter native wheel compilation issues in certain Python environments.
- **Decision:** Implement a dual-strategy Literature Engine:
  1. Primary: `sentence-transformers` using `all-MiniLM-L6-v2` when PyTorch is installed.
  2. Fallback: Robust TF-IDF / BM25 cosine similarity vectorizer built on `scikit-learn` that runs instantaneously with zero additional dependencies, maintaining identical output schema and citation integrity.

## ADR-006: Candidate Prioritization Formula Transparency
- **Context:** Multi-objective scoring combining safety, drug-likeness, and similarity.
- **Decision:** Implement the explicit PRD formula:
  $$R = 0.5 \times (1 - P_{tox}) \times 100 + 0.3 \times D + 0.2 \times (100 \times T_{max})$$
  with $D \in \{40, 100\}$.
- **Rationale:** The formula is completely deterministic, transparent, and avoids black-box heuristics. Sensitivity analysis is provided to demonstrate rank variance under parameter perturbation.
