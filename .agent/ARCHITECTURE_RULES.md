# Architecture Rules & Code Standards

## 1. Layered Monolithic Architecture
Strict dependency flow:
`Presentation (Streamlit UI)` -> `Application Services` -> `Domain & Scientific Logic` -> `Infrastructure / Data Adapters`

- Domain logic (e.g. `src/chemistry`, `src/models`, `src/similarity`, `src/ranking`, `src/explainability`) must NEVER import Streamlit (`st.*`).
- Presentation components only call application service interfaces.
- Scientific calculations must be directly importable and testable via standard Python unit tests without UI dependencies.

## 2. Chemical Identity & Cheminformatics Integrity
- Do not treat SMILES as plain text strings. Always parse and validate via RDKit.
- Maintain original user input alongside canonical isomeric SMILES.
- Explicitly report invalid or unparseable structures; never silently drop or replace them with defaults.
- Lipinski Rule of 5 must be reported as a heuristic drug-likeness indicator, never conflated with verified therapeutic efficacy or safety.
- Morgan circular fingerprints must use a strictly standardized configuration across training, inference, and similarity: radius=2, nBits=2048, bit-vector format.

## 3. Data Governance & Machine Learning Semantics
- ClinTox target: `CT_TOX` (clinical trial toxicity failure). Positive class must be verified via model `classes_` array index (not assumed column index).
- Tox21 12 targets: Treat each target as an independent classification task. Filter unlabelled rows per-endpoint. Never impute missing assay results as negative.
- Model evaluation must report ROC-AUC, PR-AUC, Balanced Accuracy, Precision, Recall, and Confusion Matrices on held-out test splits.
- Split strategy: 80/20 stratified split with fixed seed=42 for baseline reproducibility.
- Never report probabilities as biological certainty; label them as model-estimated probabilities with known applicability limitations.

## 4. Error Handling & Fault Isolation
- In batch operations, row-level parse or inference failures must not terminate the entire batch. Record row-level error status and continue processing valid rows.
- External API calls (NCBI, ChEMBL) must have timeouts, retries, and offline cache fallbacks.

## 5. Security & Configuration
- External parameters and file paths must be loaded from `config/settings.yaml` or environment variables.
- User input files (CSV) must be strictly validated before parsing.
- Pickled model artifacts must only be loaded from local project directories.
