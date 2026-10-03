# Requirements Traceability Matrix

**Project:** MoleculeIQ  
**Version:** 1.0  
**PRD / SRD Source:** MoleculeIQ PRD v1.0 & SRD v1.0  

---

## 1. Functional Requirements Traceability

| Req ID | Requirement Description | Module | Implementation Target | Verification Method | Status |
|--------|-------------------------|--------|-----------------------|---------------------|--------|
| FR-001 | Accept SMILES for single-molecule analysis | M01 | `src/chemistry/validation.py` | Unit / UI Test | In Progress |
| FR-002 | Validate molecular syntax using RDKit | M01 | `src/chemistry/validation.py` | Unit Test | In Progress |
| FR-003 | Reject invalid structures before model inference | M01 | `src/chemistry/validation.py` | Negative Unit Test | In Progress |
| FR-004 | Return readable error messages for invalid input | M01 | `src/chemistry/validation.py` | Unit Test | In Progress |
| FR-005 | Preserve original input data for traceability | M01 | `src/chemistry/validation.py` | Unit Test | In Progress |
| FR-006 | Batch validator identifies invalid rows independently | M01 | `src/chemistry/validation.py` | Batch Test | In Progress |
| FR-007 | Calculate required physicochemical descriptors | M02 | `src/chemistry/descriptors.py` | Reference Value Test | In Progress |
| FR-008 | Calculate Lipinski violation counts | M02 | `src/chemistry/descriptors.py` | Reference Value Test | In Progress |
| FR-009 | Return structured descriptor object | M02 | `src/chemistry/descriptors.py` | Type/Schema Test | In Progress |
| FR-010 | Prevent descriptor errors from returning fake zeroes | M02 | `src/chemistry/descriptors.py` | Exception Test | In Progress |
| FR-011 | Generate 2048-bit Morgan fingerprints (radius=2) | M03 | `src/chemistry/fingerprints.py` | Dimension Test | In Progress |
| FR-012 | Ensure inference and training fingerprint configs match | M03 | `src/chemistry/fingerprints.py` | Regression Test | In Progress |
| FR-013 | Ensure similarity and model fingerprint configs match | M03 | `src/chemistry/fingerprints.py` | Regression Test | In Progress |
| FR-014 | Validate fingerprint dimension before model inference | M03 | `src/chemistry/fingerprints.py` | Dimension Check | In Progress |
| FR-015 | Train ClinTox classifier with RF (300 est, balanced) | M04 | `src/models/clintox.py` | Training Pipeline | Planned |
| FR-016 | Evaluate ClinTox model on held-out stratified test set | M04 | `src/models/clintox.py` | Evaluation Suite | Planned |
| FR-017 | Report accuracy, precision, recall, F1, and ROC-AUC | M04 | `src/models/clintox.py` | Metric Report | Planned |
| FR-018 | Persist trained model artifact and metadata | M04 | `src/models/clintox.py` | Artifact Audit | Planned |
| FR-019 | Generate positive-class probability for queries | M04 | `src/models/inference.py` | Inference Test | Planned |
| FR-020 | Identify positive-class label explicitly via metadata | M04 | `src/models/inference.py` | Class Mapping Test | Planned |
| FR-021 | Record dataset version and training configuration | M04 | `src/models/clintox.py` | Metadata Audit | Planned |
| FR-022 | Train independent classifier for 12 Tox21 endpoints | M05 | `src/models/tox21.py` | Multi-Target Suite | Planned |
| FR-023 | Exclude missing assay labels per endpoint | M05 | `src/models/tox21.py` | Ingestion Audit | Planned |
| FR-024 | Report endpoint-specific evaluation metrics | M05 | `src/models/tox21.py` | Metric Report | Planned |
| FR-025 | Save and load each Tox21 model independently | M05 | `src/models/tox21.py` | Artifact Audit | Planned |
| FR-026 | Expose endpoint-level predictions and probabilities | M05 | `src/models/inference.py` | Inference Test | Planned |
| FR-027 | Isolate missing/failed endpoint models gracefully | M05 | `src/models/inference.py` | Fault Test | Planned |
| FR-028 | Maintain locally cached reference library | M06 | `src/similarity/index.py` | Cache Verification | Planned |
| FR-029 | Generate compatible reference fingerprints | M06 | `src/similarity/index.py` | Dimension Test | Planned |
| FR-030 | Calculate Tanimoto similarity accurately | M06 | `src/similarity/index.py` | Mathematical Test | Planned |
| FR-031 | Return top-N matches in descending similarity order | M06 | `src/similarity/index.py` | Ranking Test | Planned |
| FR-032 | Preserve reference identifiers and names | M06 | `src/similarity/index.py` | Metadata Test | Planned |
| FR-033 | Handle empty reference or invalid queries safely | M06 | `src/similarity/index.py` | Boundary Test | Planned |
| FR-034 | Compute safety score $S = (1 - P_{tox}) \times 100$ | M07 | `src/ranking/scoring.py` | Numerical Test | Planned |
| FR-035 | Calculate drug-likeness score $D \in \{40, 100\}$ | M07 | `src/ranking/scoring.py` | Numerical Test | Planned |
| FR-036 | Calculate similarity component $M = 100 \times T_{max}$ | M07 | `src/ranking/scoring.py` | Numerical Test | Planned |
| FR-037 | Compute composite score $R = 0.5S + 0.3D + 0.2M$ | M07 | `src/ranking/scoring.py` | Numerical Test | Planned |
| FR-038 | Display individual score components | M07 | `src/ranking/scoring.py` | Schema Test | Planned |
| FR-039 | Sort batch screening results in descending score | M07 | `src/ranking/scoring.py` | Sort Test | Planned |
| FR-040 | Report incomplete scores explicitly without fake values | M07 | `src/ranking/scoring.py` | Missing Data Test | Planned |
| FR-041 | Retain formula version in result metadata | M07 | `src/ranking/scoring.py` | Provenance Test | Planned |
| FR-042 | Calculate SHAP contributions for ClinTox predictions | M08 | `src/explainability/shap_analysis.py` | Integration Test | Planned |
| FR-043 | Identify the explained target class | M08 | `src/explainability/shap_analysis.py` | Logic Test | Planned |
| FR-044 | Select top active bits for visualization | M08 | `src/explainability/shap_analysis.py` | Feature Test | Planned |
| FR-045 | Map active bits to Morgan atom environments | M08 | `src/explainability/shap_analysis.py` | Substructure Test | Planned |
| FR-046 | Handle ambiguous bit mappings transparently | M08 | `src/explainability/shap_analysis.py` | Edge Case Test | Planned |
| FR-047 | Accompany explanations with non-causal disclaimer | M08 | `src/explainability/shap_analysis.py` | Disclaimer Test | Planned |
| FR-048 | Retrieve PubMed records via NCBI E-utilities | M09 | `src/literature/pubmed_client.py` | API Test | Planned |
| FR-049 | Support configurable record limit | M09 | `src/literature/pubmed_client.py` | Param Test | Planned |
| FR-050 | Split abstracts into clean sentences | M09 | `src/literature/retrieval.py` | Text Processing | Planned |
| FR-051 | Generate local semantic embeddings | M09 | `src/literature/retrieval.py` | Embedding Test | Planned |
| FR-052 | Rank sentences by cosine similarity to query | M09 | `src/literature/retrieval.py` | Ranking Test | Planned |
| FR-053 | Preserve PMID, title, authors, year, and URLs | M09 | `src/literature/retrieval.py` | Metadata Test | Planned |
| FR-054 | Gracefully handle empty searches and network timeouts | M09 | `src/literature/pubmed_client.py` | Resilience Test | Planned |
| FR-055 | Identify results as scientific literature snippets | M09 | `src/literature/retrieval.py` | Disclaimer Test | Planned |
| FR-056 | Provide Single, Batch, Literature, Insights modes | M10 | `app.py`, `src/ui/*` | UI Flow Test | Planned |
| FR-057 | Display input validation status | M10 | `src/ui/single_molecule.py` | UI Test | Planned |
| FR-058 | Display calculated descriptors table | M10 | `src/ui/single_molecule.py` | UI Test | Planned |
| FR-059 | Display toxicity predictions with model context | M10 | `src/ui/single_molecule.py` | UI Test | Planned |
| FR-060 | Display similarity matches and Tanimoto scores | M10 | `src/ui/single_molecule.py` | UI Test | Planned |
| FR-061 | Display composite score and component breakdown | M10 | `src/ui/single_molecule.py` | UI Test | Planned |
| FR-062 | Render SHAP atom environment highlights | M10 | `src/ui/single_molecule.py` | Visual Test | Planned |
| FR-063 | Provide CSV export for batch screening | M10 | `src/ui/batch_screening.py` | Export Test | Planned |
| FR-064 | Communicate loading, success, error states clearly | M10 | `app.py` | UX Test | Planned |
| FR-065 | Distinguish missing data from numerical zero values | M10 | `src/ui/*` | Data Display Test| Planned |

---

## 2. Technical Acceptance Criteria (TAC) Status

| TAC ID | Description | Verification Target | Status |
|--------|-------------|---------------------|--------|
| TAC-01 | Valid SMILES parsed successfully | `tests/unit/test_chemistry.py` | In Progress |
| TAC-02 | Invalid inputs rejected with controlled errors | `tests/unit/test_chemistry.py` | In Progress |
| TAC-03 | Specified descriptors calculated against known refs | `tests/unit/test_chemistry.py` | In Progress |
| TAC-04 | Morgan fingerprints have exact 2048-bit dimension | `tests/unit/test_chemistry.py` | In Progress |
| TAC-05 | ClinTox model trains and loads with documented settings | `tests/integration/test_models.py` | Planned |
| TAC-06 | Tox21 endpoint models handle missing labels correctly | `tests/integration/test_models.py` | Planned |
| TAC-07 | Model probabilities map to positive class explicitly | `tests/integration/test_models.py` | Planned |
| TAC-08 | Tanimoto similarity calculations match identity/bounds | `tests/unit/test_similarity.py` | Planned |
| TAC-09 | Ranking formula output matches manual test calculation | `tests/unit/test_ranking.py` | Planned |
| TAC-10 | SHAP output generated for supported inputs | `tests/integration/test_explainability.py` | Planned |
| TAC-11 | Literature retrieval preserves full citation metadata | `tests/integration/test_literature.py` | Planned |
| TAC-12 | Invalid batch rows do not abort batch processing | `tests/e2e/test_batch_workflow.py` | Planned |
| TAC-13 | CSV exports preserve exact defined schema | `tests/e2e/test_batch_workflow.py` | Planned |
| TAC-14 | All major UI workflows execute without crashes | Manual & automated UI tests | Planned |
| TAC-15 | Model evaluation reports generated and documented | `docs/MODEL_CARD_*.md` | Planned |
| TAC-16 | Setup and execution commands documented and verified | `docs/DEPLOYMENT.md`, `README.md` | Planned |
| TAC-17 | Scientific limitations clearly visible in documentation & UI | Review checklist | Planned |
