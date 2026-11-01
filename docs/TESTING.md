# Testing Strategy and Verification Evidence

**Project:** MoleculeIQ  
**Test Framework:** `pytest` (v9.1.1)  
**Execution Date:** 2026-10-03  

---

## 1. Multi-Layer Testing Architecture

MoleculeIQ implements a 4-tier test strategy to ensure chemistry fidelity, model reproducibility, fault isolation, and end-to-end user journey integrity:

1. **Unit Testing (`tests/unit/`):**
   - Molecular syntax parsing, canonicalization, whitespace handling, invalid chemical string rejection (`test_chemistry.py`).
   - Physicochemical descriptors calculation (MW, LogP, HBD, HBA, RotBonds, TPSA) and Lipinski Rule of 5 evaluation against known benchmarks: Ethanol, Aspirin, Caffeine, Atorvastatin (`test_chemistry.py`).
   - Morgan circular fingerprint generation, bit-vector dimensions (2048), determinism, and atom environment mapping (`test_chemistry.py`).
   - Tanimoto mathematical properties: identity $T(A,A)=1.0$, symmetry $T(A,B)=T(B,A)$, bounds $0 \le T \le 1$ (`test_similarity.py`).
   - Prioritization formula $R = 0.5S + 0.3D + 0.2M$ verified numerically against test vectors (e.g. $P_{\mathrm{tox}}=0.20$, Lipinski=Pass, $M=0.75 \implies R=85.0$) (`test_ranking.py`).
   - Prioritization sensitivity analysis across weight perturbation scenarios (`test_ranking.py`).

2. **Integration Testing (`tests/integration/`):**
   - ClinTox model loading, probability inference, and explicit positive class index mapping (`test_models.py`).
   - Tox21 12-endpoint multi-model execution and fault isolation (`test_models.py`).
   - SHAP TreeExplainer feature attribution, active bit selection, and SVG atom highlighting (`test_explainability.py`).
   - PubMed literature retrieval, sentence segmentation, and semantic cosine ranking (`test_literature.py`).

3. **End-to-End Workflow Testing (`tests/e2e/`):**
   - Workflow A: Single molecule end-to-end analysis on Aspirin (`test_single_workflow.py`).
   - Workflow B: Batch molecular screening with row-level fault isolation, ranking, and CSV export (`test_batch_workflow.py`).

---

## 2. Test Execution Evidence

All 37 test cases passed with a 100% pass rate:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.1.1
rootdir: <repository-root>
collected 37 items

tests/e2e/test_batch_workflow.py::TestBatchScreeningWorkflow::test_mixed_valid_and_invalid_batch_screening PASSED
tests/e2e/test_single_workflow.py::TestSingleMoleculeWorkflow::test_complete_aspirin_pipeline PASSED
tests/integration/test_explainability.py::TestExplainabilityEngine::test_explain_valid_molecule_aspirin PASSED
tests/integration/test_explainability.py::TestExplainabilityEngine::test_explain_invalid_molecule_fails_gracefully PASSED
tests/integration/test_literature.py::TestLiteratureEngine::test_sentence_segmentation PASSED
tests/integration/test_literature.py::TestLiteratureEngine::test_empty_query_handling PASSED
tests/integration/test_literature.py::TestLiteratureEngine::test_live_or_mock_retrieval_and_ranking PASSED
tests/integration/test_models.py::TestToxicityInference::test_artifacts_exist PASSED
tests/integration/test_models.py::TestToxicityInference::test_clintox_prediction_valid_molecule PASSED
tests/integration/test_models.py::TestToxicityInference::test_clintox_prediction_invalid_molecule PASSED
tests/integration/test_models.py::TestToxicityInference::test_tox21_multi_prediction_valid_molecule PASSED
tests/integration/test_models.py::TestToxicityInference::test_tox21_prediction_invalid_molecule PASSED
tests/unit/test_chemistry.py::TestMolecularValidation::test_valid_smiles_ethanol PASSED
tests/unit/test_chemistry.py::TestMolecularValidation::test_valid_smiles_aspirin PASSED
tests/unit/test_chemistry.py::TestMolecularValidation::test_whitespace_handling PASSED
tests/unit/test_chemistry.py::TestMolecularValidation::test_empty_input_rejected PASSED
tests/unit/test_chemistry.py::TestMolecularValidation::test_invalid_type_rejected PASSED
tests/unit/test_chemistry.py::TestMolecularValidation::test_malformed_smiles_rejected PASSED
tests/unit/test_chemistry.py::TestMolecularValidation::test_batch_validation_fault_isolation PASSED
tests/unit/test_chemistry.py::TestMolecularDescriptors::test_ethanol_descriptors PASSED
tests/unit/test_chemistry.py::TestMolecularDescriptors::test_aspirin_descriptors PASSED
tests/unit/test_chemistry.py::TestMolecularDescriptors::test_caffeine_descriptors PASSED
tests/unit/test_chemistry.py::TestMolecularDescriptors::test_lipinski_violation_detection_atorvastatin PASSED
tests/unit/test_chemistry.py::TestMolecularDescriptors::test_invalid_input_raises_value_error PASSED
tests/unit/test_chemistry.py::TestMolecularFingerprints::test_fingerprint_dimensions_and_properties PASSED
tests/unit/test_chemistry.py::TestMolecularFingerprints::test_fingerprint_determinism PASSED
tests/unit/test_chemistry.py::TestMolecularFingerprints::test_structural_distinction PASSED
tests/unit/test_chemistry.py::TestMolecularFingerprints::test_bit_info_atom_environment_mapping PASSED
tests/unit/test_chemistry.py::TestMolecularFingerprints::test_invalid_input_handling PASSED
tests/unit/test_ranking.py::TestCandidateRanking::test_srd_reference_acceptance_vector PASSED
tests/unit/test_ranking.py::TestCandidateRanking::test_lipinski_fail_score_penalty PASSED
tests/unit/test_ranking.py::TestCandidateRanking::test_boundary_zero_and_maximum_values PASSED
tests/unit/test_ranking.py::TestCandidateRanking::test_missing_data_returns_incomplete_status PASSED
tests/unit/test_ranking.py::TestCandidateRanking::test_sensitivity_analysis PASSED
tests/unit/test_similarity.py::TestChemicalSimilarity::test_tanimoto_mathematical_properties PASSED
tests/unit/test_similarity.py::TestChemicalSimilarity::test_similarity_index_search_aspirin_retrieval PASSED
tests/unit/test_similarity.py::TestChemicalSimilarity::test_invalid_query_safe_handling PASSED

============================= 37 passed in 260.91s =============================
```
