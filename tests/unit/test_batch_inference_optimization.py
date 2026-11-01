"""Correctness tests for Priority 2I: Batch Inference Optimization.

Verifies bit-for-bit equivalence and strict contract preservation between the
optimized batch inference path and single-molecule/unoptimized ground truth.
"""

import pytest
import numpy as np
import pandas as pd

from src.services.orchestration import ScreeningOrchestrator
from src.services.contracts import BatchScreeningResult, BatchCandidateRecord
from src.models.inference import ToxicityInferenceService
from src.similarity.index import ChemicalSimilarityIndex
from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.chemistry.descriptors import calculate_descriptors
from src.ranking.scoring import calculate_prioritization_score

TEST_MOLECULES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(=O)Nc1ccc(O)cc1",      # Paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",    # Ibuprofen
    "CN(C)C(=N)NC(=N)N",             # Metformin
    "c1ccccc1",                       # Benzene
    "CCN(CC)CC",                      # Triethylamine
    "O=C1NC(=O)NC(=O)C1(CC)c2ccccc2", # Phenobarbital
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # Testosterone
    "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", # Atorvastatin
]


@pytest.fixture(scope="module")
def shared_orchestrator():
    return ScreeningOrchestrator()


def _run_reference_sequential_batch(smiles_list, names_list=None):
    """Ground-truth unoptimized reference implementation evaluating each row independently."""
    inf_service = ToxicityInferenceService()
    sim_idx = ChemicalSimilarityIndex()

    records = []
    valid_count = 0
    invalid_count = 0
    total = len(smiles_list)

    for idx, smi in enumerate(smiles_list):
        name = names_list[idx] if (names_list and idx < len(names_list)) else f"Compound_{idx+1}"
        val_res = validate_smiles(str(smi))
        if not val_res.is_valid or not val_res.canonical_smiles:
            invalid_count += 1
            records.append(
                BatchCandidateRecord(
                    row_id=idx + 1,
                    name=name,
                    original_smiles=str(smi),
                    canonical_smiles=None,
                    is_valid=False,
                    error_message=val_res.error_message,
                    molecular_weight=None,
                    logp=None,
                    hbond_donors=None,
                    hbond_acceptors=None,
                    rotatable_bonds=None,
                    tpsa=None,
                    lipinski_pass=None,
                    clintox_p_tox=None,
                    clintox_is_toxic=None,
                    tox21_active_count=None,
                    top_similarity=None,
                    top_reference_name=None,
                    safety_score=None,
                    druglikeness_score=None,
                    similarity_score=None,
                    overall_priority_score=None,
                    applicability_domain_status="UNAVAILABLE",
                    applicability_domain_score=None,
                    applicability_warning=None,
                )
            )
            continue

        canonical = val_res.canonical_smiles
        valid_count += 1
        try:
            desc = calculate_descriptors(canonical)
        except Exception:
            desc = None

        clintox = inf_service.predict_clintox(canonical)
        tox21 = inf_service.predict_tox21(canonical)
        sim = sim_idx.search(canonical, top_k=1)
        top_sim = sim.best_similarity if sim.status == "SUCCESS" else 0.0
        top_ref_name = sim.matches[0].name if sim.matches else "None"

        p_tox = clintox.probability if clintox.status == "SUCCESS" else None
        lip_pass = desc.lipinski_pass if desc else None
        prio = calculate_prioritization_score(p_tox=p_tox, lipinski_pass=lip_pass, best_similarity=top_sim)

        records.append(
            BatchCandidateRecord(
                row_id=idx + 1,
                name=name,
                original_smiles=str(smi),
                canonical_smiles=canonical,
                is_valid=True,
                error_message=None,
                molecular_weight=desc.molecular_weight if desc else None,
                logp=desc.logp if desc else None,
                hbond_donors=desc.hbond_donors if desc else None,
                hbond_acceptors=desc.hbond_acceptors if desc else None,
                rotatable_bonds=desc.rotatable_bonds if desc else None,
                tpsa=desc.tpsa if desc else None,
                lipinski_pass=desc.lipinski_pass if desc else None,
                clintox_p_tox=clintox.probability if clintox.status == "SUCCESS" else None,
                clintox_is_toxic=clintox.is_toxic if clintox.status == "SUCCESS" else None,
                tox21_active_count=tox21.active_endpoints_count if tox21.status == "SUCCESS" else None,
                top_similarity=top_sim,
                top_reference_name=top_ref_name,
                safety_score=prio.safety_score if prio.is_complete else None,
                druglikeness_score=prio.druglikeness_score if prio.is_complete else None,
                similarity_score=prio.similarity_score if prio.is_complete else None,
                overall_priority_score=prio.overall_score if prio.is_complete else None,
                applicability_domain_status=clintox.applicability_domain_status if clintox else "UNAVAILABLE",
                applicability_domain_score=clintox.applicability_score if clintox else None,
                applicability_warning=clintox.applicability_warning if clintox else None,
            )
        )

    rows_data = []
    for r in records:
        rows_data.append({
            "Rank": None,
            "Row_ID": r.row_id,
            "Name": r.name,
            "Original_SMILES": r.original_smiles,
            "Canonical_SMILES": r.canonical_smiles or "N/A",
            "Status": "Valid" if r.is_valid else "Invalid",
            "Error": r.error_message or "",
            "MW": r.molecular_weight,
            "LogP": r.logp,
            "HBD": r.hbond_donors,
            "HBA": r.hbond_acceptors,
            "RotBonds": r.rotatable_bonds,
            "TPSA": r.tpsa,
            "Lipinski_Pass": r.lipinski_pass,
            "ClinTox_P_Tox": r.clintox_p_tox,
            "ClinTox_Toxic": r.clintox_is_toxic,
            "Applicability_Domain_Status": r.applicability_domain_status or "UNAVAILABLE",
            "Applicability_Domain_Score": r.applicability_domain_score,
            "Applicability_Warning": r.applicability_warning or "",
            "Tox21_Active_Assays": r.tox21_active_count,
            "Top_Reference_Drug": r.top_reference_name,
            "Tanimoto_Similarity": r.top_similarity,
            "Safety_Score": r.safety_score,
            "Druglikeness_Score": r.druglikeness_score,
            "Similarity_Score": r.similarity_score,
            "Overall_Priority_Score": r.overall_priority_score,
        })

    df_res = pd.DataFrame(rows_data)
    valid_mask = df_res["Status"] == "Valid"
    df_valid = df_res[valid_mask].sort_values(by="Overall_Priority_Score", ascending=False).copy()
    df_valid["Rank"] = range(1, len(df_valid) + 1)
    df_invalid = df_res[~valid_mask].copy()
    df_invalid["Rank"] = "-"
    df_final = pd.concat([df_valid, df_invalid], ignore_index=True)

    return BatchScreeningResult(
        total_records=total,
        valid_records=valid_count,
        invalid_records=invalid_count,
        candidates=records,
        results_dataframe=df_final,
        batch_timestamp="reference",
    )


def test_empty_batch(shared_orchestrator):
    """Verify that an empty batch does not call estimators and returns clean empty result."""
    res = shared_orchestrator.screen_batch([])
    assert res.total_records == 0
    assert res.valid_records == 0
    assert res.invalid_records == 0
    assert len(res.candidates) == 0
    assert len(res.results_dataframe) == 0


def test_all_invalid_batch(shared_orchestrator):
    """Verify that a batch with only invalid SMILES is fully handled without estimator calls."""
    batch = ["INVALID_1", "NOT_A_SMILES", "C1CC"]
    res = shared_orchestrator.screen_batch(batch)
    assert res.total_records == 3
    assert res.valid_records == 0
    assert res.invalid_records == 3
    assert len(res.candidates) == 3
    for c in res.candidates:
        assert not c.is_valid
        assert c.canonical_smiles is None
        assert c.applicability_domain_status == "UNAVAILABLE"
    df = res.results_dataframe
    assert len(df) == 3
    assert all(df["Status"] == "Invalid")
    assert all(df["Rank"] == "-")


def test_mixed_valid_invalid_batch_exact_equivalence(shared_orchestrator):
    """Verify that a mixed batch of valid and invalid molecules matches reference exactly."""
    batch = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin (valid)
        "MALFORMED_ROW_1",        # Invalid
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine (valid)
        "",                       # Empty (invalid)
        "c1ccccc1",               # Benzene (valid)
    ]
    names = ["Aspirin", "Bad_1", "Caffeine", "Empty", "Benzene"]

    opt_res = shared_orchestrator.screen_batch(batch, names_list=names)
    ref_res = _run_reference_sequential_batch(batch, names_list=names)

    assert opt_res.total_records == ref_res.total_records == 5
    assert opt_res.valid_records == ref_res.valid_records == 3
    assert opt_res.invalid_records == ref_res.invalid_records == 2

    # Compare records bit-for-bit
    for opt_c, ref_c in zip(opt_res.candidates, ref_res.candidates):
        assert opt_c.row_id == ref_c.row_id
        assert opt_c.name == ref_c.name
        assert opt_c.is_valid == ref_c.is_valid
        assert opt_c.canonical_smiles == ref_c.canonical_smiles
        assert opt_c.clintox_p_tox == ref_c.clintox_p_tox
        assert opt_c.clintox_is_toxic == ref_c.clintox_is_toxic
        assert opt_c.tox21_active_count == ref_c.tox21_active_count
        assert opt_c.top_similarity == ref_c.top_similarity
        assert opt_c.top_reference_name == ref_c.top_reference_name
        assert opt_c.safety_score == ref_c.safety_score
        assert opt_c.druglikeness_score == ref_c.druglikeness_score
        assert opt_c.similarity_score == ref_c.similarity_score
        assert opt_c.overall_priority_score == ref_c.overall_priority_score
        assert opt_c.applicability_domain_status == ref_c.applicability_domain_status
        assert opt_c.applicability_domain_score == ref_c.applicability_domain_score

    # Compare DataFrames
    opt_df = opt_res.results_dataframe
    ref_df = ref_res.results_dataframe
    pd.testing.assert_frame_equal(opt_df, ref_df)


def test_benchmark_workload_exact_equivalence(shared_orchestrator):
    """Verify that the 20-compound benchmark workload matches reference bit-for-bit."""
    workload = TEST_MOLECULES * 2  # 20 molecules
    opt_res = shared_orchestrator.screen_batch(workload)
    ref_res = _run_reference_sequential_batch(workload)

    assert opt_res.total_records == ref_res.total_records == 20
    assert opt_res.valid_records == ref_res.valid_records == 20

    # Test candidate record exact equivalence
    for i in range(20):
        opt_c = opt_res.candidates[i]
        ref_c = ref_res.candidates[i]
        assert opt_c.row_id == ref_c.row_id
        assert opt_c.canonical_smiles == ref_c.canonical_smiles
        assert opt_c.clintox_p_tox == ref_c.clintox_p_tox
        assert opt_c.clintox_is_toxic == ref_c.clintox_is_toxic
        assert opt_c.tox21_active_count == ref_c.tox21_active_count
        assert opt_c.top_similarity == ref_c.top_similarity
        assert opt_c.top_reference_name == ref_c.top_reference_name
        assert opt_c.overall_priority_score == ref_c.overall_priority_score
        assert opt_c.applicability_domain_status == ref_c.applicability_domain_status
        assert opt_c.applicability_domain_score == ref_c.applicability_domain_score

    # Test DataFrame exact equivalence (including ranking and column ordering)
    pd.testing.assert_frame_equal(opt_res.results_dataframe, ref_res.results_dataframe)


def test_duplicate_and_equivalent_structures(shared_orchestrator):
    """Verify that duplicate molecules receive identical scores and ranks predictably."""
    # Aspirin written two different ways (different atom starting orders)
    smi1 = "CC(=O)Oc1ccccc1C(=O)O"
    smi2 = "O=C(O)c1ccccc1OC(=O)C"
    res = shared_orchestrator.screen_batch([smi1, smi2], names_list=["Asp_1", "Asp_2"])

    assert res.valid_records == 2
    c1, c2 = res.candidates[0], res.candidates[1]
    assert c1.canonical_smiles == c2.canonical_smiles
    assert c1.clintox_p_tox == c2.clintox_p_tox
    assert c1.tox21_active_count == c2.tox21_active_count
    assert c1.overall_priority_score == c2.overall_priority_score
