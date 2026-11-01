"""Focused automated unit tests for Phase B.3 Stage 1: Scaffold Generalization Engine.

Covers all 12 required verification categories (Section D):
1. Deterministic scaffold and group generation.
2. Acyclic grouping behavior and oversized-group detection.
3. Duplicate structures, salts, stereoisomers, and documented parent-group behavior.
4. Zero group overlap and complete fold-index conservation.
5. Detection of deliberately injected split leakage.
6. Fixed-seed reproducibility and alternative-seed behavior (ClinTox seeds 42, 1042, 2042).
7. Endpoint-specific missing-label isolation.
8. Metric calculations on hand-verifiable synthetic data.
9. Degenerate folds and insufficient positive/negative support.
10. Deterministic scaffold-cluster bootstrap and invalid-replicate accounting.
11. No file or artifact mutation during module import or unit tests (31 SHA-256 checksums).
12. Preservation of existing production inference and ranking outputs.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.models.scaffold_generalization import (
    CLINTOX_PRESPECIFIED_SEEDS,
    STAGE1_CALIBRATION_SCOPE_NOTE,
    STRATEGY_EXACT_CANONICAL,
    STRATEGY_GENERIC_MURCKO_FRAMEWORK,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_FORMULA,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLE_BUCKET,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON,
    InfeasibleFoldConfigurationError,
    ScaffoldGeneralizationError,
    SplitIntegrityError,
    assign_scaffold_group_key,
    audit_cohort_structure_relationships,
    build_clintox_evaluation_cohort,
    build_tox21_endpoint_cohort,
    compare_paired_oof_predictions,
    compute_binary_evaluation_metrics,
    compute_full_and_cyclic_metrics,
    compute_scaffold_cluster_bootstrap,
    compute_split_seed_variation_summary,
    extract_generic_murcko_scaffold,
    extract_structure_identity,
    generate_grouped_kfold_splits,
    generate_matched_random_kfold_splits,
    generate_scaffold_group_assignments,
    summarize_fold_and_pooled_metrics,
    summarize_group_distribution,
    summarize_repeated_cv_runs,
    validate_split_integrity,
)
from src.models.tox21 import TOX21_ENDPOINTS
from src.services.orchestration import ScreeningOrchestrator


# ---------------------------------------------------------------------------
# Synthetic Fixtures
# ---------------------------------------------------------------------------


def _build_synthetic_scaffold_cohort() -> Tuple[List[str], List[int]]:
    """Construct a fast, balanced synthetic cohort with 10 distinct scaffold groups (50 molecules)."""
    smiles_list: List[str] = [
        # Group 1: Benzene derivatives (6 molecules: 2 pos, 4 neg)
        "c1ccccc1C",
        "c1ccccc1CC",
        "c1ccccc1CCC",
        "c1ccccc1O",
        "c1ccccc1N",
        "c1ccccc1Cl",
        # Group 2: Pyridine derivatives (5 molecules: 1 pos, 4 neg)
        "c1ccncc1C",
        "c1ccncc1CC",
        "c1ccncc1O",
        "c1ccncc1N",
        "c1ccncc1F",
        # Group 3: Pyrimidine derivatives (5 molecules: 1 pos, 4 neg)
        "c1cncnc1C",
        "c1cncnc1CC",
        "c1cncnc1O",
        "c1cncnc1N",
        "c1cncnc1Cl",
        # Group 4: Naphthalene derivatives (5 molecules: 1 pos, 4 neg)
        "c1ccc2ccccc2c1C",
        "c1ccc2ccccc2c1CC",
        "c1ccc2ccccc2c1O",
        "c1ccc2ccccc2c1N",
        "c1ccc2ccccc2c1F",
        # Group 5: Quinoline derivatives (5 molecules: 1 pos, 4 neg)
        "c1ccc2ncccc2c1C",
        "c1ccc2ncccc2c1CC",
        "c1ccc2ncccc2c1O",
        "c1ccc2ncccc2c1N",
        "c1ccc2ncccc2c1Cl",
        # Group 6: Thiophene derivatives (5 molecules: 1 pos, 4 neg)
        "c1ccsc1C",
        "c1ccsc1CC",
        "c1ccsc1O",
        "c1ccsc1N",
        "c1ccsc1Br",
        # Group 7: Furan derivatives (5 molecules: 1 pos, 4 neg)
        "c1ccoc1C",
        "c1ccoc1CC",
        "c1ccoc1O",
        "c1ccoc1N",
        "c1ccoc1F",
        # Group 8: Indole derivatives (5 molecules: 1 pos, 4 neg)
        "c1ccc2[nH]ccc2c1C",
        "c1ccc2[nH]ccc2c1CC",
        "c1ccc2[nH]ccc2c1O",
        "c1ccc2[nH]ccc2c1N",
        "c1ccc2[nH]ccc2c1Cl",
        # Group 9: Acyclic 4-heavy-atom linear chains (5 molecules: 1 pos, 4 neg)
        "CCCC",
        "CCCO",
        "CCCN",
        "CCCF",
        "CCCCl",
        # Group 10: Acyclic 5-heavy-atom branched chains (4 molecules: 1 pos, 3 neg)
        "CC(C)CC",
        "CC(C)CO",
        "CC(C)CN",
        "CC(C)CF",
    ]
    labels: List[int] = [
        1, 1, 0, 0, 0, 0,  # Group 1
        1, 0, 0, 0, 0,     # Group 2
        1, 0, 0, 0, 0,     # Group 3
        1, 0, 0, 0, 0,     # Group 4
        1, 0, 0, 0, 0,     # Group 5
        1, 0, 0, 0, 0,     # Group 6
        1, 0, 0, 0, 0,     # Group 7
        1, 0, 0, 0, 0,     # Group 8
        1, 0, 0, 0, 0,     # Group 9
        1, 0, 0, 0,        # Group 10
    ]
    return smiles_list, labels


def _verify_all_31_protected_checksums() -> Tuple[int, List[str]]:
    """Verify all 32 entries in artifacts/checksums.sha256 against current files on disk."""
    manifest_path = "artifacts/checksums.sha256"
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    mismatches: List[str] = []
    for line in lines:
        expected_hash, rel_path = line.split("  ", 1)
        target_file = os.path.join("artifacts", rel_path)
        if not os.path.exists(target_file):
            mismatches.append(f"Missing: {rel_path}")
            continue
        with open(target_file, "rb") as bf:
            actual_hash = hashlib.sha256(bf.read()).hexdigest()
        if actual_hash != expected_hash:
            mismatches.append(f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

    return len(lines), mismatches


# ---------------------------------------------------------------------------
# 1. Deterministic Scaffold and Group Generation
# ---------------------------------------------------------------------------


class TestDeterministicScaffoldAndGroupGeneration:
    """Category 1: Verifies deterministic scaffold decomposition and group key generation."""

    def test_cyclic_and_generic_murcko_decomposition(self) -> None:
        # Benzene and Pyridine have distinct cyclic Murcko scaffolds but identical 6-ring generic framework
        aspirin_id = extract_structure_identity("CC(=O)Oc1ccccc1C(=O)O")
        niacin_id = extract_structure_identity("O=C(O)c1cccnc1")

        assert aspirin_id.is_cyclic is True
        assert aspirin_id.bemis_murcko_scaffold == "c1ccccc1"
        assert aspirin_id.generic_murcko_framework == "C1CCCCC1"
        assert aspirin_id.acyclic_generic_skeleton == ""

        assert niacin_id.is_cyclic is True
        assert niacin_id.bemis_murcko_scaffold == "c1ccncc1"
        assert niacin_id.generic_murcko_framework == "C1CCCCC1"

        # Under murcko_cyclic_acyclic_generic, benzene and pyridine are distinct groups
        assert assign_scaffold_group_key(
            aspirin_id, STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC
        ) == "cyc_murcko:c1ccccc1"
        assert assign_scaffold_group_key(
            niacin_id, STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC
        ) == "cyc_murcko:c1ccncc1"

        # Under generic_murcko_framework, both map to the same 6-membered generic ring framework
        assert assign_scaffold_group_key(
            aspirin_id, STRATEGY_GENERIC_MURCKO_FRAMEWORK
        ) == "gen_murcko:C1CCCCC1"
        assert assign_scaffold_group_key(
            niacin_id, STRATEGY_GENERIC_MURCKO_FRAMEWORK
        ) == "gen_murcko:C1CCCCC1"

    def test_hypervalent_ring_generic_scaffold_safe_fallback(self) -> None:
        # Cyclic sulfone (saccharin-like / cyclic sulfonamide) has hexavalent ring sulfur (degree > 4)
        saccharin_smi = "O=C1NS(=O)(=O)c2ccccc12"
        ident = extract_structure_identity(saccharin_smi)
        assert ident.is_cyclic is True
        assert ident.bemis_murcko_scaffold != ""
        assert ident.generic_murcko_framework != ""

        # Platinum coordination complex (cisplatin) is acyclic with degree 4
        cisplatin_id = extract_structure_identity("[NH3][Pt]([NH3])(Cl)Cl")
        assert cisplatin_id.is_cyclic is False
        assert cisplatin_id.acyclic_generic_skeleton != ""

        # Hexacoordinate sulfur / phosphorus acyclic compound (degree > 4)
        pf6_like = "FS(F)(F)(F)(F)C"
        gen_skel = extract_generic_murcko_scaffold(pf6_like)
        assert isinstance(gen_skel, str) and len(gen_skel) > 0

    def test_repeated_calls_are_strictly_deterministic(self) -> None:
        smiles_list, _ = _build_synthetic_scaffold_cohort()
        keys_1, ids_1 = generate_scaffold_group_assignments(
            smiles_list, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC
        )
        keys_2, ids_2 = generate_scaffold_group_assignments(
            smiles_list, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC
        )
        assert keys_1 == keys_2
        assert [i.canonical_smiles for i in ids_1] == [i.canonical_smiles for i in ids_2]

        with pytest.raises(ScaffoldGeneralizationError, match="Unsupported grouping strategy"):
            generate_scaffold_group_assignments(smiles_list, strategy="non_existent_strategy")


# ---------------------------------------------------------------------------
# 2. Acyclic Grouping Behavior and Oversized-Group Detection
# ---------------------------------------------------------------------------


class TestAcyclicGroupingAndOversizedGroupDetection:
    """Category 2: Verifies acyclic grouping policies and oversized-group detection."""

    def test_acyclic_grouping_strategies_distinguish_skeletons_singletons_and_formulas(self) -> None:
        acyclic_smiles = [
            "CCCCO",    # 5 heavy atoms, linear -> generic CCCCC
            "CCCCN",    # 5 heavy atoms, linear -> generic CCCCC
            "CC(C)CO",  # 5 heavy atoms, branched -> generic CC(C)CC
            "CCCCCO",   # 6 heavy atoms, linear -> generic CCCCCC
        ]
        # 1. Generic skeleton groups linear 5-atom chains (CCCCO, CCCCN) together, separates branched & 6-atom
        g_gen, _ = generate_scaffold_group_assignments(
            acyclic_smiles, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC
        )
        assert g_gen[0] == g_gen[1] == "acyc_skeleton:CCCCC"
        assert g_gen[2] == "acyc_skeleton:CCC(C)C"
        assert g_gen[3] == "acyc_skeleton:CCCCCC"
        assert len(set(g_gen)) == 3

        # 2. Singleton strategy keeps every distinct canonical SMILES separate
        g_sing, _ = generate_scaffold_group_assignments(
            acyclic_smiles, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON
        )
        assert len(set(g_sing)) == 4
        assert all(k.startswith("acyc_mol:") for k in g_sing)

        # 3. Formula strategy groups by nHeavy/nC/nHet/nBranch
        g_form, _ = generate_scaffold_group_assignments(
            acyclic_smiles, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_FORMULA
        )
        assert g_form[0] == g_form[1] == "acyc_formula:nHeavy=5_nC=4_nHet=1_nBranch=0"
        assert g_form[2] == "acyc_formula:nHeavy=5_nC=4_nHet=1_nBranch=1"
        assert g_form[3] == "acyc_formula:nHeavy=6_nC=5_nHet=1_nBranch=0"

        # 4. Single-bucket strategy collapses all acyclic molecules into one bucket
        g_all, _ = generate_scaffold_group_assignments(
            acyclic_smiles, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLE_BUCKET
        )
        assert set(g_all) == {"acyc_all:ALL_ACYCLIC"}

    def test_oversized_group_detection_on_synthetic_and_single_bucket_cohorts(self) -> None:
        # Build a 20-molecule cohort where 8 molecules (40% > 20%) are acyclic
        smiles = [
            "CCCC", "CCCO", "CCCN", "CCCF", "CCCCl", "CCCCC", "CCCCO", "CCCCN",  # 8 acyclics
            "c1ccccc1C", "c1ccccc1O",
            "c1ccncc1C", "c1ccncc1O",
            "c1cncnc1C", "c1cncnc1O",
            "c1ccsc1C", "c1ccsc1O",
            "c1ccoc1C", "c1ccoc1O",
            "c1ccc2ccccc2c1C", "c1ccc2ccccc2c1O",
        ]
        labels = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

        # Under murcko_cyclic_acyclic_generic, max acyclic group is size 5 (25%) or under formula size 3 (15%)
        dist_formula = summarize_group_distribution(
            smiles,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_FORMULA,
            labels=labels,
            n_splits=5,
        )
        assert dist_formula.has_oversized_groups is False
        assert dist_formula.max_group_size <= 4
        assert dist_formula.is_k_fold_feasible is True

        # Under murcko_cyclic_acyclic_single_bucket, all 8 acyclics collapse into 1 group (40% > 20%)
        dist_single_bucket = summarize_group_distribution(
            smiles,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLE_BUCKET,
            labels=labels,
            n_splits=5,
            strict_no_oversized_groups=True,
        )
        assert dist_single_bucket.has_oversized_groups is True
        assert "acyc_all:ALL_ACYCLIC" in dist_single_bucket.oversized_groups
        assert dist_single_bucket.oversized_groups["acyc_all:ALL_ACYCLIC"] == 8
        assert dist_single_bucket.is_k_fold_feasible is False
        assert any("oversized" in r for r in dist_single_bucket.infeasibility_reasons)


# ---------------------------------------------------------------------------
# 3. Duplicate Structures, Salts, Stereoisomers, and Parent-Group Behavior
# ---------------------------------------------------------------------------


class TestDuplicatesSaltsStereoisomersAndParentGroups:
    """Category 3: Verifies separate auditing and grouping of salts, parent fragments, and stereoisomers."""

    def test_salt_parent_fragment_and_stereoisomer_identity_extraction(self) -> None:
        # Free base vs hydrochloride salt where counterion has no rings, and salt with cyclic counterion
        free_base = "CN1CCC(CC1)c2ccccc2"
        hcl_salt = "CN1CCC(CC1)c2ccccc2.Cl"
        tosylate_salt = "CCN(CC)CC.Cc1ccc(cc1)S(=O)(=O)O"  # acyclic amine + cyclic tosylate counterion

        id_fb = extract_structure_identity(free_base)
        id_hcl = extract_structure_identity(hcl_salt)
        id_tos = extract_structure_identity(tosylate_salt)

        assert id_fb.is_multifragment_salt is False
        assert id_hcl.is_multifragment_salt is True
        assert id_fb.parent_canonical_smiles == id_hcl.parent_canonical_smiles
        assert id_fb.canonical_smiles != id_hcl.canonical_smiles

        # For tosylate_salt, largest fragment by heavy atoms is tosylate (11 heavy atoms vs 7 for Et3N)
        assert id_tos.is_multifragment_salt is True
        assert id_tos.parent_bemis_murcko_scaffold == "c1ccccc1"

        # Enantiomers: L-alanine vs D-alanine
        l_ala = "C[C@H](N)C(=O)O"
        d_ala = "C[C@@H](N)C(=O)O"
        id_l = extract_structure_identity(l_ala)
        id_d = extract_structure_identity(d_ala)

        assert id_l.has_specified_stereo is True
        assert id_d.has_specified_stereo is True
        assert id_l.canonical_smiles != id_d.canonical_smiles
        assert id_l.achiral_canonical_smiles == id_d.achiral_canonical_smiles

        # Under singleton acyclic strategy, L-ala and D-ala have distinct keys unless merge_stereoisomer_groups=True
        k_l_sep = assign_scaffold_group_key(
            id_l, STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON, merge_stereoisomer_groups=False
        )
        k_d_sep = assign_scaffold_group_key(
            id_d, STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON, merge_stereoisomer_groups=False
        )
        assert k_l_sep != k_d_sep

        k_l_merged = assign_scaffold_group_key(
            id_l, STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON, merge_stereoisomer_groups=True
        )
        k_d_merged = assign_scaffold_group_key(
            id_d, STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON, merge_stereoisomer_groups=True
        )
        assert k_l_merged == k_d_merged

    def test_audit_cohort_structure_relationships_detects_parent_and_stereo_label_conflicts(self) -> None:
        # Construct cohort with:
        # - 1 salt pair with conflicting labels (free base=0, HCl salt=1)
        # - 1 stereoisomer pair with conflicting labels (R=0, S=1)
        # - 1 exact duplicate pair with conflicting labels (4-methylpyridine written two ways)
        smiles = [
            "c1ccccc1CCN",        # free base (label 0)
            "c1ccccc1CCN.Cl",     # HCl salt sharing parent (label 1)
            "C[C@H](O)C(=O)O",    # stereoisomer 1 (label 0)
            "C[C@@H](O)C(=O)O",   # stereoisomer 2 (label 1)
            "c1cc(C)ccn1",        # exact dup 1: 4-methylpyridine (label 0)
            "Cc1ccncc1",          # exact dup 2: 4-methylpyridine (label 1)
        ]
        labels = [0, 1, 0, 1, 0, 1]

        audit = audit_cohort_structure_relationships(smiles, labels)
        assert audit["n_records"] == 6
        assert audit["n_unique_exact_canonical"] == 5
        assert audit["n_exact_duplicate_groups"] == 1
        assert audit["n_exact_label_conflict_groups"] == 1
        assert audit["n_multifragment_salt_records"] == 1
        assert audit["n_shared_parent_fragment_groups"] == 2  # salt pair + exact dup pair
        assert audit["n_parent_fragment_label_conflict_groups"] == 2
        assert audit["n_shared_stereoisomer_groups"] == 2     # stereo pair + exact dup pair
        assert audit["n_stereoisomer_label_conflict_groups"] == 2


# ---------------------------------------------------------------------------
# 4. Zero Group Overlap and Complete Fold-Index Conservation
# ---------------------------------------------------------------------------


class TestZeroGroupOverlapAndFoldConservation:
    """Category 4: Verifies StratifiedGroupKFold invariants on synthetic and real cohorts."""

    def test_synthetic_grouped_kfold_zero_overlap_and_complete_conservation(self) -> None:
        smiles_list, labels = _build_synthetic_scaffold_cohort()
        report, dist = generate_grouped_kfold_splits(
            smiles_list,
            labels,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
            n_splits=5,
            seed=42,
        )
        assert dist.is_k_fold_feasible is True
        assert report.is_valid is True
        assert report.zero_index_overlap is True
        assert report.complete_index_conservation is True
        assert report.zero_group_overlap is True
        assert report.zero_canonical_smiles_overlap is True
        assert report.all_folds_meet_class_support is True
        assert len(report.folds) == 5
        assert sum(f.n_test for f in report.folds) == len(smiles_list)

    def test_clintox_real_cohort_grouped_kfold_and_matched_random_kfold(self) -> None:
        clintox_cohort = build_clintox_evaluation_cohort(compute_fingerprints=False)
        assert clintox_cohort.n_samples == 1461
        assert clintox_cohort.n_positives == 112
        assert clintox_cohort.n_negatives == 1349

        grouped_report, dist = generate_grouped_kfold_splits(
            clintox_cohort.identities,
            clintox_cohort.labels,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
            n_splits=5,
            seed=42,
        )
        assert dist.is_k_fold_feasible is True
        assert dist.has_oversized_groups is False
        assert grouped_report.is_valid is True
        assert grouped_report.zero_group_overlap is True
        assert grouped_report.zero_canonical_smiles_overlap is True
        assert grouped_report.min_fold_test_positives >= 20

        # Verify that every fold in StratifiedGroupKFold contains multi-molecule scaffolds as well as singletons
        # (remediating the legacy greedy split where 100% of test buckets were singletons)
        assert all(f.max_test_group_size > 1 for f in grouped_report.folds)

        # Matched random 5-fold split on the exact same cohort
        random_report = generate_matched_random_kfold_splits(
            clintox_cohort.identities,
            clintox_cohort.labels,
            reference_strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
            n_splits=5,
            seed=42,
        )
        assert random_report.is_valid is True
        assert random_report.n_samples == grouped_report.n_samples
        assert random_report.cohort_positives == grouped_report.cohort_positives
        # Random split allows scaffold groups to cross train/test boundary
        assert random_report.zero_group_overlap is False


# ---------------------------------------------------------------------------
# 5. Detection of Deliberately Injected Split Leakage
# ---------------------------------------------------------------------------


class TestInjectedSplitLeakageDetection:
    """Category 5: Verifies that validate_split_integrity catches every form of injected leakage."""

    def test_detects_injected_group_overlap_between_train_and_test(self) -> None:
        smiles_list, labels = _build_synthetic_scaffold_cohort()
        group_keys, identities = generate_scaffold_group_assignments(
            smiles_list, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC
        )
        report, _ = generate_grouped_kfold_splits(
            identities, labels, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC, n_splits=5, seed=42
        )

        # Corrupt fold 0 and fold 1 by swapping one molecule (index 0, benzene) between fold 0 and fold 1
        f0 = report.folds[0]
        f1 = report.folds[1]
        idx_swap_a = f0.test_indices[0]
        idx_swap_b = f1.test_indices[0]

        corrupted_folds = list(
            (list(f.train_indices), list(f.test_indices)) for f in report.folds
        )
        # Swap idx_swap_a and idx_swap_b between fold 0 test and fold 1 test
        corrupted_folds[0][1].remove(idx_swap_a)
        corrupted_folds[0][1].append(idx_swap_b)
        corrupted_folds[0][0].remove(idx_swap_b)
        corrupted_folds[0][0].append(idx_swap_a)

        corrupted_folds[1][1].remove(idx_swap_b)
        corrupted_folds[1][1].append(idx_swap_a)
        corrupted_folds[1][0].remove(idx_swap_a)
        corrupted_folds[1][0].append(idx_swap_b)

        with pytest.raises(SplitIntegrityError, match="scaffold group\\(s\\) leaked across train and test"):
            validate_split_integrity(
                folds=corrupted_folds,
                identities=identities,
                labels=labels,
                group_keys=group_keys,
                enforce_group_disjointness=True,
                raise_on_violation=True,
            )

    def test_detects_injected_index_overlap_and_missing_index(self) -> None:
        smiles_list, labels = _build_synthetic_scaffold_cohort()
        group_keys, identities = generate_scaffold_group_assignments(
            smiles_list, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC
        )
        report, _ = generate_grouped_kfold_splits(
            identities, labels, strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC, n_splits=5, seed=42
        )

        # Duplicate a test index into train_indices of fold 0
        bad_folds = [
            (list(f.train_indices), list(f.test_indices)) for f in report.folds
        ]
        bad_folds[0][0].append(bad_folds[0][1][0])

        with pytest.raises(SplitIntegrityError, match="overlap between train and test"):
            validate_split_integrity(
                folds=bad_folds,
                identities=identities,
                labels=labels,
                group_keys=group_keys,
                raise_on_violation=True,
            )

    def test_detects_injected_canonical_parent_and_achiral_leakage(self) -> None:
        # Cohort where a free base and its HCl salt are placed in train vs test
        smiles = [
            "c1ccccc1CCN",        # idx 0: free base
            "c1ccccc1CCN.Cl",     # idx 1: HCl salt (same parent, different canonical SMILES)
            "C[C@H](O)C(=O)O",    # idx 2: R-lactic acid
            "C[C@@H](O)C(=O)O",   # idx 3: S-lactic acid (same achiral SMILES)
            "c1ccncc1C",          # idx 4
            "c1ccncc1O",          # idx 5
        ]
        labels = [1, 1, 0, 0, 1, 0]
        group_keys, identities = generate_scaffold_group_assignments(
            smiles, strategy=STRATEGY_EXACT_CANONICAL
        )
        # Fold 0 puts idx 0, 2, 4 in train and idx 1, 3, 5 in test
        folds = [
            ([0, 2, 4], [1, 3, 5]),
            ([1, 3, 5], [0, 2, 4]),
        ]

        # When parent & achiral disjointness are enforced, both violations are caught
        res = validate_split_integrity(
            folds=folds,
            identities=identities,
            labels=labels,
            group_keys=group_keys,
            enforce_group_disjointness=True,
            enforce_parent_disjointness=True,
            enforce_achiral_disjointness=True,
            raise_on_violation=False,
        )
        assert res.is_valid is False
        assert res.zero_parent_smiles_overlap is False
        assert res.zero_achiral_smiles_overlap is False
        assert res.parent_overlap_pair_count > 0
        assert res.achiral_overlap_pair_count > 0


# ---------------------------------------------------------------------------
# 6. Fixed-Seed Reproducibility and Alternative-Seed Behavior
# ---------------------------------------------------------------------------


class TestSeedReproducibilityAndVariation:
    """Category 6: Verifies fixed-seed reproducibility and multi-seed variation (42, 1042, 2042)."""

    def test_fixed_seed_reproducibility_and_prespecified_clintox_seeds_differ(self) -> None:
        clintox_cohort = build_clintox_evaluation_cohort(compute_fingerprints=False)

        # Same seed (42) produces 100% identical fold indices
        rep_42_a, _ = generate_grouped_kfold_splits(
            clintox_cohort.identities,
            clintox_cohort.labels,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
            n_splits=5,
            seed=42,
        )
        rep_42_b, _ = generate_grouped_kfold_splits(
            clintox_cohort.identities,
            clintox_cohort.labels,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
            n_splits=5,
            seed=42,
        )
        for fa, fb in zip(rep_42_a.folds, rep_42_b.folds):
            assert fa.train_indices == fb.train_indices
            assert fa.test_indices == fb.test_indices

        # Prespecified ClinTox seeds (42, 1042, 2042) produce meaningfully distinct partitions
        rep_1042, _ = generate_grouped_kfold_splits(
            clintox_cohort.identities,
            clintox_cohort.labels,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
            n_splits=5,
            seed=1042,
        )
        rep_2042, _ = generate_grouped_kfold_splits(
            clintox_cohort.identities,
            clintox_cohort.labels,
            strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
            n_splits=5,
            seed=2042,
        )

        seed_summary = compute_split_seed_variation_summary([rep_42_a, rep_1042, rep_2042])
        assert seed_summary["seeds"] == list(CLINTOX_PRESPECIFIED_SEEDS)
        assert seed_summary["all_seed_pairs_produce_distinct_partitions"] is True
        for pair in seed_summary["pairwise_seed_comparisons"]:
            assert pair["identical_fold_count"] == 0
            assert pair["mean_max_jaccard"] < 0.50


# ---------------------------------------------------------------------------
# 7. Endpoint-Specific Missing-Label Isolation
# ---------------------------------------------------------------------------


class TestEndpointSpecificMissingLabelIsolation:
    """Category 7: Verifies that missing assay labels (NaN) are isolated per endpoint."""

    def test_tox21_endpoint_cohort_isolates_missing_labels_and_duplicates(self) -> None:
        synthetic_df = pd.DataFrame(
            {
                "smiles": [
                    "c1ccccc1C",        # row 0: valid for both
                    "c1ccccc1O",        # row 1: NR-AR=1, SR-MMP=NaN
                    "c1ccncc1C",        # row 2: NR-AR=NaN, SR-MMP=0
                    "invalid_smiles",   # row 3: invalid SMILES
                    "CCO",              # row 4: NR-AR=0, SR-MMP=1
                    "OCC",              # row 5: duplicate of CCO with conflicting NR-AR=1, SR-MMP=1
                ],
                "NR-AR": [0.0, 1.0, np.nan, 1.0, 0.0, 1.0],
                "SR-MMP": [0.0, np.nan, 0.0, 0.0, 1.0, 1.0],
            }
        )

        # Endpoint 1: NR-AR under conservative_active
        cohort_ar_cons = build_tox21_endpoint_cohort(
            "NR-AR", df_or_csv_path=synthetic_df, duplicate_policy="conservative_active"
        )
        assert cohort_ar_cons.n_raw_rows == 6
        assert cohort_ar_cons.n_invalid_smiles_excluded == 1
        assert cohort_ar_cons.n_missing_label_excluded == 1  # row 2
        assert cohort_ar_cons.n_duplicate_rows_resolved == 1  # CCO/OCC merged
        assert cohort_ar_cons.n_samples == 3
        assert cohort_ar_cons.n_positives == 2  # c1ccccc1O (1) + CCO/OCC resolved to 1
        assert cohort_ar_cons.n_negatives == 1  # c1ccccc1C (0)

        # Endpoint 1: NR-AR under drop_conflicts drops CCO/OCC
        cohort_ar_drop = build_tox21_endpoint_cohort(
            "NR-AR", df_or_csv_path=synthetic_df, duplicate_policy="drop_conflicts"
        )
        assert cohort_ar_drop.n_samples == 2
        assert "CCO" not in cohort_ar_drop.canonical_smiles

        # Endpoint 2: SR-MMP under conservative_active retains row 2 (c1ccncc1C -> Cc1cccnc1) even though NR-AR was NaN
        cohort_mmp = build_tox21_endpoint_cohort(
            "SR-MMP", df_or_csv_path=synthetic_df, duplicate_policy="conservative_active"
        )
        assert cohort_mmp.n_samples == 3
        assert "Cc1cccnc1" in cohort_mmp.canonical_smiles
        assert "Oc1ccccc1" not in cohort_mmp.canonical_smiles  # NaN in SR-MMP


# ---------------------------------------------------------------------------
# 8. Metric Calculations on Hand-Verifiable Synthetic Data
# ---------------------------------------------------------------------------


class TestHandVerifiableMetricCalculations:
    """Category 8: Verifies exact metric math on hand-verifiable synthetic data."""

    def test_exact_hand_calculated_metrics_and_thresholds(self) -> None:
        # 10 samples: 2 positives (indices 0, 1), 8 negatives (indices 2..9) -> prevalence = 0.20
        y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        y_prob = [0.90, 0.40, 0.60, 0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        cyclic_mask = [True, True, True, True, True, False, False, False, False, False]

        # Hand verification:
        # Positives have probs [0.90, 0.40]; Negatives have [0.60, 0.30, 0.10 x 6].
        # Pairwise (pos, neg) comparisons (2 * 8 = 16 pairs):
        # - pos=0.90 beats all 8 negatives (8 wins)
        # - pos=0.40 beats 7 negatives (loses only to neg=0.60) (7 wins)
        # -> ROC-AUC = 15 / 16 = 0.9375
        #
        # Average Precision (PR-AUC):
        # Sorted descending:
        # rank 1: p=0.90 (y=1) -> TP=1, FP=0 -> precision = 1/1 = 1.0
        # rank 2: p=0.60 (y=0) -> TP=1, FP=1
        # rank 3: p=0.40 (y=1) -> TP=2, FP=1 -> precision = 2/3 = 0.6666667
        # -> AP = (1.0 + 2/3) / 2 = 5/6 = 0.833333
        # -> PR-AUC lift = 5/6 - 0.20 = 0.633333
        # -> PR-AUC ratio = (5/6) / 0.20 = 25/6 = 4.166667
        #
        # At fixed threshold 0.50:
        # predicted 1: idx 0 (y=1, p=0.90) and idx 2 (y=0, p=0.60) -> TP=1, FP=1, FN=1, TN=7
        # precision = 1/2 = 0.50, recall = 1/2 = 0.50, specificity = 7/8 = 0.875,
        # F1 = 0.50, balanced_accuracy = 0.5*(0.50 + 0.875) = 0.6875, accuracy = 8/10 = 0.80.
        #
        # At exploratory training-prevalence threshold 0.25:
        # predicted 1: idx 0 (y=1), idx 2 (y=0), idx 1 (y=1), idx 3 (y=0) -> TP=2, FP=2, FN=0, TN=6
        # precision = 2/4 = 0.50, recall = 2/2 = 1.00, specificity = 6/8 = 0.75,
        # F1 = 2/3 = 0.666667, balanced_accuracy = 0.875.

        metrics = compute_binary_evaluation_metrics(
            y_true=y_true,
            y_prob=y_prob,
            fixed_threshold=0.50,
            exploratory_train_prevalence_threshold=0.25,
            min_samples=5,
            min_positives=2,
            min_negatives=2,
        )

        assert metrics.insufficient_support is False
        assert metrics.n_samples == 10
        assert metrics.n_positives == 2
        assert metrics.n_negatives == 8
        assert math.isclose(metrics.prevalence, 0.20, abs_tol=1e-6)
        assert math.isclose(metrics.roc_auc or 0.0, 0.9375, abs_tol=1e-6)
        assert math.isclose(metrics.pr_auc or 0.0, 5.0 / 6.0, abs_tol=1e-6)
        assert math.isclose(metrics.pr_auc_lift or 0.0, (5.0 / 6.0) - 0.20, abs_tol=1e-6)
        assert math.isclose(metrics.pr_auc_ratio or 0.0, (5.0 / 6.0) / 0.20, abs_tol=1e-6)

        ft = metrics.fixed_threshold_metrics
        assert ft is not None
        assert ft.confusion_matrix == ((7, 1), (1, 1))
        assert math.isclose(ft.precision, 0.50, abs_tol=1e-6)
        assert math.isclose(ft.recall, 0.50, abs_tol=1e-6)
        assert math.isclose(ft.specificity, 0.875, abs_tol=1e-6)
        assert math.isclose(ft.f1, 0.50, abs_tol=1e-6)
        assert math.isclose(ft.balanced_accuracy, 0.6875, abs_tol=1e-6)

        et = metrics.exploratory_prevalence_threshold_metrics
        assert et is not None
        assert et.confusion_matrix == ((6, 2), (0, 2))
        assert math.isclose(et.recall, 1.00, abs_tol=1e-6)
        assert math.isclose(et.specificity, 0.75, abs_tol=1e-6)

        # Full vs cyclic-only subset reporting
        both = compute_full_and_cyclic_metrics(
            y_true=y_true,
            y_prob=y_prob,
            cyclic_mask=cyclic_mask,
            min_samples=4,
            min_positives=2,
            min_negatives=2,
        )
        assert both["full_cohort"].n_samples == 10
        assert both["cyclic_only"].n_samples == 5
        assert both["cyclic_only"].n_positives == 2
        assert both["cyclic_only"].n_negatives == 3


# ---------------------------------------------------------------------------
# 9. Degenerate Folds and Insufficient Positive/Negative Support
# ---------------------------------------------------------------------------


class TestDegenerateFoldsAndInsufficientSupport:
    """Category 9: Verifies clear failure and flag handling on degenerate/sparse cohorts."""

    def test_infeasible_kfold_raises_clear_error_without_weakening_constraints(self) -> None:
        smiles_list, _ = _build_synthetic_scaffold_cohort()
        # Only 2 positive samples in Group 1 (benzene) -> only 1 positive scaffold group < 5 folds
        sparse_labels = [1, 1] + [0] * (len(smiles_list) - 2)

        with pytest.raises(
            InfeasibleFoldConfigurationError,
            match="Positive scaffold group count \\(1\\) is smaller than requested n_splits \\(5\\)",
        ):
            generate_grouped_kfold_splits(
                smiles_list,
                sparse_labels,
                strategy=STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
                n_splits=5,
                seed=42,
            )

    def test_single_class_and_invalid_probability_handling(self) -> None:
        # All-negative fold -> ROC-AUC and PR-AUC are undefined (None), insufficient_support=True,
        # while Brier and LogLoss remain defined
        m_neg_only = compute_binary_evaluation_metrics(
            y_true=[0, 0, 0, 0, 0],
            y_prob=[0.1, 0.2, 0.05, 0.3, 0.1],
        )
        assert m_neg_only.insufficient_support is True
        assert m_neg_only.roc_auc is None
        assert m_neg_only.pr_auc is None
        assert m_neg_only.brier_score is not None
        assert m_neg_only.log_loss is not None

        # Invalid probabilities (NaN or > 1.0) raise ScaffoldGeneralizationError
        with pytest.raises(ScaffoldGeneralizationError, match="NaN or infinite"):
            compute_binary_evaluation_metrics([0, 1], [0.2, float("nan")])

        with pytest.raises(ScaffoldGeneralizationError, match="within \\[0.0, 1.0\\]"):
            compute_binary_evaluation_metrics([0, 1], [0.2, 1.2])

    def test_fold_and_pooled_summary_separates_fold_dispersion_and_multi_seed_repeats(self) -> None:
        # 2 valid folds + 1 degenerate fold
        f1 = compute_binary_evaluation_metrics(
            [1, 1, 0, 0, 0, 0], [0.9, 0.8, 0.2, 0.1, 0.3, 0.1], min_samples=4, min_positives=2, min_negatives=2
        )
        f2 = compute_binary_evaluation_metrics(
            [1, 1, 0, 0, 0, 0], [0.7, 0.4, 0.6, 0.2, 0.1, 0.1], min_samples=4, min_positives=2, min_negatives=2
        )
        f3_degen = compute_binary_evaluation_metrics(
            [0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.1, 0.05, 0.1, 0.2], min_samples=4, min_positives=2, min_negatives=2
        )
        pooled_y = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pooled_p = [0.9, 0.8, 0.2, 0.1, 0.3, 0.1, 0.7, 0.4, 0.6, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.1, 0.2]

        summary = summarize_fold_and_pooled_metrics(
            [f1, f2, f3_degen],
            pooled_y_true=pooled_y,
            pooled_y_prob=pooled_p,
            min_samples=4,
            min_positives=2,
            min_negatives=2,
        )
        assert summary.n_folds == 3
        assert summary.n_Any_insufficient_support_folds == 1
        assert summary.fold_dispersion["roc_auc"].n_valid_folds == 2
        assert summary.fold_dispersion["roc_auc"].n_undefined_folds == 1
        assert summary.pooled_oof_metrics.n_samples == 18

        # Multi-seed repeat summary without pooling dependent repeats
        rep_summary = summarize_repeated_cv_runs({42: summary, 1042: summary})
        assert rep_summary["seeds"] == [42, 1042]
        assert rep_summary["across_seed_pooled_oof_summary"]["roc_auc"]["sd"] == 0.0


# ---------------------------------------------------------------------------
# 10. Deterministic Scaffold-Cluster Bootstrap and Invalid-Replicate Accounting
# ---------------------------------------------------------------------------


class TestScaffoldClusterBootstrapAndPairedComparisons:
    """Category 10: Verifies deterministic scaffold-cluster bootstrap and invalid-replicate accounting."""

    def test_cluster_bootstrap_determinism_and_sparse_positive_cluster_accounting(self) -> None:
        # Case A: Sufficient positive clusters (6 positive clusters, 6 negative clusters)
        y_true: List[int] = []
        y_prob_a: List[float] = []
        y_prob_b: List[float] = []
        clusters: List[str] = []

        for c_idx in range(6):
            # Positive-containing cluster (1 pos, 3 neg)
            cid = f"pos_cluster_{c_idx}"
            y_true.extend([1, 0, 0, 0])
            y_prob_a.extend([0.75 - 0.03 * c_idx, 0.25, 0.20, 0.15])
            y_prob_b.extend([0.85 - 0.02 * c_idx, 0.15, 0.10, 0.10])
            clusters.extend([cid] * 4)

        for c_idx in range(6):
            # Negative-only cluster (4 neg)
            cid = f"neg_cluster_{c_idx}"
            y_true.extend([0, 0, 0, 0])
            y_prob_a.extend([0.30, 0.10, 0.15, 0.05])
            y_prob_b.extend([0.20, 0.05, 0.10, 0.05])
            clusters.extend([cid] * 4)

        boot_1 = compute_scaffold_cluster_bootstrap(
            y_true, y_prob_a, clusters, n_bootstraps=100, seed=42, min_positive_clusters=5
        )
        boot_2 = compute_scaffold_cluster_bootstrap(
            y_true, y_prob_a, clusters, n_bootstraps=100, seed=42, min_positive_clusters=5
        )
        assert boot_1["roc_auc"] == boot_2["roc_auc"]
        assert boot_1["roc_auc"].insufficient_cluster_support is False
        assert boot_1["roc_auc"].ci_lower is not None
        assert boot_1["roc_auc"].ci_upper is not None
        assert boot_1["roc_auc"].ci_lower <= (boot_1["roc_auc"].point_estimate or 0.0) <= boot_1["roc_auc"].ci_upper

        # Paired comparison preserves molecule-level pairing and uses identical cluster draws
        paired = compare_paired_oof_predictions(
            y_true=y_true,
            prob_regime_a=y_prob_a,
            prob_regime_b=y_prob_b,
            cluster_ids=clusters,
            n_bootstraps=100,
            seed=42,
            min_positive_clusters=5,
        )
        assert paired["brier_score"].insufficient_cluster_support is False
        assert paired["brier_score"].delta_a_minus_b is not None
        assert paired["brier_score"]. delta_a_minus_b > 0.0  # Regime A has higher Brier loss than Regime B

        # Case B: Sparse positive cluster (only 1 cluster contains positives out of 10 clusters)
        # Resampling clusters with replacement will frequently miss that single positive cluster
        sparse_y = [1, 1] + [0] * 18
        sparse_p = [0.8, 0.7] + [0.1] * 18
        sparse_c = [f"c_{i // 2}" for i in range(20)]  # 10 clusters of size 2; only c_0 has positives

        sparse_boot = compute_scaffold_cluster_bootstrap(
            sparse_y, sparse_p, sparse_c, n_bootstraps=100, seed=42, min_positive_clusters=3
        )
        roc_res = sparse_boot["roc_auc"]
        assert roc_res.n_total_clusters == 10
        assert roc_res.n_positive_clusters == 1
        assert roc_res.insufficient_cluster_support is True
        assert roc_res.n_invalid_replicates > 0
        assert roc_res.n_valid_replicates + roc_res.n_invalid_replicates == 100
        assert roc_res.ci_lower is None and roc_res.ci_upper is None


# ---------------------------------------------------------------------------
# 11. No File or Artifact Mutation & 12. Production Regression Preservation
# ---------------------------------------------------------------------------


class TestArtifactIntegrityAndProductionPreservation:
    """Categories 11 & 12: Verifies zero artifact mutation and unchanged production inference/ranking."""

    def test_all_31_protected_artifact_checksums_remain_intact(self) -> None:
        count, mismatches = _verify_all_31_protected_checksums()
        assert count == 32
        assert mismatches == []
        assert "Stage 1" in STAGE1_CALIBRATION_SCOPE_NOTE

    def test_production_inference_and_ranking_regression_outputs_unchanged(self) -> None:
        orch = ScreeningOrchestrator()

        expected_baselines: Dict[str, Dict[str, object]] = {
            "Aspirin": {
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "clintox_prob": 0.6659,
                "clintox_uncal_prob": 1.0,
                "clintox_class": 1,
                "clintox_ad_status": "WITHIN_DOMAIN",
                "clintox_ad_score": 1.0,
                "tox21_active_count": 0,
                "nr_ar_prob": 0.0133,
                "nr_ar_cal_prob": 0.0166,
                "nr_ar_ad_score": 1.0,
                "sr_mmp_prob": 0.1367,
                "sr_mmp_cal_prob": 0.0451,
                "ranking_overall": 66.7,
                "ranking_safety": 33.41,
                "ranking_druglikeness": 100.0,
                "ranking_similarity": 100.0,
            },
            "Caffeine": {
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "clintox_prob": 0.0805,
                "clintox_uncal_prob": 0.1233,
                "clintox_class": 0,
                "clintox_ad_status": "WITHIN_DOMAIN",
                "clintox_ad_score": 1.0,
                "tox21_active_count": 1,
                "nr_ar_prob": 0.0033,
                "nr_ar_cal_prob": 0.0157,
                "nr_ar_ad_score": 0.641,
                "sr_mmp_prob": 0.0233,
                "sr_mmp_cal_prob": 0.0215,
                "ranking_overall": 95.97,
                "ranking_safety": 91.95,
                "ranking_druglikeness": 100.0,
                "ranking_similarity": 100.0,
            },
            "Ethanol": {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "clintox_prob": 0.0347,
                "clintox_uncal_prob": 0.0767,
                "clintox_class": 0,
                "clintox_ad_status": "WITHIN_DOMAIN",
                "clintox_ad_score": 1.0,
                "tox21_active_count": 0,
                "nr_ar_prob": 0.0,
                "nr_ar_cal_prob": 0.0154,
                "nr_ar_ad_score": 1.0,
                "sr_mmp_prob": 0.0067,
                "sr_mmp_cal_prob": 0.0193,
                "ranking_overall": 98.27,
                "ranking_safety": 96.53,
                "ranking_druglikeness": 100.0,
                "ranking_similarity": 100.0,
            },
            "Cisplatin": {
                "smiles": "[NH3][Pt]([NH3])(Cl)Cl",
                "canonical_smiles": "[NH3]->[Pt](<-[NH3])([Cl])[Cl]",
                "clintox_prob": 0.0912,
                "clintox_uncal_prob": 0.2633,
                "clintox_class": 0,
                "clintox_ad_status": "OUTSIDE_DOMAIN",
                "clintox_ad_score": 0.125,
                "tox21_active_count": 1,
                "nr_ar_prob": 0.0,
                "nr_ar_cal_prob": 0.0154,
                "nr_ar_ad_score": 0.3333,
                "sr_mmp_prob": 0.1467,
                "sr_mmp_cal_prob": 0.0481,
                "ranking_overall": 77.94,
                "ranking_safety": 90.88,
                "ranking_druglikeness": 100.0,
                "ranking_similarity": 12.5,
            },
        }

        for name, exp in expected_baselines.items():
            res = orch.analyze_single_molecule(str(exp["smiles"]))
            assert res.validation.canonical_smiles == exp["canonical_smiles"], f"{name} canonical_smiles"
            assert res.clintox is not None
            assert math.isclose(res.clintox.probability, float(exp["clintox_prob"]), abs_tol=1e-4)
            assert math.isclose(
                res.clintox.uncalibrated_probability or 0.0, float(exp["clintox_uncal_prob"]), abs_tol=1e-4
            )
            assert res.clintox.predicted_class == exp["clintox_class"]
            assert res.clintox.applicability_domain_status == exp["clintox_ad_status"]
            assert math.isclose(res.clintox.applicability_score or 0.0, float(exp["clintox_ad_score"]), abs_tol=1e-4)

            assert res.tox21 is not None
            assert res.tox21.active_endpoints_count == exp["tox21_active_count"]
            assert math.isclose(res.tox21.endpoints["NR-AR"].probability, float(exp["nr_ar_prob"]), abs_tol=1e-4)
            assert math.isclose(
                res.tox21.endpoints["NR-AR"].calibrated_probability or 0.0,
                float(exp["nr_ar_cal_prob"]),
                abs_tol=1e-4,
            )
            assert math.isclose(
                res.tox21.endpoints["NR-AR"].applicability_score or 0.0,
                float(exp["nr_ar_ad_score"]),
                abs_tol=1e-4,
            )
            assert math.isclose(res.tox21.endpoints["SR-MMP"].probability, float(exp["sr_mmp_prob"]), abs_tol=1e-4)
            assert math.isclose(
                res.tox21.endpoints["SR-MMP"].calibrated_probability or 0.0,
                float(exp["sr_mmp_cal_prob"]),
                abs_tol=1e-4,
            )

            assert res.prioritization is not None
            assert math.isclose(res.prioritization.overall_score, float(exp["ranking_overall"]), abs_tol=1e-2)
            assert math.isclose(res.prioritization.safety_score, float(exp["ranking_safety"]), abs_tol=1e-2)
            assert math.isclose(res.prioritization.druglikeness_score, float(exp["ranking_druglikeness"]), abs_tol=1e-2)
            assert math.isclose(res.prioritization.similarity_score, float(exp["ranking_similarity"]), abs_tol=1e-2)
