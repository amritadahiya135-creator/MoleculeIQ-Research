"""Phase B.3 Stage 1: Scaffold Generalization & Out-of-Scaffold Robustness Evaluation Engine.

Provides an isolated, deterministic, and non-destructive evaluation framework for:
1. Molecular identity analysis (exact canonical SMILES, achiral SMILES, largest covalent
   parent fragment for multi-fragment salts, cyclic Bemis-Murcko scaffolds, generic
   Bemis-Murcko frameworks, and deterministic acyclic skeleton/formula groupings).
2. Group-size distribution and k-fold feasibility diagnostics, including explicit
   detection of oversized scaffold groups (size > N / n_splits) and parent/stereoisomer
   label-conflict auditing.
3. Stratified scaffold-grouped k-fold cross-validation (StratifiedGroupKFold) and matched
   stratified random k-fold cross-validation (StratifiedKFold) with programmatic
   split-integrity and leakage verification.
4. Endpoint-specific cohort construction for ClinTox and all 12 Tox21 assays with strict
   missing-label (NaN) isolation.
5. Discrimination, probability-quality, prevalence-adjusted PR-AUC (lift and ratio), and
   fixed-threshold vs. exploratory training-prevalence threshold metrics across both full
   cohorts and cyclic-only subsets.
6. Separate reporting of fold-level dispersion, pooled out-of-fold metrics, multi-seed
   repeat summaries (seeds 42, 1042, 2042), and molecule-paired scaffold-cluster block
   bootstrap confidence intervals with invalid-replicate accounting.

Stage 1 Scope & Safeguards:
- Pure evaluation and diagnostic logic: zero import-time training, zero file writes, and
  zero mutation of production models, calibration artifacts, metadata, or checksums.
- Does not fit or replace probability calibration models (nested scaffold-grouped
  calibration is documented as a separate future evaluation outside Stage 1 scope).
- Does not conflate internal scaffold-holdout generalization with external dataset
  validation, applicability-domain Tanimoto similarity, or clinical safety.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import logging
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.chemistry.scaffolds import generate_bemis_murcko_scaffold
from src.chemistry.validation import validate_smiles
from src.config import CLINTOX_DATA_PATH, TOX21_DATA_PATH, resolve_path
from src.models.clintox import prepare_clintox_dataset
from src.models.tox21 import TOX21_ENDPOINTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Prespecified Protocol Defaults
# ---------------------------------------------------------------------------

PHASE_B3_SCHEMA_VERSION = "1.0.0-stage1"

# Prespecified ClinTox repeat seeds for split-assignment variance (Section B.3)
CLINTOX_PRESPECIFIED_SEEDS: Tuple[int, ...] = (42, 1042, 2042)
DEFAULT_CV_SPLITS: int = 5
DEFAULT_RANDOM_SEED: int = 42

# Classification threshold defaults (Section B.4)
DEFAULT_FIXED_THRESHOLD: float = 0.50
LOG_LOSS_CLIP_EPS: float = 1e-15

# Support governance thresholds
DEFAULT_MIN_FOLD_POSITIVES: int = 1
DEFAULT_MIN_FOLD_NEGATIVES: int = 1
DEFAULT_MIN_METRIC_SAMPLES: int = 10
DEFAULT_MIN_METRIC_POSITIVES: int = 5
DEFAULT_MIN_POSITIVE_CLUSTERS_FOR_BOOTSTRAP: int = 5

# Supported deterministic scaffold-grouping strategies (Section B.1)
STRATEGY_EXACT_CANONICAL = "exact_canonical"
STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC = "murcko_cyclic_acyclic_generic"
STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON = "murcko_cyclic_acyclic_singleton"
STRATEGY_MURCKO_CYCLIC_ACYCLIC_FORMULA = "murcko_cyclic_acyclic_formula"
STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLE_BUCKET = "murcko_cyclic_acyclic_single_bucket"
STRATEGY_GENERIC_MURCKO_FRAMEWORK = "generic_murcko_framework"

SUPPORTED_GROUPING_STRATEGIES: Tuple[str, ...] = (
    STRATEGY_EXACT_CANONICAL,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_FORMULA,
    STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLE_BUCKET,
    STRATEGY_GENERIC_MURCKO_FRAMEWORK,
)

# Scientific scope and calibration boundary note (Section B.5)
STAGE1_CALIBRATION_SCOPE_NOTE = (
    "Stage 1 evaluates raw/fixed-protocol model predictions without fitting or replacing "
    "calibration models. Nested scaffold-grouped probability calibration is reserved as "
    "a separate future evaluation outside Stage 1 scope."
)

CLUSTER_BOOTSTRAP_LIMITATIONS_NOTE = (
    "Scaffold-cluster block bootstrap resamples scaffold clusters with replacement to "
    "account for within-scaffold dependence. When positive events are concentrated in a "
    "small number of scaffold clusters, bootstrap replicates may exhibit discrete jumps "
    "or draw zero positives (tracked via n_invalid_replicates); cluster bootstrap CIs "
    "are not guaranteed to be wider than naive molecule-level bootstrap CIs."
)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class ScaffoldGeneralizationError(ValueError):
    """Base exception for scaffold generalization evaluation failures."""


class InfeasibleFoldConfigurationError(ScaffoldGeneralizationError):
    """Raised when a cohort or grouping strategy cannot support the requested k-fold split."""


class SplitIntegrityError(ScaffoldGeneralizationError):
    """Raised when a train/test split violates index, group, or structure disjointness."""


# ---------------------------------------------------------------------------
# 1. Molecular Structure Identity & Scaffold / Acyclic Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolecularStructureIdentity:
    """Comprehensive structural identity and scaffold decomposition for a single molecule."""

    raw_smiles: str
    canonical_smiles: str
    achiral_canonical_smiles: str
    parent_canonical_smiles: str
    parent_achiral_smiles: str
    is_multifragment_salt: bool
    has_specified_stereo: bool
    is_cyclic: bool
    parent_is_cyclic: bool
    bemis_murcko_scaffold: str
    parent_bemis_murcko_scaffold: str
    generic_murcko_framework: str
    parent_generic_murcko_framework: str
    acyclic_generic_skeleton: str
    parent_acyclic_generic_skeleton: str
    acyclic_formula_class: str
    parent_acyclic_formula_class: str


def _select_largest_covalent_fragment(mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
    """Return the largest covalent fragment of an RDKit Mol and whether multiple fragments existed.

    Fragments are ranked deterministically by:
    1. Number of heavy atoms (descending)
    2. Total number of atoms (descending)
    3. Canonical isomeric SMILES string (ascending tie-breaker for deterministic ordering)
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return mol, False

    def _frag_sort_key(frag: Chem.Mol) -> Tuple[int, int, str]:
        smi = Chem.MolToSmiles(frag, isomericSmiles=True, canonical=True)
        return (-int(frag.GetNumHeavyAtoms()), -int(frag.GetNumAtoms()), smi)

    sorted_frags = sorted(frags, key=_frag_sort_key)
    return sorted_frags[0], True


def extract_generic_murcko_scaffold(mol_or_smiles: Union[str, Chem.Mol, None]) -> str:
    """Deterministically compute a generic carbon-skeleton framework from an RDKit Mol or SMILES.

    Standard RDKit ``MurckoScaffold.MakeScaffoldGeneric`` sets every atom to atomic number 6
    (Carbon, max valence 4) and every bond to single. On molecules containing pentavalent or
    hexavalent ring heteroatoms (e.g. sulfonyl/sulfinyl ring sulfur, ring phosphorus, or
    coordination metal complexes with degree > 4), ``MakeScaffoldGeneric`` raises an
    ``AtomValenceException``.

    This function first checks whether all atoms have degree <= 4. If so, it invokes
    ``MurckoScaffold.MakeScaffoldGeneric`` under an RDKit error-log guard. If any atom has
    degree > 4 or if ``MakeScaffoldGeneric`` fails sanitization, it falls back to a
    deterministic graph normalization that converts all bonds to single non-aromatic bonds
    and converts all atoms with degree <= 4 to neutral Carbon while preserving the atomic
    number of degree > 4 centers so valence rules remain valid.
    """
    if mol_or_smiles is None:
        return ""

    if isinstance(mol_or_smiles, str):
        clean = mol_or_smiles.strip()
        if not clean:
            return ""
        mol = Chem.MolFromSmiles(clean)
        if mol is None:
            return ""
    else:
        mol = mol_or_smiles

    if mol.GetNumAtoms() == 0:
        return ""

    has_high_degree_atom = any(atom.GetDegree() > 4 for atom in mol.GetAtoms())

    if not has_high_degree_atom:
        RDLogger.DisableLog("rdApp.error")
        try:
            gen_mol = MurckoScaffold.MakeScaffoldGeneric(mol)
            return Chem.MolToSmiles(gen_mol, isomericSmiles=False, canonical=True)
        except Exception:
            pass
        finally:
            RDLogger.EnableLog("rdApp.error")

    # Safe fallback for hypervalent / coordination centers with degree > 4
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumExplicitHs(0)
        atom.SetIsotope(0)
        if atom.GetDegree() <= 4:
            atom.SetAtomicNum(6)

    RDLogger.DisableLog("rdApp.error")
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw, isomericSmiles=False, canonical=True)
    except Exception:
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    finally:
        RDLogger.EnableLog("rdApp.error")


def _compute_acyclic_formula_class(mol: Chem.Mol) -> str:
    """Compute a deterministic topological/compositional class key for an acyclic molecule."""
    n_heavy = int(mol.GetNumHeavyAtoms())
    n_carbon = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    n_het = n_heavy - n_carbon
    n_branch = sum(1 for a in mol.GetAtoms() if a.GetDegree() >= 3)
    return f"nHeavy={n_heavy}_nC={n_carbon}_nHet={n_het}_nBranch={n_branch}"


def extract_structure_identity(smiles: str) -> MolecularStructureIdentity:
    """Extract complete canonical, parent-fragment, achiral, and scaffold identities for a SMILES.

    Preserves the existing production ``validate_smiles`` canonicalization policy while
    separately computing parent-fragment and stereo-stripped representations for split
    auditing and sensitivity analyses.
    """
    val_res = validate_smiles(smiles)
    if not val_res.is_valid or not val_res.canonical_smiles:
        raise ScaffoldGeneralizationError(
            f"Invalid SMILES string '{smiles}': {val_res.error_message or 'validation failed'}"
        )

    canonical_smiles = val_res.canonical_smiles
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ScaffoldGeneralizationError(
            f"Failed to construct RDKit Mol from canonical SMILES '{canonical_smiles}'."
        )
    achiral_canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    has_specified_stereo = canonical_smiles != achiral_canonical_smiles

    parent_mol, is_multifrag = _select_largest_covalent_fragment(mol)
    if is_multifrag:
        parent_canonical_smiles = Chem.MolToSmiles(parent_mol, isomericSmiles=True, canonical=True)
        parent_achiral_smiles = Chem.MolToSmiles(parent_mol, isomericSmiles=False, canonical=True)
    else:
        parent_canonical_smiles = canonical_smiles
        parent_achiral_smiles = achiral_canonical_smiles

    # Raw canonical Murcko decomposition
    murcko_scaf = generate_bemis_murcko_scaffold(canonical_smiles)
    is_cyclic = bool(murcko_scaf != "")
    if is_cyclic:
        scaf_mol = Chem.MolFromSmiles(murcko_scaf)
        generic_framework = extract_generic_murcko_scaffold(scaf_mol)
        acyclic_skeleton = ""
        acyclic_formula = ""
    else:
        generic_framework = ""
        acyclic_skeleton = extract_generic_murcko_scaffold(mol)
        acyclic_formula = _compute_acyclic_formula_class(mol)

    # Parent-fragment Murcko decomposition
    if is_multifrag:
        parent_murcko_scaf = generate_bemis_murcko_scaffold(parent_canonical_smiles)
        parent_is_cyclic = bool(parent_murcko_scaf != "")
        if parent_is_cyclic:
            p_scaf_mol = Chem.MolFromSmiles(parent_murcko_scaf)
            parent_generic_framework = extract_generic_murcko_scaffold(p_scaf_mol)
            parent_acyclic_skeleton = ""
            parent_acyclic_formula = ""
        else:
            parent_generic_framework = ""
            parent_acyclic_skeleton = extract_generic_murcko_scaffold(parent_mol)
            parent_acyclic_formula = _compute_acyclic_formula_class(parent_mol)
    else:
        parent_is_cyclic = is_cyclic
        parent_murcko_scaf = murcko_scaf
        parent_generic_framework = generic_framework
        parent_acyclic_skeleton = acyclic_skeleton
        parent_acyclic_formula = acyclic_formula

    return MolecularStructureIdentity(
        raw_smiles=str(smiles),
        canonical_smiles=canonical_smiles,
        achiral_canonical_smiles=achiral_canonical_smiles,
        parent_canonical_smiles=parent_canonical_smiles,
        parent_achiral_smiles=parent_achiral_smiles,
        is_multifragment_salt=is_multifrag,
        has_specified_stereo=has_specified_stereo,
        is_cyclic=is_cyclic,
        parent_is_cyclic=parent_is_cyclic,
        bemis_murcko_scaffold=murcko_scaf,
        parent_bemis_murcko_scaffold=parent_murcko_scaf,
        generic_murcko_framework=generic_framework,
        parent_generic_murcko_framework=parent_generic_framework,
        acyclic_generic_skeleton=acyclic_skeleton,
        parent_acyclic_generic_skeleton=parent_acyclic_skeleton,
        acyclic_formula_class=acyclic_formula,
        parent_acyclic_formula_class=parent_acyclic_formula,
    )


def audit_cohort_structure_relationships(
    smiles_Or_identities: Sequence[Union[str, MolecularStructureIdentity]],
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Audit exact duplicates, multi-fragment salts, parent fragments, and stereoisomers.

    Separately quantifies whether distinct canonical SMILES share a largest covalent
    parent fragment (salts) or an achiral connectivity graph (stereoisomers), and counts
    label conflicts across shared parent/stereoisomer groups. This provides explicit
    empirical evidence for why salt and stereoisomer records must not have their labels
    silently merged without explicit chemical justification.
    """
    identities: List[MolecularStructureIdentity] = [
        item if isinstance(item, MolecularStructureIdentity) else extract_structure_identity(str(item))
        for item in smiles_Or_identities
    ]
    n_total = len(identities)

    if labels is not None and len(labels) != n_total:
        raise ScaffoldGeneralizationError(
            f"Length of labels ({len(labels)}) must match length of structures ({n_total})."
        )

    y_arr = np.asarray(labels, dtype=int) if labels is not None else None

    exact_groups: Dict[str, List[int]] = defaultdict(list)
    parent_groups: Dict[str, List[int]] = defaultdict(list)
    achiral_groups: Dict[str, List[int]] = defaultdict(list)
    parent_achiral_groups: Dict[str, List[int]] = defaultdict(list)

    n_salts = 0
    n_stereo = 0
    n_scaffold_changed_by_parent = 0

    for idx, ident in enumerate(identities):
        exact_groups[ident.canonical_smiles].append(idx)
        parent_groups[ident.parent_canonical_smiles].append(idx)
        achiral_groups[ident.achiral_canonical_smiles].append(idx)
        parent_achiral_groups[ident.parent_achiral_smiles].append(idx)

        if ident.is_multifragment_salt:
            n_salts += 1
        if ident.has_specified_stereo:
            n_stereo += 1
        if ident.bemis_murcko_scaffold != ident.parent_bemis_murcko_scaffold:
            n_scaffold_changed_by_parent += 1

    def _count_multi_and_conflicts(group_map: Dict[str, List[int]]) -> Tuple[int, int]:
        multi_groups = sum(1 for idxs in group_map.values() if len(idxs) > 1)
        if y_arr is None:
            return multi_groups, 0
        conflict_groups = sum(
            1
            for idxs in group_map.values()
            if len(idxs) > 1 and len(set(int(y_arr[i]) for i in idxs)) > 1
        )
        return multi_groups, conflict_groups

    exact_multi, exact_conflicts = _count_multi_and_conflicts(exact_groups)
    parent_multi, parent_conflicts = _count_multi_and_conflicts(parent_groups)
    achiral_multi, achiral_conflicts = _count_multi_and_conflicts(achiral_groups)
    pa_multi, pa_conflicts = _count_multi_and_conflicts(parent_achiral_groups)

    return {
        "n_records": n_total,
        "n_unique_exact_canonical": len(exact_groups),
        "n_exact_duplicate_groups": exact_multi,
        "n_exact_label_conflict_groups": exact_conflicts,
        "n_multifragment_salt_records": n_salts,
        "n_records_where_parent_changes_murcko_scaffold": n_scaffold_changed_by_parent,
        "n_unique_parent_fragments": len(parent_groups),
        "n_shared_parent_fragment_groups": parent_multi,
        "n_parent_fragment_label_conflict_groups": parent_conflicts,
        "n_specified_stereo_records": n_stereo,
        "n_unique_achiral_structures": len(achiral_groups),
        "n_shared_stereoisomer_groups": achiral_multi,
        "n_stereoisomer_label_conflict_groups": achiral_conflicts,
        "n_unique_parent_achiral_structures": len(parent_achiral_groups),
        "n_shared_parent_achiral_groups": pa_multi,
        "n_parent_achiral_label_conflict_groups": pa_conflicts,
    }


# ---------------------------------------------------------------------------
# 2. Scaffold Group Assignment & Group-Size / Oversized-Group Diagnostics
# ---------------------------------------------------------------------------


def assign_scaffold_group_key(
    identity: MolecularStructureIdentity,
    strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    use_parent_fragment: bool = False,
    merge_stereoisomer_groups: bool = False,
) -> str:
    """Assign a deterministic scaffold group key to a molecule under the specified strategy.

    Parameters
    ----------
    identity : MolecularStructureIdentity
        Pre-extracted structural identity for the molecule.
    strategy : str
        One of ``SUPPORTED_GROUPING_STRATEGIES``:
        - ``"exact_canonical"``: Exact canonical SMILES (or parent/achiral variant if flags set).
        - ``"murcko_cyclic_acyclic_generic"``: Cyclic Bemis-Murcko scaffold for cyclic
          molecules; generic carbon skeleton for acyclic molecules.
        - ``"murcko_cyclic_acyclic_singleton"``: Cyclic Bemis-Murcko scaffold for cyclic
          molecules; individual canonical SMILES singleton group for each acyclic molecule.
        - ``"murcko_cyclic_acyclic_formula"``: Cyclic Bemis-Murcko scaffold for cyclic
          molecules; heavy-atom/carbon/heteroatom/branching class for acyclic molecules.
        - ``"murcko_cyclic_acyclic_single_bucket"``: Cyclic Bemis-Murcko scaffold for cyclic
          molecules; single shared ``"acyc_all"`` bucket for all acyclic molecules
          (diagnostic strategy illustrating oversized acyclic grouping).
        - ``"generic_murcko_framework"``: Generic Bemis-Murcko topological framework for
          cyclic molecules; generic carbon skeleton for acyclic molecules.
    use_parent_fragment : bool, default=False
        If True, derives the group key from the largest covalent parent fragment so salt
        counterions do not separate a parent structure across train/test splits. Does not
        alter sample count or labels.
    merge_stereoisomer_groups : bool, default=False
        If True, uses stereo-stripped canonical SMILES when forming exact or acyclic-singleton
        keys so stereoisomers stay in the same group. (Bemis-Murcko scaffolds already strip
        chirality via ``includeChirality=False``.)
    """
    if strategy not in SUPPORTED_GROUPING_STRATEGIES:
        raise ScaffoldGeneralizationError(
            f"Unsupported grouping strategy '{strategy}'. Supported strategies: {SUPPORTED_GROUPING_STRATEGIES}"
        )

    if use_parent_fragment:
        smi = identity.parent_achiral_smiles if merge_stereoisomer_groups else identity.parent_canonical_smiles
        is_cyc = identity.parent_is_cyclic
        murcko = identity.parent_bemis_murcko_scaffold
        gen_murcko = identity.parent_generic_murcko_framework
        acyc_skel = identity.parent_acyclic_generic_skeleton
        acyc_form = identity.parent_acyclic_formula_class
    else:
        smi = identity.achiral_canonical_smiles if merge_stereoisomer_groups else identity.canonical_smiles
        is_cyc = identity.is_cyclic
        murcko = identity.bemis_murcko_scaffold
        gen_murcko = identity.generic_murcko_framework
        acyc_skel = identity.acyclic_generic_skeleton
        acyc_form = identity.acyclic_formula_class

    if strategy == STRATEGY_EXACT_CANONICAL:
        return f"exact:{smi}"

    if strategy == STRATEGY_GENERIC_MURCKO_FRAMEWORK:
        if is_cyc:
            return f"gen_murcko:{gen_murcko}"
        return f"acyc_skeleton:{acyc_skel}"

    # All remaining strategies use cyclic Bemis-Murcko scaffold for cyclic compounds
    if is_cyc:
        return f"cyc_murcko:{murcko}"

    if strategy == STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC:
        return f"acyc_skeleton:{acyc_skel}"
    if strategy == STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLETON:
        return f"acyc_mol:{smi}"
    if strategy == STRATEGY_MURCKO_CYCLIC_ACYCLIC_FORMULA:
        return f"acyc_formula:{acyc_form}"
    if strategy == STRATEGY_MURCKO_CYCLIC_ACYCLIC_SINGLE_BUCKET:
        return "acyc_all:ALL_ACYCLIC"

    raise ScaffoldGeneralizationError(f"Unhandled grouping strategy '{strategy}'.")


def generate_scaffold_group_assignments(
    smiles_or_identities: Sequence[Union[str, MolecularStructureIdentity]],
    strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    use_parent_fragment: bool = False,
    merge_stereoisomer_groups: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[MolecularStructureIdentity, ...]]:
    """Generate deterministic group keys and structural identities for a sequence of molecules."""
    if len(smiles_or_identities) == 0:
        raise ScaffoldGeneralizationError("Cannot generate scaffold group assignments for an empty sequence.")

    identities: List[MolecularStructureIdentity] = []
    group_keys: List[str] = []

    for item in smiles_or_identities:
        ident = (
            item
            if isinstance(item, MolecularStructureIdentity)
            else extract_structure_identity(str(item))
        )
        identities.append(ident)
        group_keys.append(
            assign_scaffold_group_key(
                ident,
                strategy=strategy,
                use_parent_fragment=use_parent_fragment,
                merge_stereoisomer_groups=merge_stereoisomer_groups,
            )
        )

    return tuple(group_keys), tuple(identities)


@dataclass(frozen=True)
class GroupDistributionSummary:
    """Quantitative summary of scaffold group sizes and k-fold feasibility."""

    strategy: str
    use_parent_fragment: bool
    merge_stereoisomer_groups: bool
    n_samples: int
    n_cyclic_samples: int
    n_acyclic_samples: int
    n_groups: int
    n_cyclic_groups: int
    n_acyclic_groups: int
    n_singleton_groups: int
    singleton_group_fraction: float
    singleton_sample_fraction: float
    max_group_key: str
    max_group_size: int
    max_group_fraction: float
    max_acyclic_group_key: str
    max_acyclic_group_size: int
    max_acyclic_group_fraction: float
    group_size_quantiles: Dict[str, float]
    requested_n_splits: int
    ideal_fold_size: float
    oversized_threshold_fraction: float
    oversized_groups: Dict[str, int]
    has_oversized_groups: bool
    n_positive_samples: Optional[int]
    n_negative_samples: Optional[int]
    n_positive_groups: Optional[int]
    n_negative_groups: Optional[int]
    is_k_fold_feasible: bool
    infeasibility_reasons: Tuple[str, ...]


def summarize_group_distribution(
    smiles_or_identities: Sequence[Union[str, MolecularStructureIdentity]],
    strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    labels: Optional[Sequence[int]] = None,
    n_splits: int = DEFAULT_CV_SPLITS,
    use_parent_fragment: bool = False,
    merge_stereoisomer_groups: bool = False,
    oversized_threshold_fraction: Optional[float] = None,
    strict_no_oversized_groups: bool = False,
) -> GroupDistributionSummary:
    """Quantify group-size distributions, oversized groups, and k-fold split feasibility.

    An oversized group is defined as any group whose sample count exceeds
    ``oversized_threshold_fraction * n_samples`` (default ``1.0 / n_splits``, i.e. 20% for
    5-fold CV). When a group exceeds ``1 / n_splits`` of the dataset, keeping the group
    intact forces the fold receiving that group to exceed the uniform test-set size.
    """
    if n_splits < 2:
        raise ScaffoldGeneralizationError(f"n_splits must be >= 2, got {n_splits}.")

    group_keys, identities = generate_scaffold_group_assignments(
        smiles_or_identities,
        strategy=strategy,
        use_parent_fragment=use_parent_fragment,
        merge_stereoisomer_groups=merge_stereoisomer_groups,
    )
    n_samples = len(group_keys)

    if labels is not None and len(labels) != n_samples:
        raise ScaffoldGeneralizationError(
            f"Length of labels ({len(labels)}) must match n_samples ({n_samples})."
        )

    threshold_frac = (
        float(oversized_threshold_fraction)
        if oversized_threshold_fraction is not None
        else (1.0 / float(n_splits))
    )
    max_allowed_size = threshold_frac * float(n_samples)

    counts = Counter(group_keys)
    cyclic_group_counts: Counter[str] = Counter()
    acyclic_group_counts: Counter[str] = Counter()
    n_cyclic_samples = 0
    n_acyclic_samples = 0

    for gk, ident in zip(group_keys, identities):
        is_cyc = ident.parent_is_cyclic if use_parent_fragment else ident.is_cyclic
        if is_cyc:
            n_cyclic_samples += 1
            cyclic_group_counts[gk] += 1
        else:
            n_acyclic_samples += 1
            acyclic_group_counts[gk] += 1

    # Deterministic sorting of group counts: size descending, key ascending
    sorted_groups = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    max_group_key, max_group_size = sorted_groups[0]
    max_group_fraction = float(max_group_size) / float(n_samples)

    if acyclic_group_counts:
        sorted_acyc = sorted(acyclic_group_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        max_acyc_key, max_acyc_size = sorted_acyc[0]
        max_acyc_frac = float(max_acyc_size) / float(n_samples)
    else:
        max_acyc_key, max_acyc_size, max_acyc_frac = "", 0, 0.0

    n_groups = len(counts)
    n_singleton_groups = sum(1 for sz in counts.values() if sz == 1)
    singleton_group_fraction = float(n_singleton_groups) / float(n_groups)
    singleton_sample_fraction = float(n_singleton_groups) / float(n_samples)

    sizes_arr = np.array(list(counts.values()), dtype=float)
    quantiles = {
        "min": float(np.min(sizes_arr)),
        "p50": float(np.percentile(sizes_arr, 50)),
        "p90": float(np.percentile(sizes_arr, 90)),
        "p95": float(np.percentile(sizes_arr, 95)),
        "p99": float(np.percentile(sizes_arr, 99)),
        "max": float(np.max(sizes_arr)),
    }

    oversized_groups = {
        gk: int(sz) for gk, sz in sorted_groups if float(sz) > max_allowed_size
    }
    has_oversized = len(oversized_groups) > 0

    infeasibility_reasons: List[str] = []
    if n_samples < n_splits:
        infeasibility_reasons.append(
            f"Cohort sample count ({n_samples}) is smaller than requested n_splits ({n_splits})."
        )
    if n_groups < n_splits:
        infeasibility_reasons.append(
            f"Distinct scaffold group count ({n_groups}) is smaller than requested n_splits ({n_splits})."
        )

    n_pos_samples: Optional[int] = None
    n_neg_samples: Optional[int] = None
    n_pos_groups: Optional[int] = None
    n_neg_groups: Optional[int] = None

    if labels is not None:
        y_arr = np.asarray(labels, dtype=int)
        unique_vals = set(np.unique(y_arr).tolist())
        if not unique_vals.issubset({0, 1}):
            raise ScaffoldGeneralizationError(
                f"Binary labels must be in {{0, 1}}, found values: {sorted(unique_vals)}."
            )
        n_pos_samples = int(np.sum(y_arr == 1))
        n_neg_samples = int(np.sum(y_arr == 0))
        pos_group_set = {gk for gk, y_val in zip(group_keys, y_arr) if int(y_val) == 1}
        neg_group_set = {gk for gk, y_val in zip(group_keys, y_arr) if int(y_val) == 0}
        n_pos_groups = len(pos_group_set)
        n_neg_groups = len(neg_group_set)

        if n_pos_samples < n_splits:
            infeasibility_reasons.append(
                f"Positive sample count ({n_pos_samples}) is smaller than requested n_splits ({n_splits})."
            )
        if n_neg_samples < n_splits:
            infeasibility_reasons.append(
                f"Negative sample count ({n_neg_samples}) is smaller than requested n_splits ({n_splits})."
            )
        if n_pos_groups < n_splits:
            infeasibility_reasons.append(
                f"Positive scaffold group count ({n_pos_groups}) is smaller than requested "
                f"n_splits ({n_splits}); cannot place >= 1 positive group in every fold without "
                f"splitting a group."
            )
        if n_neg_groups < n_splits:
            infeasibility_reasons.append(
                f"Negative scaffold group count ({n_neg_groups}) is smaller than requested "
                f"n_splits ({n_splits}); cannot place >= 1 negative group in every fold without "
                f"splitting a group."
            )

    if strict_no_oversized_groups and has_oversized:
        infeasibility_reasons.append(
            f"Strategy '{strategy}' produced {len(oversized_groups)} oversized group(s) exceeding "
            f"{threshold_frac * 100:.1f}% of N={n_samples} (max group '{max_group_key}' has "
            f"size {max_group_size} = {max_group_fraction * 100:.2f}%)."
        )

    return GroupDistributionSummary(
        strategy=strategy,
        use_parent_fragment=use_parent_fragment,
        merge_stereoisomer_groups=merge_stereoisomer_groups,
        n_samples=n_samples,
        n_cyclic_samples=n_cyclic_samples,
        n_acyclic_samples=n_acyclic_samples,
        n_groups=n_groups,
        n_cyclic_groups=len(cyclic_group_counts),
        n_acyclic_groups=len(acyclic_group_counts),
        n_singleton_groups=n_singleton_groups,
        singleton_group_fraction=round(singleton_group_fraction, 6),
        singleton_sample_fraction=round(singleton_sample_fraction, 6),
        max_group_key=max_group_key,
        max_group_size=int(max_group_size),
        max_group_fraction=round(max_group_fraction, 6),
        max_acyclic_group_key=max_acyc_key,
        max_acyclic_group_size=int(max_acyc_size),
        max_acyclic_group_fraction=round(max_acyc_frac, 6),
        group_size_quantiles=quantiles,
        requested_n_splits=n_splits,
        ideal_fold_size=round(float(n_samples) / float(n_splits), 4),
        oversized_threshold_fraction=round(threshold_frac, 6),
        oversized_groups=oversized_groups,
        has_oversized_groups=has_oversized,
        n_positive_samples=n_pos_samples,
        n_negative_samples=n_neg_samples,
        n_positive_groups=n_pos_groups,
        n_negative_groups=n_neg_groups,
        is_k_fold_feasible=(len(infeasibility_reasons) == 0),
        infeasibility_reasons=tuple(infeasibility_reasons),
    )


# ---------------------------------------------------------------------------
# 3. Grouped Cross-Validation & Programmatic Split-Integrity Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSplitSummary:
    """Detailed composition and class-balance summary for a single cross-validation fold."""

    fold_index: int
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    n_train: int
    n_test: int
    n_train_positives: int
    n_train_negatives: int
    n_test_positives: int
    n_test_negatives: int
    train_prevalence: float
    test_prevalence: float
    prevalence_drift_from_cohort: float
    n_train_groups: int
    n_test_groups: int
    n_test_singleton_groups: int
    n_test_cyclic: int
    n_test_acyclic: int
    max_test_group_size: int


@dataclass(frozen=True)
class SplitIntegrityReport:
    """Comprehensive verification report for a k-fold cross-validation partition."""

    is_valid: bool
    regime: str
    strategy: str
    n_splits: int
    seed: int
    n_samples: int
    cohort_positives: int
    cohort_negatives: int
    cohort_prevalence: float
    zero_index_overlap: bool
    complete_index_conservation: bool
    zero_group_overlap: bool
    zero_canonical_smiles_overlap: bool
    zero_parent_smiles_overlap: bool
    zero_achiral_smiles_overlap: bool
    parent_overlap_pair_count: int
    achiral_overlap_pair_count: int
    all_folds_meet_class_support: bool
    min_fold_test_positives: int
    min_fold_test_negatives: int
    max_abs_prevalence_drift: float
    fold_test_size_min: int
    fold_test_size_max: int
    violations: Tuple[str, ...]
    warnings: Tuple[str, ...]
    folds: Tuple[FoldSplitSummary, ...]


def validate_split_integrity(
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    identities: Sequence[MolecularStructureIdentity],
    labels: Sequence[int],
    group_keys: Sequence[str],
    regime: str = "grouped_kfold",
    strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    seed: int = DEFAULT_RANDOM_SEED,
    enforce_group_disjointness: bool = True,
    enforce_parent_disjointness: bool = False,
    enforce_achiral_disjointness: bool = False,
    min_train_positives: int = DEFAULT_MIN_FOLD_POSITIVES,
    min_train_negatives: int = DEFAULT_MIN_FOLD_NEGATIVES,
    min_test_positives: int = DEFAULT_MIN_FOLD_POSITIVES,
    min_test_negatives: int = DEFAULT_MIN_FOLD_NEGATIVES,
    raise_on_violation: bool = True,
) -> SplitIntegrityReport:
    """Verify index conservation, group/structure disjointness, and class support across folds.

    Does not enforce an artificial universal +/- 2.5 percentage-point prevalence tolerance
    when grouping constraints make it unattainable; instead records exact ``max_abs_prevalence_drift``
    and fold-size imbalance and flags advisory warnings when drift exceeds 0.025 (2.5 percentage
    points) or when an oversized group causes a fold test size to deviate by > 10% from N / K.
    """
    n_samples = len(identities)
    if len(labels) != n_samples or len(group_keys) != n_samples:
        raise ScaffoldGeneralizationError(
            f"Lengths of identities ({n_samples}), labels ({len(labels)}), and "
            f"group_keys ({len(group_keys)}) must match."
        )
    if len(folds) == 0:
        raise ScaffoldGeneralizationError("folds sequence cannot be empty.")

    y_arr = np.asarray(labels, dtype=int)
    cohort_pos = int(np.sum(y_arr == 1))
    cohort_neg = int(np.sum(y_arr == 0))
    cohort_prev = float(cohort_pos) / float(n_samples) if n_samples > 0 else 0.0

    expected_all_indices = set(range(n_samples))
    pooled_test_indices: List[int] = []
    seen_test_groups: set[str] = set()

    zero_index_overlap = True
    complete_index_conservation = True
    zero_group_overlap = True
    zero_canonical_overlap = True
    zero_parent_overlap = True
    zero_achiral_overlap = True
    all_folds_meet_support = True

    total_parent_overlap_pairs = 0
    total_achiral_overlap_pairs = 0

    violations: List[str] = []
    warnings_list: List[str] = []
    fold_summaries: List[FoldSplitSummary] = []

    for fold_idx, (tr_raw, te_raw) in enumerate(folds):
        tr_list = [int(i) for i in tr_raw]
        te_list = [int(i) for i in te_raw]
        tr_set = set(tr_list)
        te_set = set(te_list)

        if len(tr_list) == 0 or len(te_list) == 0:
            zero_index_overlap = False
            complete_index_conservation = False
            violations.append(f"Fold {fold_idx}: empty train ({len(tr_list)}) or test ({len(te_list)}) partition.")

        if len(tr_list) != len(tr_set) or len(te_list) != len(te_set):
            zero_index_overlap = False
            violations.append(f"Fold {fold_idx}: duplicate sample indices within train or test partition.")

        overlap_idx = tr_set & te_set
        if overlap_idx:
            zero_index_overlap = False
            violations.append(
                f"Fold {fold_idx}: {len(overlap_idx)} sample index(es) overlap between train and test."
            )

        union_idx = tr_set | te_set
        if union_idx != expected_all_indices:
            complete_index_conservation = False
            missing = len(expected_all_indices - union_idx)
            extra = len(union_idx - expected_all_indices)
            violations.append(
                f"Fold {fold_idx}: train U test does not equal 0..{n_samples - 1} "
                f"(missing={missing}, out_of_bounds={extra})."
            )

        pooled_test_indices.extend(te_list)

        # Group overlap checks
        tr_groups = {group_keys[i] for i in tr_set if 0 <= i < n_samples}
        te_groups = {group_keys[i] for i in te_set if 0 <= i < n_samples}
        grp_overlap = tr_groups & te_groups
        if grp_overlap:
            zero_group_overlap = False
            if enforce_group_disjointness:
                violations.append(
                    f"Fold {fold_idx}: {len(grp_overlap)} scaffold group(s) leaked across train and test "
                    f"(e.g. {sorted(grp_overlap)[:3]})."
                )

        cross_fold_grp_overlap = seen_test_groups & te_groups
        if cross_fold_grp_overlap:
            zero_group_overlap = False
            if enforce_group_disjointness:
                violations.append(
                    f"Fold {fold_idx}: {len(cross_fold_grp_overlap)} group(s) appeared in multiple test folds."
                )
        seen_test_groups.update(te_groups)

        # Canonical SMILES overlap check
        tr_canon = {identities[i].canonical_smiles for i in tr_set if 0 <= i < n_samples}
        te_canon = {identities[i].canonical_smiles for i in te_set if 0 <= i < n_samples}
        canon_overlap = tr_canon & te_canon
        if canon_overlap:
            zero_canonical_overlap = False
            violations.append(
                f"Fold {fold_idx}: {len(canon_overlap)} canonical SMILES leaked across train and test "
                f"(e.g. {sorted(canon_overlap)[:3]})."
            )

        # Parent-fragment overlap audit
        tr_parent = {identities[i].parent_canonical_smiles for i in tr_set if 0 <= i < n_samples}
        te_parent = {identities[i].parent_canonical_smiles for i in te_set if 0 <= i < n_samples}
        parent_overlap = tr_parent & te_parent
        if parent_overlap:
            zero_parent_overlap = False
            total_parent_overlap_pairs += len(parent_overlap)
            if enforce_parent_disjointness:
                violations.append(
                    f"Fold {fold_idx}: {len(parent_overlap)} parent fragment structure(s) leaked across "
                    f"train and test (e.g. {sorted(parent_overlap)[:3]})."
                )
            else:
                warnings_list.append(
                    f"Fold {fold_idx}: {len(parent_overlap)} shared parent fragment(s) cross train/test "
                    f"boundary (salt/counterion pairs with distinct raw SMILES)."
                )

        # Achiral stereoisomer overlap audit
        tr_achiral = {identities[i].achiral_canonical_smiles for i in tr_set if 0 <= i < n_samples}
        te_achiral = {identities[i].achiral_canonical_smiles for i in te_set if 0 <= i < n_samples}
        achiral_overlap = tr_achiral & te_achiral
        if achiral_overlap:
            zero_achiral_overlap = False
            total_achiral_overlap_pairs += len(achiral_overlap)
            if enforce_achiral_disjointness:
                violations.append(
                    f"Fold {fold_idx}: {len(achiral_overlap)} achiral stereoisomer structure(s) leaked "
                    f"across train and test (e.g. {sorted(achiral_overlap)[:3]})."
                )
            else:
                warnings_list.append(
                    f"Fold {fold_idx}: {len(achiral_overlap)} achiral stereoisomer graph(s) cross "
                    f"train/test boundary."
                )

        # Class support & prevalence
        valid_tr = [i for i in tr_list if 0 <= i < n_samples]
        valid_te = [i for i in te_list if 0 <= i < n_samples]
        y_tr = y_arr[valid_tr]
        y_te = y_arr[valid_te]

        n_tr_pos = int(np.sum(y_tr == 1))
        n_tr_neg = int(np.sum(y_tr == 0))
        n_te_pos = int(np.sum(y_te == 1))
        n_te_neg = int(np.sum(y_te == 0))

        if (
            n_tr_pos < min_train_positives
            or n_tr_neg < min_train_negatives
            or n_te_pos < min_test_positives
            or n_te_neg < min_test_negatives
        ):
            all_folds_meet_support = False
            violations.append(
                f"Fold {fold_idx}: insufficient class support "
                f"(train_pos={n_tr_pos}, train_neg={n_tr_neg}, test_pos={n_te_pos}, test_neg={n_te_neg}; "
                f"required min_train=({min_train_positives},{min_train_negatives}), "
                f"min_test=({min_test_positives},{min_test_negatives}))."
            )

        tr_prev = float(n_tr_pos) / float(len(valid_tr)) if len(valid_tr) > 0 else 0.0
        te_prev = float(n_te_pos) / float(len(valid_te)) if len(valid_te) > 0 else 0.0
        prev_drift = te_prev - cohort_prev

        if abs(prev_drift) > 0.025:
            warnings_list.append(
                f"Fold {fold_idx}: test prevalence ({te_prev * 100:.2f}%) differs from cohort "
                f"prevalence ({cohort_prev * 100:.2f}%) by {prev_drift * 100:+.2f} percentage points "
                f"due to discrete scaffold-group size constraints."
            )

        te_group_counts = Counter(group_keys[i] for i in valid_te)
        n_te_singletons = sum(1 for sz in te_group_counts.values() if sz == 1)
        max_te_grp_size = max(te_group_counts.values()) if te_group_counts else 0
        n_te_cyc = sum(1 for i in valid_te if identities[i].is_cyclic)
        n_te_acyc = len(valid_te) - n_te_cyc

        fold_summaries.append(
            FoldSplitSummary(
                fold_index=fold_idx,
                train_indices=tuple(tr_list),
                test_indices=tuple(te_list),
                n_train=len(tr_list),
                n_test=len(te_list),
                n_train_positives=n_tr_pos,
                n_train_negatives=n_tr_neg,
                n_test_positives=n_te_pos,
                n_test_negatives=n_te_neg,
                train_prevalence=round(tr_prev, 6),
                test_prevalence=round(te_prev, 6),
                prevalence_drift_from_cohort=round(prev_drift, 6),
                n_train_groups=len(tr_groups),
                n_test_groups=len(te_groups),
                n_test_singleton_groups=n_te_singletons,
                n_test_cyclic=n_te_cyc,
                n_test_acyclic=n_te_acyc,
                max_test_group_size=int(max_te_grp_size),
            )
        )

    # Cross-fold conservation of test indices
    if len(pooled_test_indices) != n_samples or set(pooled_test_indices) != expected_all_indices:
        complete_index_conservation = False
        if len(pooled_test_indices) != len(set(pooled_test_indices)):
            zero_index_overlap = False
            violations.append(
                f"Cross-fold test overlap: {len(pooled_test_indices) - len(set(pooled_test_indices))} "
                f"sample index(es) appear in more than one test fold."
            )
        else:
            violations.append(
                f"Cross-fold index conservation failure: pooled test indices ({len(pooled_test_indices)}) "
                f"do not cover all {n_samples} samples."
            )

    min_te_pos = min((f.n_test_positives for f in fold_summaries), default=0)
    min_te_neg = min((f.n_test_negatives for f in fold_summaries), default=0)
    max_abs_drift = max((abs(f.prevalence_drift_from_cohort) for f in fold_summaries), default=0.0)
    te_size_min = min((f.n_test for f in fold_summaries), default=0)
    te_size_max = max((f.n_test for f in fold_summaries), default=0)

    is_valid = len(violations) == 0
    if raise_on_violation and not is_valid:
        raise SplitIntegrityError(
            "Split integrity validation failed:\n  - " + "\n  - ".join(violations)
        )

    return SplitIntegrityReport(
        is_valid=is_valid,
        regime=regime,
        strategy=strategy,
        n_splits=len(folds),
        seed=seed,
        n_samples=n_samples,
        cohort_positives=cohort_pos,
        cohort_negatives=cohort_neg,
        cohort_prevalence=round(cohort_prev, 6),
        zero_index_overlap=zero_index_overlap,
        complete_index_conservation=complete_index_conservation,
        zero_group_overlap=zero_group_overlap,
        zero_canonical_smiles_overlap=zero_canonical_overlap,
        zero_parent_smiles_overlap=zero_parent_overlap,
        zero_achiral_smiles_overlap=zero_achiral_overlap,
        parent_overlap_pair_count=total_parent_overlap_pairs,
        achiral_overlap_pair_count=total_achiral_overlap_pairs,
        all_folds_meet_class_support=all_folds_meet_support,
        min_fold_test_positives=min_te_pos,
        min_fold_test_negatives=min_te_neg,
        max_abs_prevalence_drift=round(max_abs_drift, 6),
        fold_test_size_min=te_size_min,
        fold_test_size_max=te_size_max,
        violations=tuple(violations),
        warnings=tuple(warnings_list),
        folds=tuple(fold_summaries),
    )


def generate_grouped_kfold_splits(
    smiles_or_identities: Sequence[Union[str, MolecularStructureIdentity]],
    labels: Sequence[int],
    strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    n_splits: int = DEFAULT_CV_SPLITS,
    seed: int = DEFAULT_RANDOM_SEED,
    use_parent_fragment: bool = False,
    merge_stereoisomer_groups: bool = False,
    strict_no_oversized_groups: bool = False,
    min_test_positives: int = DEFAULT_MIN_FOLD_POSITIVES,
    min_test_negatives: int = DEFAULT_MIN_FOLD_NEGATIVES,
) -> Tuple[SplitIntegrityReport, GroupDistributionSummary]:
    """Generate and validate a deterministic StratifiedGroupKFold split for a molecular cohort.

    Fails gracefully with ``InfeasibleFoldConfigurationError`` if the cohort or grouping
    strategy cannot support ``n_splits`` folds (e.g., fewer positive scaffold groups than
    folds), never weakening grouping constraints to force a split.
    """
    dist_summary = summarize_group_distribution(
        smiles_or_identities=smiles_or_identities,
        strategy=strategy,
        labels=labels,
        n_splits=n_splits,
        use_parent_fragment=use_parent_fragment,
        merge_stereoisomer_groups=merge_stereoisomer_groups,
        strict_no_oversized_groups=strict_no_oversized_groups,
    )
    if not dist_summary.is_k_fold_feasible:
        raise InfeasibleFoldConfigurationError(
            f"Cannot construct {n_splits}-fold grouped split under strategy '{strategy}': "
            + "; ".join(dist_summary.infeasibility_reasons)
        )

    group_keys, identities = generate_scaffold_group_assignments(
        smiles_or_identities,
        strategy=strategy,
        use_parent_fragment=use_parent_fragment,
        merge_stereoisomer_groups=merge_stereoisomer_groups,
    )
    y_arr = np.asarray(labels, dtype=int)
    dummy_X = np.zeros((len(y_arr), 1), dtype=np.int8)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raw_folds = [
        (tuple(int(i) for i in tr_idx), tuple(int(i) for i in te_idx))
        for tr_idx, te_idx in sgkf.split(dummy_X, y_arr, groups=group_keys)
    ]

    try:
        report = validate_split_integrity(
            folds=raw_folds,
            identities=identities,
            labels=y_arr,
            group_keys=group_keys,
            regime="grouped_kfold",
            strategy=strategy,
            seed=seed,
            enforce_group_disjointness=True,
            enforce_parent_disjointness=use_parent_fragment,
            enforce_achiral_disjointness=merge_stereoisomer_groups,
            min_test_positives=min_test_positives,
            min_test_negatives=min_test_negatives,
            raise_on_violation=True,
        )
    except SplitIntegrityError as exc:
        raise InfeasibleFoldConfigurationError(
            f"StratifiedGroupKFold(n_splits={n_splits}, seed={seed}) produced a fold "
            f"violating support or integrity constraints: {exc}"
        ) from exc

    return report, dist_summary


def generate_matched_random_kfold_splits(
    smiles_or_identities: Sequence[Union[str, MolecularStructureIdentity]],
    labels: Sequence[int],
    reference_strategy: str = STRATEGY_MURCKO_CYCLIC_ACYCLIC_GENERIC,
    n_splits: int = DEFAULT_CV_SPLITS,
    seed: int = DEFAULT_RANDOM_SEED,
    min_test_positives: int = DEFAULT_MIN_FOLD_POSITIVES,
    min_test_negatives: int = DEFAULT_MIN_FOLD_NEGATIVES,
) -> SplitIntegrityReport:
    """Generate a matched StratifiedKFold random split on the exact same cohort.

    Checks index conservation and exact canonical SMILES disjointness (assuming deduplicated
    cohort) while allowing scaffold groups to cross train/test boundaries so random vs.
    scaffold regimes can be compared on identical molecular cohorts.
    """
    if n_splits < 2:
        raise ScaffoldGeneralizationError(f"n_splits must be >= 2, got {n_splits}.")

    group_keys, identities = generate_scaffold_group_assignments(
        smiles_or_identities,
        strategy=reference_strategy,
    )
    y_arr = np.asarray(labels, dtype=int)
    n_samples = len(y_arr)
    if n_samples < n_splits:
        raise InfeasibleFoldConfigurationError(
            f"Cohort sample count ({n_samples}) is smaller than n_splits ({n_splits})."
        )

    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    if n_pos < n_splits or n_neg < n_splits:
        raise InfeasibleFoldConfigurationError(
            f"Insufficient class counts (pos={n_pos}, neg={n_neg}) for {n_splits}-fold StratifiedKFold."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_X = np.zeros((n_samples, 1), dtype=np.int8)
    raw_folds = [
        (tuple(int(i) for i in tr_idx), tuple(int(i) for i in te_idx))
        for tr_idx, te_idx in skf.split(dummy_X, y_arr)
    ]

    return validate_split_integrity(
        folds=raw_folds,
        identities=identities,
        labels=y_arr,
        group_keys=group_keys,
        regime="random_stratified_kfold",
        strategy=reference_strategy,
        seed=seed,
        enforce_group_disjointness=False,
        enforce_parent_disjointness=False,
        enforce_achiral_disjointness=False,
        min_test_positives=min_test_positives,
        min_test_negatives=min_test_negatives,
        raise_on_violation=True,
    )


def compute_split_seed_variation_summary(
    reports: Sequence[SplitIntegrityReport],
) -> Dict[str, Any]:
    """Quantify fold-assignment sensitivity across multiple random seeds (e.g. 42, 1042, 2042).

    For each pair of seeds (s1, s2), computes the maximum pairwise Jaccard similarity
    between test folds of s1 and test folds of s2, verifying that changing the seed
    genuinely alters fold assignments while preserving deterministic reproducibility for
    any fixed seed.
    """
    if len(reports) < 2:
        raise ScaffoldGeneralizationError("At least 2 SplitIntegrityReport objects are required.")

    n_samples = reports[0].n_samples
    n_splits = reports[0].n_splits
    for rep in reports[1:]:
        if rep.n_samples != n_samples or rep.n_splits != n_splits:
            raise ScaffoldGeneralizationError("All reports must share the same n_samples and n_splits.")

    pairwise_Comparisons: List[Dict[str, Any]] = []
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            r1 = reports[i]
            r2 = reports[j]
            best_jaccards: List[float] = []
            exact_matches = 0
            for f1 in r1.folds:
                s1 = set(f1.test_indices)
                j_vals = []
                for f2 in r2.folds:
                    s2 = set(f2.test_indices)
                    if s1 == s2:
                        exact_matches += 1
                    union_len = len(s1 | s2)
                    j_vals.append(float(len(s1 & s2)) / float(union_len) if union_len > 0 else 1.0)
                best_jaccards.append(round(max(j_vals), 4))

            pairwise_Comparisons.append(
                {
                    "seed_a": r1.seed,
                    "seed_b": r2.seed,
                    "max_jaccard_per_fold": best_jaccards,
                    "mean_max_jaccard": round(float(np.mean(best_jaccards)), 4),
                    "identical_fold_count": exact_matches,
                    "assignments_differ": exact_matches < n_splits,
                }
            )

    return {
        "seeds": [r.seed for r in reports],
        "n_splits": n_splits,
        "n_samples": n_samples,
        "pairwise_seed_comparisons": pairwise_Comparisons,
        "all_seed_pairs_produce_distinct_partitions": all(
            c["assignments_differ"] for c in pairwise_Comparisons
        ),
    }


# ---------------------------------------------------------------------------
# 4. Endpoint-Specific Cohort Builders (ClinTox & 12 Tox21 Assays)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationCohort:
    """Isolated, validated molecular evaluation cohort for a single dataset/endpoint."""

    dataset_name: str
    endpoint: str
    n_raw_rows: int
    n_invalid_smiles_excluded: int
    n_missing_label_excluded: int
    n_duplicate_rows_resolved: int
    n_samples: int
    n_positives: int
    n_negatives: int
    prevalence: float
    canonical_smiles: Tuple[str, ...]
    labels: np.ndarray
    fingerprints: Optional[np.ndarray]
    cyclic_mask: np.ndarray
    identities: Tuple[MolecularStructureIdentity, ...]


def build_clintox_evaluation_cohort(
    raw_csv_path: Optional[Union[str, Path]] = None,
    duplicate_policy: str = "conservative_toxic",
    compute_fingerprints: bool = False,
) -> EvaluationCohort:
    """Construct the validated ClinTox (CT_TOX) evaluation cohort without writing any files."""
    csv_path = resolve_path(raw_csv_path or CLINTOX_DATA_PATH)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ClinTox raw dataset not found at '{csv_path}'.")

    raw_df = pd.read_csv(csv_path)
    n_raw = len(raw_df)

    X_arr, y_arr, clean_df = prepare_clintox_dataset(
        csv_path=csv_path,
        duplicate_policy=duplicate_policy,
    )
    smiles_list = [str(s) for s in clean_df["canonical_smiles"].tolist()]
    identities = tuple(extract_structure_identity(s) for s in smiles_list)
    cyclic_mask = np.array([ident.is_cyclic for ident in identities], dtype=bool)

    # Count invalid vs duplicate exclusions deterministically
    n_invalid = 0
    n_missing = 0
    valid_non_missing = 0
    for _, row in raw_df.iterrows():
        vr = validate_smiles(str(row.get("smiles", "")))
        if not vr.is_valid or not vr.canonical_smiles:
            n_invalid += 1
        elif pd.isna(row.get("CT_TOX")):
            n_missing += 1
        else:
            valid_non_missing += 1

    n_samples = len(smiles_list)
    n_dup_resolved = max(0, valid_non_missing - n_samples)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    prev = float(n_pos) / float(n_samples) if n_samples > 0 else 0.0

    return EvaluationCohort(
        dataset_name="ClinTox",
        endpoint="CT_TOX",
        n_raw_rows=n_raw,
        n_invalid_smiles_excluded=n_invalid,
        n_missing_label_excluded=n_missing,
        n_duplicate_rows_resolved=n_dup_resolved,
        n_samples=n_samples,
        n_positives=n_pos,
        n_negatives=n_neg,
        prevalence=round(prev, 6),
        canonical_smiles=tuple(smiles_list),
        labels=y_arr.astype(int),
        fingerprints=X_arr if compute_fingerprints else None,
        cyclic_mask=cyclic_mask,
        identities=identities,
    )


def build_tox21_endpoint_cohort(
    endpoint: str,
    df_or_csv_path: Optional[Union[pd.DataFrame, str, Path]] = None,
    duplicate_policy: str = "conservative_active",
    compute_fingerprints: bool = False,
    radius: int = 2,
    n_bits: int = 2048,
) -> EvaluationCohort:
    """Construct an endpoint-specific Tox21 evaluation cohort with strict NaN label isolation.

    Missing assay labels (``NaN``) for ``endpoint`` are strictly excluded from that endpoint's
    cohort and are never imputed as inactive (0), nor do missing labels in other Tox21 columns
    exclude valid rows for ``endpoint``.

    Parameters
    ----------
    endpoint : str
        Target Tox21 assay column name (e.g. ``"NR-AR"``, ``"SR-MMP"``).
    df_or_csv_path : Optional[Union[pd.DataFrame, str, Path]]
        In-memory DataFrame (for fast unit testing) or path to ``tox21.csv``.
    duplicate_policy : str, default="conservative_active"
        Policy for handling duplicate canonical SMILES within the endpoint cohort:
        - ``"conservative_active"``: Resolves conflicting duplicate labels to 1 (active).
        - ``"drop_conflicts"``: Drops canonical SMILES with conflicting binary labels.
        - ``"keep_all"``: Preserves all rows without deduplicating canonical SMILES.
    compute_fingerprints : bool, default=False
        Whether to compute 2048-bit Morgan fingerprints for the cohort.
    """
    if duplicate_policy not in ("conservative_active", "drop_conflicts", "keep_all"):
        raise ScaffoldGeneralizationError(
            f"Unsupported duplicate_policy '{duplicate_policy}'. Expected "
            f"'conservative_active', 'drop_conflicts', or 'keep_all'."
        )

    if isinstance(df_or_csv_path, pd.DataFrame):
        df = df_or_csv_path.copy()
    else:
        csv_path = resolve_path(df_or_csv_path or TOX21_DATA_PATH)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tox21 raw dataset not found at '{csv_path}'.")
        df = pd.read_csv(csv_path)

    if "smiles" not in df.columns:
        raise ScaffoldGeneralizationError("Input Tox21 table is missing required 'smiles' column.")
    if endpoint not in df.columns:
        raise ScaffoldGeneralizationError(f"Endpoint '{endpoint}' not found in dataset columns.")

    n_raw = len(df)
    n_invalid = 0
    n_missing = 0

    valid_records: List[Tuple[str, int]] = []
    for _, row in df.iterrows():
        raw_smi = row["smiles"]
        if pd.isna(raw_smi):
            n_invalid += 1
            continue
        vr = validate_smiles(str(raw_smi))
        if not vr.is_valid or not vr.canonical_smiles:
            n_invalid += 1
            continue

        raw_label = row[endpoint]
        if pd.isna(raw_label):
            n_missing += 1
            continue

        label_int = int(float(raw_label))
        if label_int not in (0, 1):
            raise ScaffoldGeneralizationError(
                f"Invalid non-binary label '{raw_label}' for endpoint '{endpoint}'."
            )
        valid_records.append((vr.canonical_smiles, label_int))

    if len(valid_records) == 0:
        raise ScaffoldGeneralizationError(
            f"Endpoint '{endpoint}' has 0 valid non-null records (raw={n_raw}, "
            f"invalid_smiles={n_invalid}, missing_label={n_missing})."
        )

    if duplicate_policy == "keep_all":
        final_smiles = [s for s, _ in valid_records]
        final_labels = [y for _, y in valid_records]
        n_dup_resolved = 0
    else:
        grouped_labels: Dict[str, List[int]] = defaultdict(list)
        ordered_unique_smiles: List[str] = []
        for smi, lbl in valid_records:
            if smi not in grouped_labels:
                ordered_unique_smiles.append(smi)
            grouped_labels[smi].append(lbl)

        final_smiles = []
        final_labels = []
        for smi in ordered_unique_smiles:
            lbls = grouped_labels[smi]
            unique_lbls = set(lbls)
            if len(unique_lbls) > 1 and duplicate_policy == "drop_conflicts":
                continue
            resolved_lbl = 1 if (1 in unique_lbls) else 0
            final_smiles.append(smi)
            final_labels.append(resolved_lbl)
        n_dup_resolved = len(valid_records) - len(final_smiles)

    if len(final_smiles) == 0:
        raise ScaffoldGeneralizationError(
            f"Endpoint '{endpoint}' has 0 remaining records after duplicate resolution."
        )

    identities = tuple(extract_structure_identity(s) for s in final_smiles)
    cyclic_mask = np.array([ident.is_cyclic for ident in identities], dtype=bool)
    y_arr = np.asarray(final_labels, dtype=int)

    fps_arr: Optional[np.ndarray] = None
    if compute_fingerprints:
        fps_list = [
            generate_morgan_fingerprint(s, radius=radius, n_bits=n_bits, return_bit_info=False).numpy_vector
            for s in final_smiles
        ]
        fps_arr = np.asarray(fps_list, dtype=np.int8)

    n_samples = len(final_smiles)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    prev = float(n_pos) / float(n_samples)

    return EvaluationCohort(
        dataset_name="Tox21",
        endpoint=endpoint,
        n_raw_rows=n_raw,
        n_invalid_smiles_excluded=n_invalid,
        n_missing_label_excluded=n_missing,
        n_duplicate_rows_resolved=n_dup_resolved,
        n_samples=n_samples,
        n_positives=n_pos,
        n_negatives=n_neg,
        prevalence=round(prev, 6),
        canonical_smiles=tuple(final_smiles),
        labels=y_arr,
        fingerprints=fps_arr,
        cyclic_mask=cyclic_mask,
        identities=identities,
    )


# ---------------------------------------------------------------------------
# 5. Metrics, Prevalence-Adjusted PR-AUC, and Threshold Semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdMetrics:
    """Classification metrics and confusion matrix evaluated at a documented probability threshold."""

    threshold: float
    threshold_type: str
    tn: int
    fp: int
    fn: int
    tp: int
    confusion_matrix: Tuple[Tuple[int, int], Tuple[int, int]]
    precision: float
    recall: float
    specificity: float
    f1: float
    balanced_accuracy: float
    accuracy: float


@dataclass(frozen=True)
class BinaryEvaluationMetrics:
    """Complete discrimination, probability-quality, and threshold metrics for a sample cohort."""

    subset_name: str
    n_samples: int
    n_positives: int
    n_negatives: int
    prevalence: float
    insufficient_support: bool
    insufficient_support_reasons: Tuple[str, ...]
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    pr_auc_lift: Optional[float]
    pr_auc_ratio: Optional[float]
    brier_score: Optional[float]
    brier_skill_score: Optional[float]
    log_loss: Optional[float]
    fixed_threshold_metrics: Optional[ThresholdMetrics]
    exploratory_prevalence_threshold_metrics: Optional[ThresholdMetrics]


def _compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    threshold_type: str,
) -> ThresholdMetrics:
    """Compute binary classification metrics at a specific probability threshold."""
    if not (0.0 <= float(threshold) <= 1.0):
        raise ScaffoldGeneralizationError(f"Classification threshold must be in [0.0, 1.0], got {threshold}.")

    y_pred = (y_prob >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0.0))
    spec = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    f1_val = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0.0))
    bal_acc = 0.5 * (rec + spec) if ((tp + fn) > 0 and (tn + fp) > 0) else float(accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))

    return ThresholdMetrics(
        threshold=round(float(threshold), 6),
        threshold_type=threshold_type,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        confusion_matrix=((tn, fp), (fn, tp)),
        precision=round(prec, 6),
        recall=round(rec, 6),
        specificity=round(spec, 6),
        f1=round(f1_val, 6),
        balanced_accuracy=round(bal_acc, 6),
        accuracy=round(acc, 6),
    )


def compute_binary_evaluation_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    subset_name: str = "full_cohort",
    fixed_threshold: float = DEFAULT_FIXED_THRESHOLD,
    exploratory_train_prevalence_threshold: Optional[float] = None,
    min_samples: int = DEFAULT_MIN_METRIC_SAMPLES,
    min_positives: int = DEFAULT_MIN_METRIC_POSITIVES,
    min_negatives: int = DEFAULT_MIN_METRIC_POSITIVES,
) -> BinaryEvaluationMetrics:
    """Compute discrimination, calibration-loss, and threshold metrics for binary predictions.

    Definitions of prevalence-adjusted PR-AUC metrics:
    - ``prevalence`` (pi): ``n_positives / n_samples``
    - ``pr_auc``: Average precision score (area under precision-recall step curve)
    - ``pr_auc_lift``: ``pr_auc - prevalence`` (additive gain over a random-ranking baseline
      whose expected precision equals ``prevalence``)
    - ``pr_auc_ratio``: ``pr_auc / prevalence`` (multiplicative enrichment factor over a
      random-ranking baseline)

    Threshold governance:
    - ``fixed_threshold`` defaults to ``0.50`` (primary decision threshold).
    - ``exploratory_train_prevalence_threshold``, if provided, must be the positive prevalence
      of the *training* partition (never tuned on test labels) and is reported strictly as a
      prespecified exploratory sensitivity metric.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob, dtype=float)

    if y_t.ndim != 1 or y_p.ndim != 1:
        raise ScaffoldGeneralizationError("y_true and y_prob must be 1D sequences.")
    if len(y_t) != len(y_p):
        raise ScaffoldGeneralizationError(
            f"Length mismatch between y_true ({len(y_t)}) and y_prob ({len(y_p)})."
        )
    if len(y_t) == 0:
        return BinaryEvaluationMetrics(
            subset_name=subset_name,
            n_samples=0,
            n_positives=0,
            n_negatives=0,
            prevalence=0.0,
            insufficient_support=True,
            insufficient_support_reasons=("Empty evaluation subset (n_samples=0).",),
            roc_auc=None,
            pr_auc=None,
            pr_auc_lift=None,
            pr_auc_ratio=None,
            brier_score=None,
            brier_skill_score=None,
            log_loss=None,
            fixed_threshold_metrics=None,
            exploratory_prevalence_threshold_metrics=None,
        )

    if not np.all(np.isfinite(y_p)):
        raise ScaffoldGeneralizationError("y_prob contains NaN or infinite values.")
    if np.any((y_p < 0.0) | (y_p > 1.0)):
        raise ScaffoldGeneralizationError(
            f"y_prob values must lie within [0.0, 1.0]; found min={np.min(y_p)}, max={np.max(y_p)}."
        )

    unique_labels = set(np.unique(y_t).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ScaffoldGeneralizationError(
            f"y_true must contain binary labels in {{0, 1}}, found: {sorted(unique_labels)}."
        )
    y_t = y_t.astype(int)

    n_samples = len(y_t)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    prevalence = float(n_pos) / float(n_samples)

    support_reasons: List[str] = []
    if n_pos == 0:
        support_reasons.append("Zero positive samples (single-class negative subset).")
    if n_neg == 0:
        support_reasons.append("Zero negative samples (single-class positive subset).")
    if n_samples < min_samples:
        support_reasons.append(f"Sample count ({n_samples}) is below min_samples ({min_samples}).")
    if 0 < n_pos < min_positives:
        support_reasons.append(f"Positive count ({n_pos}) is below min_positives ({min_positives}).")
    if 0 < n_neg < min_negatives:
        support_reasons.append(f"Negative count ({n_neg}) is below min_negatives ({min_negatives}).")

    insufficient_support = len(support_reasons) > 0
    both_classes_present = (n_pos > 0) and (n_neg > 0)

    # Proper scoring rules (Brier & Log Loss) are mathematically defined for any non-empty
    # valid probability vector; Brier Skill Score and AUC metrics require both classes.
    brier = float(brier_score_loss(y_t, y_p))
    y_p_clipped = np.clip(y_p.copy(), LOG_LOSS_CLIP_EPS, 1.0 - LOG_LOSS_CLIP_EPS)
    ll = float(
        -np.mean(y_t * np.log(y_p_clipped) + (1.0 - y_t) * np.log(1.0 - y_p_clipped))
    )

    fixed_tm = _compute_threshold_metrics(
        y_true=y_t,
        y_prob=y_p,
        threshold=fixed_threshold,
        threshold_type=f"fixed_{fixed_threshold:.2f}",
    )

    exploratory_tm: Optional[ThresholdMetrics] = None
    if exploratory_train_prevalence_threshold is not None:
        exploratory_tm = _compute_threshold_metrics(
            y_true=y_t,
            y_prob=y_p,
            threshold=float(exploratory_train_prevalence_threshold),
            threshold_type="exploratory_train_prevalence",
        )

    if not both_classes_present:
        return BinaryEvaluationMetrics(
            subset_name=subset_name,
            n_samples=n_samples,
            n_positives=n_pos,
            n_negatives=n_neg,
            prevalence=round(prevalence, 6),
            insufficient_support=True,
            insufficient_support_reasons=tuple(support_reasons),
            roc_auc=None,
            pr_auc=None,
            pr_auc_lift=None,
            pr_auc_ratio=None,
            brier_score=round(brier, 6),
            brier_skill_score=None,
            log_loss=round(ll, 6),
            fixed_threshold_metrics=fixed_tm,
            exploratory_prevalence_threshold_metrics=exploratory_tm,
        )

    roc_val = float(roc_auc_score(y_t, y_p))
    pr_val = float(average_precision_score(y_t, y_p))
    pr_lift = pr_val - prevalence
    pr_ratio = pr_val / prevalence if prevalence > 0.0 else None

    brier_ref = prevalence * (1.0 - prevalence)
    bss_val = 1.0 - (brier / brier_ref) if brier_ref > 0.0 else None

    return BinaryEvaluationMetrics(
        subset_name=subset_name,
        n_samples=n_samples,
        n_positives=n_pos,
        n_negatives=n_neg,
        prevalence=round(prevalence, 6),
        insufficient_support=insufficient_support,
        insufficient_support_reasons=tuple(support_reasons),
        roc_auc=round(roc_val, 6),
        pr_auc=round(pr_val, 6),
        pr_auc_lift=round(pr_lift, 6),
        pr_auc_ratio=round(pr_ratio, 6) if pr_ratio is not None else None,
        brier_score=round(brier, 6),
        brier_skill_score=round(bss_val, 6) if bss_val is not None else None,
        log_loss=round(ll, 6),
        fixed_threshold_metrics=fixed_tm,
        exploratory_prevalence_threshold_metrics=exploratory_tm,
    )


def compute_full_and_cyclic_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    cyclic_mask: Sequence[bool],
    fixed_threshold: float = DEFAULT_FIXED_THRESHOLD,
    exploratory_train_prevalence_threshold: Optional[float] = None,
    min_samples: int = DEFAULT_MIN_METRIC_SAMPLES,
    min_positives: int = DEFAULT_MIN_METRIC_POSITIVES,
    min_negatives: int = DEFAULT_MIN_METRIC_POSITIVES,
) -> Dict[str, BinaryEvaluationMetrics]:
    """Compute evaluation metrics on both the full cohort and the cyclic-only subset."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    c_mask = np.asarray(cyclic_mask, dtype=bool)

    if len(y_t) != len(c_mask) or len(y_p) != len(c_mask):
        raise ScaffoldGeneralizationError(
            f"Lengths of y_true ({len(y_t)}), y_prob ({len(y_p)}), and cyclic_mask ({len(c_mask)}) must match."
        )

    full_metrics = compute_binary_evaluation_metrics(
        y_true=y_t,
        y_prob=y_p,
        subset_name="full_cohort",
        fixed_threshold=fixed_threshold,
        exploratory_train_prevalence_threshold=exploratory_train_prevalence_threshold,
        min_samples=min_samples,
        min_positives=min_positives,
        min_negatives=min_negatives,
    )

    cyclic_metrics = compute_binary_evaluation_metrics(
        y_true=y_t[c_mask],
        y_prob=y_p[c_mask],
        subset_name="cyclic_only",
        fixed_threshold=fixed_threshold,
        exploratory_train_prevalence_threshold=exploratory_train_prevalence_threshold,
        min_samples=min_samples,
        min_positives=min_positives,
        min_negatives=min_negatives,
    )

    return {
        "full_cohort": full_metrics,
        "cyclic_only": cyclic_metrics,
    }


# ---------------------------------------------------------------------------
# 6. Fold Dispersion & Out-of-Fold Separation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDispersion:
    """Fold-to-fold dispersion statistics (mean, sample SD, min, max) for a single scalar metric."""

    metric_name: str
    n_total_folds: int
    n_valid_folds: int
    n_undefined_folds: int
    mean: Optional[float]
    sd: Optional[float]
    min: Optional[float]
    max: Optional[float]
    fold_values: Tuple[Optional[float], ...]


@dataclass(frozen=True)
class CrossValidationFoldSummary:
    """Separately reports fold-level metric dispersion and pooled out-of-fold metrics."""

    subset_name: str
    n_folds: int
    n_Any_insufficient_support_folds: int
    fold_metrics: Tuple[BinaryEvaluationMetrics, ...]
    fold_dispersion: Dict[str, MetricDispersion]
    pooled_oof_metrics: BinaryEvaluationMetrics


def summarize_fold_and_pooled_metrics(
    fold_metrics_list: Sequence[BinaryEvaluationMetrics],
    pooled_y_true: Sequence[int],
    pooled_y_prob: Sequence[float],
    subset_name: str = "full_cohort",
    fixed_threshold: float = DEFAULT_FIXED_THRESHOLD,
    exploratory_train_prevalence_threshold: Optional[float] = None,
    min_samples: int = DEFAULT_MIN_METRIC_SAMPLES,
    min_positives: int = DEFAULT_MIN_METRIC_POSITIVES,
    min_negatives: int = DEFAULT_MIN_METRIC_POSITIVES,
) -> CrossValidationFoldSummary:
    """Summarize fold-level dispersion separately from pooled out-of-fold metrics.

    Avoids treating dependent folds as independent experiments while quantifying both:
    1. ``fold_dispersion``: mean, sample standard deviation (ddof=1 when >=2 valid folds),
       min, and max across the K folds.
    2. ``pooled_oof_metrics``: metrics computed across the concatenated out-of-fold
       predictions where each molecule is predicted once by the model from which its
       scaffold was held out.
    """
    if len(fold_metrics_list) == 0:
        raise ScaffoldGeneralizationError("fold_metrics_list cannot be empty.")

    n_folds = len(fold_metrics_list)
    n_insufficient = sum(1 for fm in fold_metrics_list if fm.insufficient_support)

    def _extract_stat(fm: BinaryEvaluationMetrics, name: str) -> Optional[float]:
        if name in (
            "roc_auc",
            "pr_auc",
            "pr_auc_lift",
            "pr_auc_ratio",
            "brier_score",
            "brier_skill_score",
            "log_loss",
            "prevalence",
        ):
            return getattr(fm, name)
        if fm.fixed_threshold_metrics is not None and hasattr(fm.fixed_threshold_metrics, name):
            return float(getattr(fm.fixed_threshold_metrics, name))
        return None

    tracked_metric_names = (
        "roc_auc",
        "pr_auc",
        "pr_auc_lift",
        "pr_auc_ratio",
        "brier_score",
        "brier_skill_score",
        "log_loss",
        "prevalence",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
    )

    dispersion_map: Dict[str, MetricDispersion] = {}
    for m_name in tracked_metric_names:
        raw_vals = tuple(_extract_stat(fm, m_name) for fm in fold_metrics_list)
        valid_vals = [float(v) for v in raw_vals if v is not None and np.isfinite(v)]
        n_valid = len(valid_vals)
        n_undef = n_folds - n_valid

        if n_valid == 0:
            dispersion_map[m_name] = MetricDispersion(
                metric_name=m_name,
                n_total_folds=n_folds,
                n_valid_folds=0,
                n_undefined_folds=n_undef,
                mean=None,
                sd=None,
                min=None,
                max=None,
                fold_values=raw_vals,
            )
        else:
            arr = np.asarray(valid_vals, dtype=float)
            sd_val = float(np.std(arr, ddof=1)) if n_valid >= 2 else 0.0
            dispersion_map[m_name] = MetricDispersion(
                metric_name=m_name,
                n_total_folds=n_folds,
                n_valid_folds=n_valid,
                n_undefined_folds=n_undef,
                mean=round(float(np.mean(arr)), 6),
                sd=round(sd_val, 6),
                min=round(float(np.min(arr)), 6),
                max=round(float(np.max(arr)), 6),
                fold_values=raw_vals,
            )

    pooled_metrics = compute_binary_evaluation_metrics(
        y_true=pooled_y_true,
        y_prob=pooled_y_prob,
        subset_name=subset_name,
        fixed_threshold=fixed_threshold,
        exploratory_train_prevalence_threshold=exploratory_train_prevalence_threshold,
        min_samples=min_samples,
        min_positives=min_positives,
        min_negatives=min_negatives,
    )

    return CrossValidationFoldSummary(
        subset_name=subset_name,
        n_folds=n_folds,
        n_Any_insufficient_support_folds=n_insufficient,
        fold_metrics=tuple(fold_metrics_list),
        fold_dispersion=dispersion_map,
        pooled_oof_metrics=pooled_metrics,
    )


def summarize_repeated_cv_runs(
    seed_to_summary: Dict[int, CrossValidationFoldSummary],
) -> Dict[str, Any]:
    """Summarize repeated k-fold CV runs across seeds (e.g. 42, 1042, 2042) without pooling repeats.

    Because repeated CV predictions across seeds evaluate the same N molecules multiple
    times, concatenating R * N predictions would artificially inflate sample support by R.
    Instead, this function summarizes each seed's pooled out-of-fold metric and reports the
    across-seed mean, SD, min, and max.
    """
    if len(seed_to_summary) == 0:
        raise ScaffoldGeneralizationError("seed_to_summary cannot be empty.")

    ordered_seeds = sorted(seed_to_summary.keys())
    metric_names = (
        "roc_auc",
        "pr_auc",
        "pr_auc_lift",
        "pr_auc_ratio",
        "brier_score",
        "brier_skill_score",
        "log_loss",
    )

    across_seed_stats: Dict[str, Dict[str, Any]] = {}
    for m_name in metric_names:
        per_seed = {
            int(s): getattr(seed_to_summary[s].pooled_oof_metrics, m_name)
            for s in ordered_seeds
        }
        valid = [float(v) for v in per_seed.values() if v is not None and np.isfinite(v)]
        if not valid:
            across_seed_stats[m_name] = {
                "per_seed": per_seed,
                "mean": None,
                "sd": None,
                "min": None,
                "max": None,
            }
        else:
            arr = np.asarray(valid, dtype=float)
            across_seed_stats[m_name] = {
                "per_seed": per_seed,
                "mean": round(float(np.mean(arr)), 6),
                "sd": round(float(np.std(arr, ddof=1)), 6) if len(valid) >= 2 else 0.0,
                "min": round(float(np.min(arr)), 6),
                "max": round(float(np.max(arr)), 6),
            }

    return {
        "seeds": ordered_seeds,
        "n_repeats": len(ordered_seeds),
        "across_seed_pooled_oof_summary": across_seed_stats,
    }


# ---------------------------------------------------------------------------
# 7. Scaffold-Cluster Block Bootstrap & Molecule-Paired Regime Comparisons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterBootstrapInterval:
    """Scaffold-cluster block bootstrap confidence interval with invalid-replicate accounting."""

    metric_name: str
    point_estimate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    ci_level: float
    n_requested_replicates: int
    n_valid_replicates: int
    n_invalid_replicates: int
    n_total_clusters: int
    n_positive_clusters: int
    insufficient_cluster_support: bool
    insufficient_support_reason: Optional[str]
    limitations_note: str = CLUSTER_BOOTSTRAP_LIMITATIONS_NOTE


@dataclass(frozen=True)
class PairedClusterBootstrapComparison:
    """Molecule-paired scaffold-cluster block bootstrap comparison between two regimes."""

    metric_name: str
    regime_a_name: str
    regime_b_name: str
    estimate_a: Optional[float]
    estimate_b: Optional[float]
    delta_a_minus_b: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    ci_level: float
    ci_excludes_zero: bool
    n_molecules: int
    n_requested_replicates: int
    n_valid_replicates: int
    n_invalid_replicates: int
    n_total_clusters: int
    n_positive_clusters: int
    insufficient_cluster_support: bool
    interpretation_note: str


def _eval_scalar_metric(
    metric_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Optional[float]:
    """Compute a single scalar metric on a bootstrap sample, returning None if undefined."""
    n = len(y_true)
    if n == 0:
        return None
    n_pos = int(np.sum(y_true == 1))
    n_neg = n - n_pos
    prev = float(n_pos) / float(n)

    if metric_name == "brier_score":
        return float(brier_score_loss(y_true, y_prob))
    if metric_name == "log_loss":
        clipped = np.clip(y_prob.copy(), LOG_LOSS_CLIP_EPS, 1.0 - LOG_LOSS_CLIP_EPS)
        return float(-np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)))

    # Metrics requiring both positive and negative classes
    if n_pos == 0 or n_neg == 0:
        return None

    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if metric_name == "pr_auc":
        return float(average_precision_score(y_true, y_prob))
    if metric_name == "pr_auc_lift":
        return float(average_precision_score(y_true, y_prob)) - prev
    if metric_name == "pr_auc_ratio":
        return float(average_precision_score(y_true, y_prob)) / prev if prev > 0.0 else None
    if metric_name == "brier_skill_score":
        brier = float(brier_score_loss(y_true, y_prob))
        denom = prev * (1.0 - prev)
        return 1.0 - (brier / denom) if denom > 0.0 else None

    raise ScaffoldGeneralizationError(f"Unsupported bootstrap metric '{metric_name}'.")


def compute_scaffold_cluster_bootstrap(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    cluster_ids: Sequence[str],
    metric_names: Sequence[str] = ("roc_auc", "pr_auc", "pr_auc_lift", "brier_score", "log_loss"),
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
    seed: int = DEFAULT_RANDOM_SEED,
    min_positive_clusters: int = DEFAULT_MIN_POSITIVE_CLUSTERS_FOR_BOOTSTRAP,
    min_valid_replicate_fraction: float = 0.80,
) -> Dict[str, ClusterBootstrapInterval]:
    """Compute deterministic scaffold-cluster block bootstrap confidence intervals.

    Resamples scaffold clusters ``c_1*, ..., c_G*`` with replacement (where ``G`` is the number
    of distinct clusters) and concatenates all member molecules of the drawn clusters in each
    replicate. Explicitly tracks degenerate/invalid bootstrap replicates (e.g. when sparse
    positive clusters result in a replicate with 0 positives) rather than silently crashing
    or misreporting.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    if len(y_t) != len(y_p) or len(y_t) != len(cluster_ids):
        raise ScaffoldGeneralizationError(
            f"Lengths of y_true ({len(y_t)}), y_prob ({len(y_p)}), and "
            f"cluster_ids ({len(cluster_ids)}) must match."
        )
    if n_bootstraps < 10:
        raise ScaffoldGeneralizationError(f"n_bootstraps must be >= 10, got {n_bootstraps}.")
    if not (0.50 <= ci_level < 1.0):
        raise ScaffoldGeneralizationError(f"ci_level must be in [0.50, 1.0), got {ci_level}.")

    # Group sample indices by cluster ID in deterministic sorted order of cluster IDs
    cluster_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices[str(cid)].append(idx)

    unique_clusters = sorted(cluster_to_indices.keys())
    n_clusters = len(unique_clusters)
    cluster_idx_arrays = [np.asarray(cluster_to_indices[c], dtype=int) for c in unique_clusters]

    n_pos_clusters = sum(
        1 for arr in cluster_idx_arrays if np.any(y_t[arr] == 1)
    )
    n_neg_clusters = sum(
        1 for arr in cluster_idx_arrays if np.any(y_t[arr] == 0)
    )

    point_estimates = {
        m_name: _eval_scalar_metric(m_name, y_t, y_p)
        for m_name in metric_names
    }

    insufficient_reason: Optional[str] = None
    if n_clusters < 2:
        insufficient_reason = f"Only {n_clusters} distinct scaffold cluster(s) present; cannot block-bootstrap."
    elif n_pos_clusters < min_positive_clusters:
        insufficient_reason = (
            f"Positive events occur in only {n_pos_clusters} scaffold cluster(s) "
            f"(minimum required for reliable cluster bootstrap: {min_positive_clusters})."
        )
    elif n_neg_clusters < 2:
        insufficient_reason = f"Negative events occur in only {n_neg_clusters} scaffold cluster(s)."

    # Still execute the bootstrap draws when n_clusters >= 2 so valid/invalid replicate counts
    # are empirically recorded, flagging insufficient_cluster_support when applicable.
    replicate_values: Dict[str, List[float]] = {m_name: [] for m_name in metric_names}
    invalid_counts: Dict[str, int] = {m_name: 0 for m_name in metric_names}

    if n_clusters >= 1 and len(y_t) > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstraps):
            drawn_cluster_positions = rng.integers(0, n_clusters, size=n_clusters)
            sampled_indices = np.concatenate([cluster_idx_arrays[pos] for pos in drawn_cluster_positions])
            y_t_b = y_t[sampled_indices]
            y_p_b = y_p[sampled_indices]

            for m_name in metric_names:
                val = _eval_scalar_metric(m_name, y_t_b, y_p_b)
                if val is None or not np.isfinite(val):
                    invalid_counts[m_name] += 1
                else:
                    replicate_values[m_name].append(float(val))
    else:
        for m_name in metric_names:
            invalid_counts[m_name] = n_bootstraps

    alpha = 1.0 - float(ci_level)
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    min_valid_reps = int(np.ceil(min_valid_replicate_fraction * float(n_bootstraps)))

    results: Dict[str, ClusterBootstrapInterval] = {}
    for m_name in metric_names:
        vals = replicate_values[m_name]
        n_valid = len(vals)
        n_invalid = invalid_counts[m_name]
        pt = point_estimates[m_name]

        m_insufficient = (insufficient_reason is not None) or (n_valid < min_valid_reps) or (pt is None)
        m_reason = insufficient_reason
        if m_reason is None and n_valid < min_valid_reps:
            m_reason = (
                f"Only {n_valid}/{n_bootstraps} valid bootstrap replicates "
                f"(minimum required: {min_valid_reps})."
            )

        if pt is not None and n_valid >= min_valid_reps and insufficient_reason is None:
            arr = np.asarray(vals, dtype=float)
            ci_lo = round(float(np.percentile(arr, lower_pct)), 6)
            ci_hi = round(float(np.percentile(arr, upper_pct)), 6)
        else:
            ci_lo = None
            ci_hi = None

        results[m_name] = ClusterBootstrapInterval(
            metric_name=m_name,
            point_estimate=round(pt, 6) if pt is not None else None,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_level=ci_level,
            n_requested_replicates=n_bootstraps,
            n_valid_replicates=n_valid,
            n_invalid_replicates=n_invalid,
            n_total_clusters=n_clusters,
            n_positive_clusters=n_pos_clusters,
            insufficient_cluster_support=m_insufficient,
            insufficient_support_reason=m_reason,
        )

    return results


def compare_paired_oof_predictions(
    y_true: Sequence[int],
    prob_regime_a: Sequence[float],
    prob_regime_b: Sequence[float],
    cluster_ids: Sequence[str],
    regime_a_name: str = "scaffold_grouped_kfold",
    regime_b_name: str = "random_stratified_kfold",
    metric_names: Sequence[str] = ("roc_auc", "pr_auc", "pr_auc_lift", "brier_score", "log_loss"),
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
    seed: int = DEFAULT_RANDOM_SEED,
    min_positive_clusters: int = DEFAULT_MIN_POSITIVE_CLUSTERS_FOR_BOOTSTRAP,
    min_valid_replicate_fraction: float = 0.80,
) -> Dict[str, PairedClusterBootstrapComparison]:
    """Compare two out-of-fold prediction vectors on a matched cohort via paired cluster bootstrap.

    Preserves molecule-level pairing ``(y_true[i], prob_regime_a[i], prob_regime_b[i])`` for
    every molecule ``i`` and applies identical scaffold-cluster draws to both regimes in each
    bootstrap replicate. Does not claim statistical significance automatically; reports the
    observed difference ``delta_a_minus_b``, percentile interval, and descriptive note.
    """
    y_t = np.asarray(y_true, dtype=int)
    p_a = np.asarray(prob_regime_a, dtype=float)
    p_b = np.asarray(prob_regime_b, dtype=float)

    n_mol = len(y_t)
    if len(p_a) != n_mol or len(p_b) != n_mol or len(cluster_ids) != n_mol:
        raise ScaffoldGeneralizationError(
            f"Matched comparison requires identical lengths: y_true={n_mol}, "
            f"prob_a={len(p_a)}, prob_b={len(p_b)}, cluster_ids={len(cluster_ids)}."
        )
    if not (np.all(np.isfinite(p_a)) and np.all(np.isfinite(p_b))):
        raise ScaffoldGeneralizationError("Prediction arrays must not contain NaN or infinite values.")
    if np.any((p_a < 0.0) | (p_a > 1.0)) or np.any((p_b < 0.0) | (p_b > 1.0)):
        raise ScaffoldGeneralizationError("Prediction probabilities must lie within [0.0, 1.0].")

    cluster_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices[str(cid)].append(idx)

    unique_clusters = sorted(cluster_to_indices.keys())
    n_clusters = len(unique_clusters)
    cluster_idx_arrays = [np.asarray(cluster_to_indices[c], dtype=int) for c in unique_clusters]
    n_pos_clusters = sum(1 for arr in cluster_idx_arrays if np.any(y_t[arr] == 1))
    n_neg_clusters = sum(1 for arr in cluster_idx_arrays if np.any(y_t[arr] == 0))

    insufficient_base = (
        n_clusters < 2
        or n_pos_clusters < min_positive_clusters
        or n_neg_clusters < 2
    )

    deltas_by_metric: Dict[str, List[float]] = {m_name: [] for m_name in metric_names}
    invalid_by_metric: Dict[str, int] = {m_name: 0 for m_name in metric_names}

    if n_clusters >= 1 and n_mol > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstraps):
            drawn = rng.integers(0, n_clusters, size=n_clusters)
            sampled_idx = np.concatenate([cluster_idx_arrays[pos] for pos in drawn])
            y_b = y_t[sampled_idx]
            pa_b = p_a[sampled_idx]
            pb_b = p_b[sampled_idx]

            for m_name in metric_names:
                va = _eval_scalar_metric(m_name, y_b, pa_b)
                vb = _eval_scalar_metric(m_name, y_b, pb_b)
                if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)):
                    invalid_by_metric[m_name] += 1
                else:
                    deltas_by_metric[m_name].append(float(va - vb))
    else:
        for m_name in metric_names:
            invalid_by_metric[m_name] = n_bootstraps

    alpha = 1.0 - float(ci_level)
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    min_valid_reps = int(np.ceil(min_valid_replicate_fraction * float(n_bootstraps)))

    comparisons: Dict[str, PairedClusterBootstrapComparison] = {}
    for m_name in metric_names:
        est_a = _eval_scalar_metric(m_name, y_t, p_a)
        est_b = _eval_scalar_metric(m_name, y_t, p_b)
        delta_pt = (est_a - est_b) if (est_a is not None and est_b is not None) else None

        d_vals = deltas_by_metric[m_name]
        n_valid = len(d_vals)
        n_invalid = invalid_by_metric[m_name]
        is_insuff = insufficient_base or (n_valid < min_valid_reps) or (delta_pt is None)

        if not is_insuff:
            arr = np.asarray(d_vals, dtype=float)
            ci_lo = round(float(np.percentile(arr, lower_pct)), 6)
            ci_hi = round(float(np.percentile(arr, upper_pct)), 6)
            excludes_zero = bool(ci_lo > 0.0 or ci_hi < 0.0)
            note = (
                f"Paired cluster bootstrap ({n_valid}/{n_bootstraps} valid replicates across "
                f"{n_clusters} clusters, {n_pos_clusters} positive clusters): "
                f"delta ({regime_a_name} - {regime_b_name}) = {delta_pt:+.4f} "
                f"[{ci_lo:+.4f}, {ci_hi:+.4f}]."
            )
        else:
            ci_lo = None
            ci_hi = None
            excludes_zero = False
            note = (
                f"Insufficient cluster support for reliable paired bootstrap CI "
                f"(n_clusters={n_clusters}, n_positive_clusters={n_pos_clusters}, "
                f"valid_replicates={n_valid}/{n_bootstraps})."
            )

        comparisons[m_name] = PairedClusterBootstrapComparison(
            metric_name=m_name,
            regime_a_name=regime_a_name,
            regime_b_name=regime_b_name,
            estimate_a=round(est_a, 6) if est_a is not None else None,
            estimate_b=round(est_b, 6) if est_b is not None else None,
            delta_a_minus_b=round(delta_pt, 6) if delta_pt is not None else None,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_level=ci_level,
            ci_excludes_zero=excludes_zero,
            n_molecules=n_mol,
            n_requested_replicates=n_bootstraps,
            n_valid_replicates=n_valid,
            n_invalid_replicates=n_invalid,
            n_total_clusters=n_clusters,
            n_positive_clusters=n_pos_clusters,
            insufficient_cluster_support=is_insuff,
            interpretation_note=note,
        )

    return comparisons
