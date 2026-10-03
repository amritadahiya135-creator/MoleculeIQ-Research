"""Bemis-Murcko scaffold decomposition and scaffold-based dataset splitting (Module M01, M03)."""

import logging
from collections import defaultdict
from typing import List, Tuple, Union, Dict, Set
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def generate_bemis_murcko_scaffold(
    mol_or_smiles: Union[Chem.Mol, str],
    include_chirality: bool = False,
) -> str:
    """
    Extract the core Bemis-Murcko molecular scaffold SMILES.

    Parameters
    ----------
    mol_or_smiles : Union[Chem.Mol, str]
        Input RDKit molecule or SMILES string.
    include_chirality : bool, default=False
        Whether to retain stereochemical tags in the extracted scaffold.

    Returns
    -------
    str
        Canonical SMILES of the Murcko scaffold, or empty string "" if acyclic.
    """
    if isinstance(mol_or_smiles, str):
        smi = mol_or_smiles.strip()
        if not smi:
            return ""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
    elif isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
    else:
        raise TypeError(f"Expected RDKit Mol or SMILES string, got {type(mol_or_smiles).__name__}.")

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality,
        )
        return scaffold
    except Exception as exc:
        logger.warning(f"Murcko scaffold generation failed: {exc}")
        return ""


def scaffold_train_test_split(
    smiles_list: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Partition compounds into train and test splits based on Bemis-Murcko scaffolds.

    Guarantees zero scaffold leakage: all molecules with the same Murcko scaffold
    are placed exclusively into the training set or test set, never both.

    Parameters
    ----------
    smiles_list : List[str]
        List of canonical SMILES strings corresponding to indexed dataset samples.
    test_size : float, default=0.2
        Target proportion of the dataset to allocate to the test split.
    random_state : int, default=42
        Deterministic seed for tie-breaking scaffold bucket assignments.

    Returns
    -------
    Tuple[List[int], List[int]]
        (train_indices, test_indices)
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}.")

    total_count = len(smiles_list)
    if total_count == 0:
        return [], []

    # Map each compound to its scaffold
    scaffold_buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaf = generate_bemis_murcko_scaffold(smi)
        # For acyclic molecules (empty scaffold), treat each unique SMILES as its own cluster
        # to prevent all acyclics from coalescing into a single massive cluster
        bucket_key = scaf if scaf else f"acyclic_{smi}"
        scaffold_buckets[bucket_key].append(idx)

    # Sort scaffold buckets by size descending, then by scaffold key for deterministic ordering
    sorted_groups = sorted(
        scaffold_buckets.values(),
        key=lambda grp: (len(grp), grp[0]),
        reverse=True,
    )

    target_train_count = int(total_count * (1.0 - test_size))
    train_indices: List[int] = []
    test_indices: List[int] = []

    for group in sorted_groups:
        if len(train_indices) + len(group) <= target_train_count:
            train_indices.extend(group)
        else:
            test_indices.extend(group)

    # Sanity checks
    assert len(set(train_indices).intersection(set(test_indices))) == 0, "Index overlap detected!"
    assert len(train_indices) + len(test_indices) == total_count, "Sample loss detected!"

    logger.info(
        f"Scaffold split: {len(train_indices)} train ({len(train_indices)/total_count:.1%}), "
        f"{len(test_indices)} test ({len(test_indices)/total_count:.1%}) across {len(scaffold_buckets)} scaffolds."
    )

    return train_indices, test_indices
