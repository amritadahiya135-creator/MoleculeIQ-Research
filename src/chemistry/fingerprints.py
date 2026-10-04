"""Molecular fingerprint generation engine (Module M03)."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ExplicitBitVect, ConvertToNumpyArray


@dataclass
class FingerprintResult:
    """Container for generated molecular fingerprints and topological bit provenance."""
    bit_vector: ExplicitBitVect
    numpy_vector: np.ndarray
    radius: int
    n_bits: int
    bit_info: Dict[int, List[Tuple[int, int]]]  # bit_id -> list of (atom_idx, radius)


def generate_morgan_fingerprint(
    mol_or_smiles: Union[Chem.Mol, str],
    radius: int = 2,
    n_bits: int = 2048,
    return_bit_info: bool = True,
) -> FingerprintResult:
    """
    Generate standard Morgan circular fingerprint (ECFP4 equivalent) for a molecule.

    Ensures mathematical compatibility across training, inference, and similarity indexing.

    Parameters
    ----------
    mol_or_smiles : Union[Chem.Mol, str]
        Target RDKit molecule or SMILES string.
    radius : int, default=2
        Circular radius in bonds (radius 2 corresponds to topological diameter 4, i.e., ECFP4).
    n_bits : int, default=2048
        Bit length for fixed-size folded bit vector.
    return_bit_info : bool, default=True
        Whether to calculate atom environments mapped to active bits (used for SHAP XAI).

    Returns
    -------
    FingerprintResult
        Contains ExplicitBitVect, NumPy array (2048,), and atom bit environments.

    Raises
    ------
    ValueError
        If SMILES is invalid or mol is None.
    """
    if isinstance(mol_or_smiles, str):
        clean = mol_or_smiles.strip()
        if not clean:
            raise ValueError("Cannot generate fingerprint for an empty SMILES string.")
        mol = Chem.MolFromSmiles(clean)
        if mol is None:
            raise ValueError(f"Cannot generate fingerprint: Invalid SMILES '{mol_or_smiles}'.")
    elif isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
    else:
        raise TypeError(f"Expected RDKit Mol or SMILES string, got {type(mol_or_smiles).__name__}.")

    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    
    # Generate explicit circular fingerprint with atom environment mapping
    blocker = Chem.rdBase.BlockLogs()
    try:
        if hasattr(Chem, "rdFingerprintGenerator"):
            gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            if return_bit_info:
                ao = Chem.rdFingerprintGenerator.AdditionalOutput()
                ao.AllocateBitInfoMap()
                fp_bitvect = gen.GetFingerprint(mol, additionalOutput=ao)
                raw_info = ao.GetBitInfoMap()
                bit_info = {int(k): [tuple(x) for x in v] for k, v in raw_info.items()}
            else:
                fp_bitvect = gen.GetFingerprint(mol)
        else:
            fp_bitvect = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=n_bits,
                bitInfo=bit_info if return_bit_info else None,
            )
    finally:
        del blocker

    # Convert to 1D binary NumPy array for scikit-learn models
    arr = np.zeros((n_bits,), dtype=np.int8)
    ConvertToNumpyArray(fp_bitvect, arr)

    if arr.shape[0] != n_bits:
        raise ValueError(f"Fingerprint dimension mismatch: Expected {n_bits}, got {arr.shape[0]}")

    return FingerprintResult(
        bit_vector=fp_bitvect,
        numpy_vector=arr,
        radius=radius,
        n_bits=n_bits,
        bit_info=bit_info,
    )
