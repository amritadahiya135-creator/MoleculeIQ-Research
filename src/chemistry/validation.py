"""Molecular structure validation and canonicalization engine (Module M01)."""

from dataclasses import dataclass
from typing import Optional, List
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdMolDescriptors


@dataclass(frozen=True)
class MoleculeValidationResult:
    """Immutable validation record for an input molecular structure."""
    original_smiles: str
    canonical_smiles: Optional[str]
    is_valid: bool
    formula: Optional[str] = None
    num_atoms: Optional[int] = None
    num_heavy_atoms: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


def validate_smiles(smiles: str) -> MoleculeValidationResult:
    """
    Validate and canonicalize a SMILES string using RDKit.

    Preserves original input string for data traceability.
    Catches empty strings, whitespace, malformed chemical syntax, and valence errors.

    Parameters
    ----------
    smiles : str
        Input chemical line entry notation.

    Returns
    -------
    MoleculeValidationResult
        Structured validation outcome with canonical representation and metadata.
    """
    if not isinstance(smiles, str):
        return MoleculeValidationResult(
            original_smiles=str(smiles),
            canonical_smiles=None,
            is_valid=False,
            error_code="E-001_INVALID_TYPE",
            error_message="Input SMILES must be a string.",
        )

    clean_smiles = smiles.strip()
    if not clean_smiles:
        return MoleculeValidationResult(
            original_smiles=smiles,
            canonical_smiles=None,
            is_valid=False,
            error_code="E-001_EMPTY_INPUT",
            error_message="SMILES string is empty or contains only whitespace.",
        )

    # Disable RDKit C++ stderr spam temporarily during validation
    blocker = rdBase.BlockLogs()
    try:
        mol = Chem.MolFromSmiles(clean_smiles)
    except Exception as exc:
        return MoleculeValidationResult(
            original_smiles=smiles,
            canonical_smiles=None,
            is_valid=False,
            error_code="E-001_PARSE_EXCEPTION",
            error_message=f"RDKit parse exception: {str(exc)}",
        )
    finally:
        del blocker

    if mol is None:
        return MoleculeValidationResult(
            original_smiles=smiles,
            canonical_smiles=None,
            is_valid=False,
            error_code="E-001_MALFORMED_SMILES",
            error_message="Failed to parse SMILES: chemically invalid structure or invalid syntax.",
        )

    try:
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        num_atoms = mol.GetNumAtoms()
        num_heavy_atoms = mol.GetNumHeavyAtoms()

        return MoleculeValidationResult(
            original_smiles=smiles,
            canonical_smiles=canonical_smiles,
            is_valid=True,
            formula=formula,
            num_atoms=num_atoms,
            num_heavy_atoms=num_heavy_atoms,
            error_code=None,
            error_message=None,
        )
    except Exception as exc:
        return MoleculeValidationResult(
            original_smiles=smiles,
            canonical_smiles=None,
            is_valid=False,
            error_code="E-001_PROCESSING_ERROR",
            error_message=f"Error generating canonical structure: {str(exc)}",
        )


def validate_batch_smiles(smiles_list: List[str]) -> List[MoleculeValidationResult]:
    """
    Validate a sequence of SMILES strings with row-level fault isolation.

    An error in one row will never impede or invalidate the validation of subsequent rows.
    """
    return [validate_smiles(s) for s in smiles_list]
