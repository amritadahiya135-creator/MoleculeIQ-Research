"""Molecular descriptor and Lipinski rule evaluation engine (Module M02)."""

from dataclasses import dataclass
from typing import Optional, List, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


@dataclass(frozen=True)
class MolecularDescriptors:
    """Immutable record of calculated physicochemical properties and drug-likeness indicators."""
    molecular_weight: float
    logp: float
    hbond_donors: int
    hbond_acceptors: int
    rotatable_bonds: int
    tpsa: float
    lipinski_violations: int
    lipinski_pass: bool
    violation_details: List[str]


def calculate_descriptors(mol_or_smiles: Union[Chem.Mol, str]) -> MolecularDescriptors:
    """
    Calculate essential physicochemical properties and Lipinski Rule of 5 compliance.

    Evaluated thresholds (Lipinski 1997 heuristic for oral bioavailability):
      - Molecular Weight <= 500 Da
      - MolLogP <= 5.0
      - Hydrogen Bond Donors (NumHDonors) <= 5
      - Hydrogen Bond Acceptors (NumHAcceptors) <= 10

    Parameters
    ----------
    mol_or_smiles : Union[Chem.Mol, str]
        Either a pre-parsed RDKit Mol object or a SMILES string.

    Returns
    -------
    MolecularDescriptors
        Calculated physicochemical properties and Lipinski evaluation.

    Raises
    ------
    ValueError
        If the structure cannot be parsed or is invalid.
    """
    if isinstance(mol_or_smiles, str):
        clean = mol_or_smiles.strip()
        if not clean:
            raise ValueError("Cannot calculate descriptors for an empty SMILES string.")
        mol = Chem.MolFromSmiles(clean)
        if mol is None:
            raise ValueError(f"Cannot calculate descriptors: Invalid SMILES '{mol_or_smiles}'.")
    elif isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
    else:
        raise TypeError(f"Expected RDKit Mol or SMILES string, got {type(mol_or_smiles).__name__}.")

    # Calculate exact properties via RDKit standard routines
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot_bonds = int(Lipinski.NumRotatableBonds(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))

    # Evaluate Lipinski violations
    violations = 0
    violation_details = []

    if mw > 500.0:
        violations += 1
        violation_details.append(f"Molecular weight ({mw:.2f} Da) exceeds 500 Da.")
    if logp > 5.0:
        violations += 1
        violation_details.append(f"LogP ({logp:.2f}) exceeds 5.0.")
    if hbd > 5:
        violations += 1
        violation_details.append(f"H-bond donors ({hbd}) exceed 5.")
    if hba > 10:
        violations += 1
        violation_details.append(f"H-bond acceptors ({hba}) exceed 10.")

    lipinski_pass = (violations == 0)

    return MolecularDescriptors(
        molecular_weight=round(mw, 3),
        logp=round(logp, 3),
        hbond_donors=hbd,
        hbond_acceptors=hba,
        rotatable_bonds=rot_bonds,
        tpsa=round(tpsa, 3),
        lipinski_violations=violations,
        lipinski_pass=lipinski_pass,
        violation_details=violation_details,
    )
