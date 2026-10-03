"""Curator and cache manager for the approved small-molecule reference library (Module M06)."""

import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from src.chemistry.validation import validate_smiles
from src.chemistry.descriptors import calculate_descriptors

logger = logging.getLogger(__name__)

REFERENCE_CACHE_PATH = "data/reference/approved_drugs.csv"

# Essential well-characterized pharmaceutical reference compounds for gold-standard anchoring
ANCHOR_DRUGS = [
    {"id": "CHEMBL25", "name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"id": "CHEMBL112", "name": "Acetaminophen (Paracetamol)", "smiles": "CC(=O)Nc1ccc(O)cc1"},
    {"id": "CHEMBL521", "name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"},
    {"id": "CHEMBL113", "name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    {"id": "CHEMBL1487", "name": "Atorvastatin", "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"},
    {"id": "CHEMBL1431", "name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N"},
    {"id": "CHEMBL1503", "name": "Omeprazole", "smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"},
    {"id": "CHEMBL148", "name": "Penicillin G", "smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"},
    {"id": "CHEMBL564", "name": "Amoxicillin", "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"},
    {"id": "CHEMBL154", "name": "Ciprofloxacin", "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"},
    {"id": "CHEMBL143", "name": "Simvastatin", "smiles": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12"},
    {"id": "CHEMBL417", "name": "Lisinopril", "smiles": "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O"},
]


def build_approved_reference_cache(
    clintox_path: str = "data/raw/clintox.csv",
    output_path: str = REFERENCE_CACHE_PATH,
) -> pd.DataFrame:
    """
    Construct and persist the curated reference small-molecule library.

    Combines canonical FDA-approved small molecules from ClinTox and anchored approved drugs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records: List[Dict[str, Any]] = []
    seen_canonical = set()

    # Step 1: Add Anchor Drugs with verified names
    for drug in ANCHOR_DRUGS:
        val_res = validate_smiles(drug["smiles"])
        if val_res.is_valid and val_res.canonical_smiles:
            seen_canonical.add(val_res.canonical_smiles)
            desc = calculate_descriptors(val_res.canonical_smiles)
            records.append({
                "reference_id": drug["id"],
                "name": drug["name"],
                "smiles": drug["smiles"],
                "canonical_smiles": val_res.canonical_smiles,
                "formula": val_res.formula,
                "molecular_weight": desc.molecular_weight,
                "logp": desc.logp,
                "lipinski_pass": desc.lipinski_pass,
                "source": "ChEMBL_Curated_Anchor",
            })

    # Step 2: Add Approved Molecules from ClinTox (FDA_APPROVED == 1)
    if os.path.exists(clintox_path):
        df_clin = pd.read_csv(clintox_path)
        app_df = df_clin[df_clin["FDA_APPROVED"] == 1]

        for idx, row in app_df.iterrows():
            smi = str(row["smiles"])
            val_res = validate_smiles(smi)
            if not val_res.is_valid or not val_res.canonical_smiles:
                continue

            if val_res.canonical_smiles in seen_canonical:
                continue

            seen_canonical.add(val_res.canonical_smiles)
            try:
                desc = calculate_descriptors(val_res.canonical_smiles)
                records.append({
                    "reference_id": f"FDA_APP_{len(records)+1:04d}",
                    "name": f"Approved Compound {len(records)+1}",
                    "smiles": smi,
                    "canonical_smiles": val_res.canonical_smiles,
                    "formula": val_res.formula,
                    "molecular_weight": desc.molecular_weight,
                    "logp": desc.logp,
                    "lipinski_pass": desc.lipinski_pass,
                    "source": "FDA_Approved_ClinTox",
                })
            except Exception:
                continue

    ref_df = pd.DataFrame(records)
    ref_df.to_csv(output_path, index=False)
    logger.info(f"Built reference library with {len(ref_df)} approved small molecules at '{output_path}'.")
    return ref_df


def load_or_create_reference_cache(cache_path: str = REFERENCE_CACHE_PATH) -> pd.DataFrame:
    """Load cached reference library or build it on the fly if missing."""
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    return build_approved_reference_cache(output_path=cache_path)
