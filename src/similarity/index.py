"""Chemical similarity search engine utilizing Tanimoto coefficient on 2048-bit Morgan fingerprints (Module M06)."""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from src.chemistry.validation import validate_smiles
from src.chemistry.fingerprints import generate_morgan_fingerprint, FingerprintResult
from src.similarity.chembl_cache import load_or_create_reference_cache


@dataclass(frozen=True)
class SimilarityMatch:
    """Individual structural similarity hit against reference compound."""
    reference_id: str
    name: str
    canonical_smiles: str
    similarity: float
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    lipinski_pass: Optional[bool] = None


@dataclass(frozen=True)
class SimilaritySearchResult:
    """Aggregate similarity search result."""
    query_smiles: str
    matches: List[SimilarityMatch]
    best_similarity: float
    total_references: int
    status: str
    error_message: Optional[str] = None


def compute_tanimoto_similarity(
    fp1: Union[ExplicitBitVect, np.ndarray, str],
    fp2: Union[ExplicitBitVect, np.ndarray, str],
) -> float:
    """
    Calculate exact Tanimoto similarity coefficient between two molecular structures.

    Formula:
        T(A, B) = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    fp1, fp2 : ExplicitBitVect, np.ndarray, or SMILES string.

    Returns
    -------
    float
        Tanimoto similarity in range [0.0, 1.0].
    """
    # Convert to ExplicitBitVect if needed
    def _to_bitvect(item) -> ExplicitBitVect:
        if isinstance(item, ExplicitBitVect):
            return item
        if isinstance(item, str):
            res = generate_morgan_fingerprint(item, radius=2, n_bits=2048)
            return res.bit_vector
        if isinstance(item, np.ndarray):
            bv = ExplicitBitVect(len(item))
            for idx in np.where(item > 0)[0]:
                bv.SetBit(int(idx))
            return bv
        raise TypeError(f"Unsupported fingerprint type: {type(item)}")

    bv1 = _to_bitvect(fp1)
    bv2 = _to_bitvect(fp2)

    return float(DataStructs.TanimotoSimilarity(bv1, bv2))


class ChemicalSimilarityIndex:
    """
    High-performance in-memory search index for structural similarity against reference libraries.
    """

    def __init__(self, reference_df: Optional[pd.DataFrame] = None):
        if reference_df is None:
            reference_df = load_or_create_reference_cache()

        self.df = reference_df.copy()
        self._ref_fps: List[ExplicitBitVect] = []
        self._valid_indices: List[int] = []

        # Precompute fingerprints for all reference compounds
        for idx, row in self.df.iterrows():
            smi = str(row["canonical_smiles"])
            try:
                fp_res = generate_morgan_fingerprint(smi, radius=2, n_bits=2048, return_bit_info=False)
                self._ref_fps.append(fp_res.bit_vector)
                self._valid_indices.append(idx)
            except Exception:
                continue

    @property
    def total_compounds(self) -> int:
        return len(self._ref_fps)

    def search(
        self,
        query: Union[str, ExplicitBitVect, FingerprintResult],
        top_k: int = 5,
    ) -> SimilaritySearchResult:
        """
        Retrieve the top-K most similar reference small molecules.

        Parameters
        ----------
        query : str, ExplicitBitVect, or FingerprintResult
            Query molecule SMILES or precomputed fingerprint.
        top_k : int, default=5
            Number of top candidates to return.

        Returns
        -------
        SimilaritySearchResult
            Ranked list of reference matches and best similarity score.
        """
        if self.total_compounds == 0:
            return SimilaritySearchResult(
                query_smiles=str(query),
                matches=[],
                best_similarity=0.0,
                total_references=0,
                status="EMPTY_REFERENCE_INDEX",
                error_message="Reference library index contains no valid molecules.",
            )

        # Obtain query ExplicitBitVect
        query_smiles = ""
        if isinstance(query, str):
            val_res = validate_smiles(query)
            if not val_res.is_valid or not val_res.canonical_smiles:
                return SimilaritySearchResult(
                    query_smiles=query,
                    matches=[],
                    best_similarity=0.0,
                    total_references=self.total_compounds,
                    status="INVALID_INPUT",
                    error_message=val_res.error_message,
                )
            query_smiles = val_res.canonical_smiles
            query_fp = generate_morgan_fingerprint(val_res.canonical_smiles, radius=2, n_bits=2048).bit_vector
        elif isinstance(query, FingerprintResult):
            query_fp = query.bit_vector
        elif isinstance(query, ExplicitBitVect):
            query_fp = query
        else:
            return SimilaritySearchResult(
                query_smiles="",
                matches=[],
                best_similarity=0.0,
                total_references=self.total_compounds,
                status="INVALID_INPUT",
                error_message=f"Unsupported query type {type(query).__name__}",
            )

        # BulkTanimotoSimilarity is executed in optimized C++
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, self._ref_fps)

        # Sort indices in descending similarity order
        sorted_indices = np.argsort(sims)[::-1][:top_k]

        matches: List[SimilarityMatch] = []
        for idx in sorted_indices:
            orig_row_idx = self._valid_indices[idx]
            row = self.df.loc[orig_row_idx]
            matches.append(
                SimilarityMatch(
                    reference_id=str(row.get("reference_id", f"REF_{idx}")),
                    name=str(row.get("name", "Unknown")),
                    canonical_smiles=str(row.get("canonical_smiles", "")),
                    similarity=round(float(sims[idx]), 4),
                    formula=str(row.get("formula", "")),
                    molecular_weight=float(row.get("molecular_weight", 0.0)),
                    lipinski_pass=bool(row.get("lipinski_pass", True)),
                )
            )

        best_sim = float(matches[0].similarity) if matches else 0.0

        return SimilaritySearchResult(
            query_smiles=query_smiles,
            matches=matches,
            best_similarity=best_sim,
            total_references=self.total_compounds,
            status="SUCCESS",
            error_message=None,
        )
