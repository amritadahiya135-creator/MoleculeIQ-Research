"""Typed application data contracts and DTOs (SRD Section 4.4)."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

from src.chemistry.validation import MoleculeValidationResult
from src.chemistry.descriptors import MolecularDescriptors
from src.models.inference import ClinToxPrediction, Tox21MultiPrediction
from src.similarity.index import SimilaritySearchResult
from src.ranking.scoring import PrioritizationScore
from src.explainability.shap_analysis import ExplanationResult


@dataclass(frozen=True)
class MoleculeAnalysisResult:
    """Comprehensive single-molecule evaluation container."""
    validation: MoleculeValidationResult
    descriptors: Optional[MolecularDescriptors]
    clintox: Optional[ClinToxPrediction]
    tox21: Optional[Tox21MultiPrediction]
    similarity: Optional[SimilaritySearchResult]
    prioritization: Optional[PrioritizationScore]
    explanation: Optional[ExplanationResult]
    analysis_timestamp: str
    is_valid: bool
    status: str
    error_message: Optional[str] = None


@dataclass(frozen=True)
class BatchCandidateRecord:
    """Row record within a batch screening job."""
    row_id: int
    name: str
    original_smiles: str
    canonical_smiles: Optional[str]
    is_valid: bool
    error_message: Optional[str]
    molecular_weight: Optional[float]
    logp: Optional[float]
    hbond_donors: Optional[int]
    hbond_acceptors: Optional[int]
    rotatable_bonds: Optional[int]
    tpsa: Optional[float]
    lipinski_pass: Optional[bool]
    clintox_p_tox: Optional[float]
    clintox_is_toxic: Optional[bool]
    tox21_active_count: Optional[int]
    top_similarity: Optional[float]
    top_reference_name: Optional[str]
    safety_score: Optional[float]
    druglikeness_score: Optional[float]
    similarity_score: Optional[float]
    overall_priority_score: Optional[float]
    applicability_domain_status: Optional[str] = None
    applicability_domain_score: Optional[float] = None
    applicability_warning: Optional[str] = None


@dataclass
class BatchScreeningResult:
    """Complete batch screening summary with fault-isolated records."""
    total_records: int
    valid_records: int
    invalid_records: int
    candidates: List[BatchCandidateRecord]
    results_dataframe: pd.DataFrame
    batch_timestamp: str
