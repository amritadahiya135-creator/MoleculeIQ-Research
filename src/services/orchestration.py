"""Application service orchestrator coordinating scientific workflows (SRD Section 2.2)."""

import datetime
import logging
from typing import List, Optional, Callable, Dict, Any
import pandas as pd

from src.chemistry.validation import validate_smiles
from src.chemistry.descriptors import calculate_descriptors
from src.models.inference import ToxicityInferenceService
from src.similarity.index import ChemicalSimilarityIndex
from src.ranking.scoring import calculate_prioritization_score
from src.explainability.shap_analysis import ClinToxExplainer
from src.literature.retrieval import LiteratureAssistantService, LiteratureSearchResult
from src.services.contracts import (
    MoleculeAnalysisResult,
    BatchCandidateRecord,
    BatchScreeningResult,
)

logger = logging.getLogger(__name__)


class ScreeningOrchestrator:
    """
    Central application service managing single-molecule, batch screening,
    and literature intelligence workflows.
    """

    def __init__(
        self,
        inference_service: Optional[ToxicityInferenceService] = None,
        similarity_index: Optional[ChemicalSimilarityIndex] = None,
        explainer: Optional[ClinToxExplainer] = None,
        literature_service: Optional[LiteratureAssistantService] = None,
    ):
        self.inference_service = inference_service or ToxicityInferenceService()
        self.similarity_index = similarity_index or ChemicalSimilarityIndex()
        self.explainer = explainer or ClinToxExplainer()
        self.literature_service = literature_service or LiteratureAssistantService()

    def analyze_single_molecule(
        self,
        smiles: str,
        compute_explanation: bool = True,
    ) -> MoleculeAnalysisResult:
        """
        Execute Workflow A: Single Molecule Analysis across all available engines.

        Steps:
          1. Validation (M01)
          2. Physicochemical Descriptors (M02)
          3. ClinTox Prediction (M04)
          4. Tox21 12-Endpoint Predictions (M05)
          5. Chemical Similarity Search (M06)
          6. Composite Prioritization Score (M07)
          7. SHAP Feature Attribution & Highlight (M08)
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        val_res = validate_smiles(smiles)

        if not val_res.is_valid or not val_res.canonical_smiles:
            return MoleculeAnalysisResult(
                validation=val_res,
                descriptors=None,
                clintox=None,
                tox21=None,
                similarity=None,
                prioritization=None,
                explanation=None,
                analysis_timestamp=now_iso,
                is_valid=False,
                status="INVALID_MOLECULE",
                error_message=val_res.error_message,
            )

        canonical = val_res.canonical_smiles

        # 1. Physicochemical Descriptors
        try:
            desc = calculate_descriptors(canonical)
        except Exception as exc:
            logger.error(f"Descriptor calculation failed: {exc}")
            desc = None

        # 2. ClinTox Toxicity Prediction
        clintox_pred = self.inference_service.predict_clintox(canonical)

        # 3. Tox21 Multi-Endpoint Prediction
        tox21_pred = self.inference_service.predict_tox21(canonical)

        # 4. Chemical Similarity Search
        sim_res = self.similarity_index.search(canonical, top_k=5)

        # 5. Composite Prioritization Score
        p_tox = clintox_pred.probability if clintox_pred.status == "SUCCESS" else None
        lip_pass = desc.lipinski_pass if desc else None
        best_sim = sim_res.best_similarity if sim_res.status == "SUCCESS" else None

        prio_score = calculate_prioritization_score(
            p_tox=p_tox,
            lipinski_pass=lip_pass,
            best_similarity=best_sim,
        )

        # 6. SHAP Explainability
        explanation = None
        if compute_explanation:
            try:
                explanation = self.explainer.explain_molecule(canonical, top_k_bits=5)
            except Exception as exc:
                logger.error(f"SHAP explanation failed: {exc}")
                explanation = None

        return MoleculeAnalysisResult(
            validation=val_res,
            descriptors=desc,
            clintox=clintox_pred,
            tox21=tox21_pred,
            similarity=sim_res,
            prioritization=prio_score,
            explanation=explanation,
            analysis_timestamp=now_iso,
            is_valid=True,
            status="SUCCESS",
            error_message=None,
        )

    def screen_batch(
        self,
        smiles_list: List[str],
        names_list: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchScreeningResult:
        """
        Execute Workflow B: Batch Molecular Screening with strict row-level fault isolation.

        Processes valid rows while preserving error reasons for invalid rows.
        Ranks candidates by composite priority score.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        total = len(smiles_list)
        if total > 5000:
            raise ValueError(
                f"Batch size ({total}) exceeds maximum supported limit of 5,000 molecules. "
                "Please partition the library into smaller sub-batches."
            )

        records: List[BatchCandidateRecord] = []
        valid_count = 0
        invalid_count = 0

        for idx, smi in enumerate(smiles_list):
            name = names_list[idx] if (names_list and idx < len(names_list)) else f"Compound_{idx+1}"

            # Step 1: Validate row
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
                    )
                )
                if progress_callback:
                    progress_callback(idx + 1, total)
                continue

            # Valid row
            canonical = val_res.canonical_smiles
            valid_count += 1

            # Descriptors
            try:
                desc = calculate_descriptors(canonical)
            except Exception:
                desc = None

            # Models
            clintox = self.inference_service.predict_clintox(canonical)
            tox21 = self.inference_service.predict_tox21(canonical)

            # Similarity
            sim = self.similarity_index.search(canonical, top_k=1)
            top_sim = sim.best_similarity if sim.status == "SUCCESS" else 0.0
            top_ref_name = sim.matches[0].name if sim.matches else "None"

            # Prioritization Score
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
                )
            )

            if progress_callback:
                progress_callback(idx + 1, total)

        # Convert to DataFrame
        rows_data = []
        for r in records:
            rows_data.append({
                "Rank": None,  # Will be populated
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
                "Tox21_Active_Assays": r.tox21_active_count,
                "Top_Reference_Drug": r.top_reference_name,
                "Tanimoto_Similarity": r.top_similarity,
                "Safety_Score": r.safety_score,
                "Druglikeness_Score": r.druglikeness_score,
                "Similarity_Score": r.similarity_score,
                "Overall_Priority_Score": r.overall_priority_score,
            })

        df_res = pd.DataFrame(rows_data)

        # Sort valid candidates descending by overall priority score, keep invalid at bottom
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
            batch_timestamp=now_iso,
        )

    def search_literature(self, query: str, max_articles: int = 5) -> LiteratureSearchResult:
        """Execute Workflow D: Literature Intelligence Search and Ranking."""
        return self.literature_service.search_and_rank(query, max_articles=max_articles)
