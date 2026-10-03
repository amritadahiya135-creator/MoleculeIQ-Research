"""MoleculeIQ — Main Application Entrypoint (Module M10)."""

import os
import streamlit as st

# Must be the very first Streamlit call
st.set_page_config(
    page_title="MoleculeIQ — Molecular Screening & Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.services.orchestration import ScreeningOrchestrator
from src.ui.components import render_scientific_boundary_banner
from src.ui.single_molecule import render_single_molecule_view
from src.ui.batch_screening import render_batch_screening_view
from src.ui.model_insights import render_model_insights_view
from src.ui.literature_assistant import render_literature_assistant_view


@st.cache_resource
def get_orchestrator() -> ScreeningOrchestrator:
    """Initialize and cache the screening orchestrator across application sessions."""
    return ScreeningOrchestrator()


def render_overview():
    """Render Platform Overview & Scientific Architecture."""
    st.markdown(
        """
        # MoleculeIQ: Integrated Molecular Screening & Literature Intelligence

        **MoleculeIQ** is a research-grade, transparent computational platform that unifies 
        cheminformatics property calculation, benchmark machine learning toxicity estimation, 
        structural similarity search, explainable AI, and scientific literature retrieval into an integrated workspace.

        ---

        ### Core Scientific Workflows
        
        1. **Single Molecule Analysis:** Full profiling of chemical candidates from SMILES notation, including RDKit 2D vector depiction, Lipinski Rule of 5 validation, ClinTox clinical trial toxicity probability, Tox21 12-pathway profiling, approved ChEMBL drug similarity, composite prioritization scoring, and SHAP TreeExplainer substructure attribution.
        2. **Batch Molecular Screening:** High-throughput triage of compound libraries (CSV) with row-level fault isolation, automated ranking by multi-objective prioritization score ($R = 0.5S + 0.3D + 0.2M$), and comprehensive enriched dataset export.
        3. **Model Insights & Evaluation Evidence:** Auditable evaluation metrics on held-out benchmark test sets (ROC-AUC, PR-AUC, Balanced Accuracy, confusion matrices) and explicit class distribution documentation.
        4. **Literature Intelligence Assistant:** Semantic literature discovery querying live NCBI PubMed records, parsing abstracts into distinct scientific sentences, and ranking by cosine relevance with direct source links.

        ---

        ### Platform Governance & Boundaries
        - **In Silico Triage:** MoleculeIQ is intended for academic research, education, and early computational prioritization.
        - **Non-Diagnostic:** Predictions do not constitute clinical safety, regulatory toxicology certification, or guarantees of drug efficacy.
        - **Empirical Superiority:** Computational estimates must never replace empirical biological assays.
        """
    )


def main():
    orchestrator = get_orchestrator()

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
    st.sidebar.title("MoleculeIQ")
    st.sidebar.caption("v1.0.0 | Academic & Research Platform")

    mode = st.sidebar.radio(
        "Application Mode:",
        [
            "Overview & Principles",
            "Single Molecule Analysis",
            "Batch Molecular Screening",
            "Model Insights & Metrics",
            "Literature Intelligence",
        ],
        index=1,
    )

    st.sidebar.divider()
    st.sidebar.markdown("**System Health & Status**")
    st.sidebar.success("ClinTox Model: Ready (ROC-AUC: 0.852)")
    st.sidebar.success("Tox21 Suite: Ready (12 Endpoints)")
    st.sidebar.success(f"Reference Cache: {orchestrator.similarity_index.total_compounds} Approved Drugs")
    st.sidebar.info("Literature API: NCBI E-utilities (Live)")

    # Mandatory Scientific Boundary Disclaimer Banner
    render_scientific_boundary_banner()

    if mode == "Overview & Principles":
        render_overview()
    elif mode == "Single Molecule Analysis":
        render_single_molecule_view(orchestrator)
    elif mode == "Batch Molecular Screening":
        render_batch_screening_view(orchestrator)
    elif mode == "Model Insights & Metrics":
        render_model_insights_view()
    elif mode == "Literature Intelligence":
        render_literature_assistant_view(orchestrator)


if __name__ == "__main__":
    main()
