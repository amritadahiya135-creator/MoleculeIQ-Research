"""MoleculeIQ — Main Application Entrypoint (Module M10)."""

import os
import streamlit as st

# Must be the very first Streamlit call
st.set_page_config(
    page_title="MoleculeIQ — Molecular Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import importlib
import src.ui.components
import src.ui.overview
import src.ui.single_molecule
import src.ui.batch_screening
import src.ui.model_insights
import src.ui.literature_assistant

# Ensure UI modules are consistently fresh across long-running server sessions
importlib.reload(src.ui.components)
importlib.reload(src.ui.overview)
importlib.reload(src.ui.single_molecule)
importlib.reload(src.ui.batch_screening)
importlib.reload(src.ui.model_insights)
importlib.reload(src.ui.literature_assistant)

from src.services.orchestration import ScreeningOrchestrator
from src.ui.brand import render_brand_header
from src.ui.theme import apply_theme
from src.ui.components import render_scientific_boundary_banner
from src.ui.overview import render_overview
from src.ui.single_molecule import render_single_molecule_view
from src.ui.batch_screening import render_batch_screening_view
from src.ui.model_insights import render_model_insights_view
from src.ui.literature_assistant import render_literature_assistant_view

# Navigation Definitions & Mapping
MODE_MAP = {
    "Overview & Principles": "⌂ Overview & Architecture",
    "Single Molecule Analysis": "◎ Analyze Molecule",
    "Batch Molecular Screening": "◫ Batch Screening",
    "Literature Intelligence": "⌕ Literature Intelligence",
    "Model Insights & Metrics": "◉ Model Insights & Benchmarks",
}
DISPLAY_TO_INTERNAL = {v: k for k, v in MODE_MAP.items()}
NAV_DISPLAY_LIST = list(MODE_MAP.values())


@st.cache_resource
def get_orchestrator() -> ScreeningOrchestrator:
    """Initialize and cache the screening orchestrator across application sessions."""
    return ScreeningOrchestrator()


def main():
    # Apply custom design system CSS and typography
    apply_theme()

    orchestrator = get_orchestrator()

    # Sidebar Header with Original Geometric Brand Identity
    st.sidebar.markdown(
        f'<div style="padding: 0.6rem 0 0.85rem 0; border-bottom: 1px solid #1C2B40; margin-bottom: 0.85rem;">{render_brand_header(width=215)}</div>',
        unsafe_allow_html=True,
    )

    # Navigation callback for sidebar radio
    def on_nav_change():
        selected = st.session_state.get("active_nav_selector")
        if selected in DISPLAY_TO_INTERNAL:
            st.session_state["nav_mode"] = DISPLAY_TO_INTERNAL[selected]

    if "active_nav_selector" not in st.session_state:
        initial_mode = st.session_state.get("nav_mode", "Overview & Principles")
        st.session_state["active_nav_selector"] = MODE_MAP.get(initial_mode, NAV_DISPLAY_LIST[0])

    st.sidebar.markdown("""<div class="sidebar-category">RESEARCH WORKSPACES</div>""", unsafe_allow_html=True)

    selected_display = st.sidebar.radio(
        "Application Workspace:",
        NAV_DISPLAY_LIST,
        key="active_nav_selector",
        on_change=on_nav_change,
        label_visibility="collapsed",
    )

    # Update session state with selected mode
    active_internal_mode = DISPLAY_TO_INTERNAL.get(selected_display, "Overview & Principles")
    st.session_state["nav_mode"] = active_internal_mode

    # Sidebar System Health & Engine Readiness
    chembl_count = orchestrator.similarity_index.total_compounds

    st.sidebar.markdown("""<div class="sidebar-category">SYSTEM READINESS</div>""", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div style="background: #1C2B40; border: 1px solid #2A3B52; border-radius: 8px; padding: 0.75rem 0.85rem;">
            <div style="font-size: 0.76rem; color: #CBD5E1; display: flex; flex-direction: column; gap: 0.35rem;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8;">ClinTox Random Forest</span>
                    <span style="color: #27C7B8; font-weight: 600;">● 0.852 AUC</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8;">Tox21 Assay Suite</span>
                    <span style="color: #27C7B8; font-weight: 600;">● 12 Models</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8;">Approved Drug Cache</span>
                    <span style="color: #8E7CFF; font-weight: 600;">● {chembl_count} Refs</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8;">NCBI PubMed Retrieval</span>
                    <span style="color: #27C7B8; font-weight: 600;">● Live E-Utils</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="padding-top: 1.25rem; font-size: 0.7rem; color: #94A3B8; line-height: 1.4;">
            MoleculeIQ is an academic in silico screening platform. Results do not constitute clinical diagnostics or regulatory clearance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mandatory Scientific Boundary Banner
    render_scientific_boundary_banner()

    # Route Workspace
    if active_internal_mode == "Overview & Principles":
        render_overview()
    elif active_internal_mode == "Single Molecule Analysis":
        render_single_molecule_view(orchestrator)
    elif active_internal_mode == "Batch Molecular Screening":
        render_batch_screening_view(orchestrator)
    elif active_internal_mode == "Model Insights & Metrics":
        render_model_insights_view()
    elif active_internal_mode == "Literature Intelligence":
        render_literature_assistant_view(orchestrator)


if __name__ == "__main__":
    main()
