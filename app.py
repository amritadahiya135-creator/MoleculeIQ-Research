"""MoleculeIQ — Main Application Entrypoint (Module M10)."""

import os
from typing import Tuple
import streamlit as st

# Must be the very first Streamlit call
st.set_page_config(
    page_title="MoleculeIQ — Molecular Observatory",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from scripts.download_release_artifacts import ensure_artifacts_hydrated
from src.services.orchestration import ScreeningOrchestrator
from src.ui.brand import render_brand_header
from src.ui.theme import apply_theme
from src.ui.components import render_scientific_boundary_banner
from src.ui.landing_page import render_landing_page
from src.ui.overview import render_overview
from src.ui.single_molecule import render_single_molecule_view
from src.ui.batch_screening import render_batch_screening_view
from src.ui.model_insights import render_model_insights_view
from src.ui.literature_assistant import render_literature_assistant_view
from src.ui.reporting import render_reporting_studio_view
from src.ui.settings import render_settings_view
from src.ui.about import render_about_view

# Navigation Definitions & Mapping
MODE_MAP = {
    # Internal key -> Display string
    "Molecular Observatory": "🌌 Molecular Observatory",
    "Overview Dashboard": "⌂ Overview Dashboard",
    "Overview & Principles": "⌂ Overview Dashboard",  # backward compatibility
    "Single Molecule Analysis": "◎ Single Molecule Analysis",
    "Batch Molecular Screening": "◫ Batch Screening",
    "Literature Intelligence": "⌕ Literature Research",
    "Model Insights & Metrics": "◉ Model Insights",
    "Export & Reporting": "📋 Export & Reporting",
    "Settings & Telemetry": "⚙ Settings & Telemetry",
    "About & Architecture": "ℹ About & Architecture",
}

DISPLAY_TO_INTERNAL = {
    "🌌 Molecular Observatory": "Molecular Observatory",
    "⌂ Overview Dashboard": "Overview Dashboard",
    "◎ Single Molecule Analysis": "Single Molecule Analysis",
    "◫ Batch Screening": "Batch Molecular Screening",
    "⌕ Literature Research": "Literature Intelligence",
    "◉ Model Insights": "Model Insights & Metrics",
    "📋 Export & Reporting": "Export & Reporting",
    "⚙ Settings & Telemetry": "Settings & Telemetry",
    "ℹ About & Architecture": "About & Architecture",
}

NAV_DISPLAY_LIST = [
    "🌌 Molecular Observatory",
    "⌂ Overview Dashboard",
    "◎ Single Molecule Analysis",
    "◫ Batch Screening",
    "⌕ Literature Research",
    "◉ Model Insights",
    "📋 Export & Reporting",
    "⚙ Settings & Telemetry",
    "ℹ About & Architecture",
]


@st.cache_resource(show_spinner="Verifying and downloading MoleculeIQ ML models (v5.0.0)... This occurs once on initial startup.")
def bootstrap_model_artifacts() -> Tuple[bool, str]:
    """Verify local model artifacts, hydrating from GitHub Releases if missing."""
    return ensure_artifacts_hydrated()


@st.cache_resource
def get_orchestrator() -> ScreeningOrchestrator:
    """Initialize and cache the screening orchestrator across application sessions."""
    return ScreeningOrchestrator()


def main():
    # Apply custom design system CSS and typography
    apply_theme()

    # Ensure model artifacts are available or downloaded from published v5.0.0 release
    models_ready, hydration_msg = bootstrap_model_artifacts()

    orchestrator = get_orchestrator()

    # Sidebar Header with Original Geometric Brand Identity
    st.sidebar.markdown(
        f'<div style="padding: 0.6rem 0 0.85rem 0; border-bottom: 1px solid #142236; margin-bottom: 0.85rem;">{render_brand_header(width=230)}</div>',
        unsafe_allow_html=True,
    )

    if "nav_mode" not in st.session_state:
        st.session_state["nav_mode"] = "Molecular Observatory"
        st.session_state["active_nav_selector"] = "🌌 Molecular Observatory"
        st.session_state["_last_nav_mode"] = "Molecular Observatory"

    # Navigation callback for sidebar radio
    def on_nav_change():
        selected = st.session_state.get("active_nav_selector")
        if selected in DISPLAY_TO_INTERNAL:
            internal = DISPLAY_TO_INTERNAL[selected]
            st.session_state["nav_mode"] = internal
            st.session_state["_last_nav_mode"] = internal

    # Sync external nav_mode changes (from page buttons) back to radio selector
    if st.session_state.get("_last_nav_mode") != st.session_state.get("nav_mode"):
        desired_display = MODE_MAP.get(st.session_state["nav_mode"])
        if desired_display:
            st.session_state["active_nav_selector"] = desired_display
        st.session_state["_last_nav_mode"] = st.session_state["nav_mode"]

    st.sidebar.markdown("""<div class="sidebar-category">OBSERVATORY WORKSPACES</div>""", unsafe_allow_html=True)

    selected_display = st.sidebar.radio(
        "Application Workspace:",
        NAV_DISPLAY_LIST,
        key="active_nav_selector",
        on_change=on_nav_change,
        label_visibility="collapsed",
    )

    # Update session state with selected mode
    active_internal_mode = DISPLAY_TO_INTERNAL.get(selected_display, "Molecular Observatory")
    st.session_state["nav_mode"] = active_internal_mode
    st.session_state["_last_nav_mode"] = active_internal_mode

    # Sidebar System Health & Engine Readiness
    chembl_count = orchestrator.similarity_index.total_compounds
    clintox_label = "● 0.852 AUC" if models_ready else "● Unavailable"
    clintox_color = "#39C6D8" if models_ready else "#64748B"
    tox21_label = "● 12 Models" if models_ready else "● Unavailable"
    tox21_color = "#39C6D8" if models_ready else "#64748B"

    st.sidebar.markdown("""<div class="sidebar-category">SYSTEM READINESS</div>""", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div style="background: #0B1220; border: 1px solid #1E314B; border-radius: 8px; padding: 0.75rem 0.85rem;">
            <div style="font-size: 0.76rem; color: #CBD5E1; display: flex; flex-direction: column; gap: 0.35rem;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #AAB7C5;">ClinTox Random Forest</span>
                    <span style="color: {clintox_color}; font-weight: 600;">{clintox_label}</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #AAB7C5;">Tox21 Assay Suite</span>
                    <span style="color: {tox21_color}; font-weight: 600;">{tox21_label}</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #AAB7C5;">Approved Drug Cache</span>
                    <span style="color: #8E7CFF; font-weight: 600;">● {chembl_count:,} Refs</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #AAB7C5;">NCBI PubMed Retrieval</span>
                    <span style="color: {clintox_color}; font-weight: 600;">● Live E-Utils</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not models_ready:
        st.sidebar.warning("⚠️ ML models unavailable. Using heuristic/literature mode.")
        if st.sidebar.button("🔄 Retry Model Hydration", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    st.sidebar.markdown(
        """
        <div style="padding-top: 1.25rem; font-size: 0.7rem; color: #AAB7C5; line-height: 1.4;">
            MoleculeIQ is an academic in silico screening platform. Results do not constitute clinical diagnostics or regulatory clearance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mandatory Scientific Boundary Banner
    render_scientific_boundary_banner()

    if not models_ready:
        st.warning(
            f"⚠️ **Model Artifact Notice:** Machine learning model weights could not be loaded or verified "
            f"({hydration_msg}). Physicochemical descriptors, Lipinski profiling, ChEMBL reference similarity, and "
            f"NCBI PubMed literature search remain operational, but clinical toxicity and Tox21 assay predictions are unavailable."
        )

    # Route Workspace
    if active_internal_mode == "Molecular Observatory":
        render_landing_page()
    elif active_internal_mode in ("Overview Dashboard", "Overview & Principles"):
        render_overview()
    elif active_internal_mode == "Single Molecule Analysis":
        render_single_molecule_view(orchestrator)
    elif active_internal_mode == "Batch Molecular Screening":
        render_batch_screening_view(orchestrator)
    elif active_internal_mode == "Literature Intelligence":
        render_literature_assistant_view(orchestrator)
    elif active_internal_mode == "Model Insights & Metrics":
        render_model_insights_view()
    elif active_internal_mode == "Export & Reporting":
        render_reporting_studio_view()
    elif active_internal_mode == "Settings & Telemetry":
        render_settings_view()
    elif active_internal_mode == "About & Architecture":
        render_about_view()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()
