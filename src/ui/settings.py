"""MoleculeIQ — System Settings & Platform Telemetry (Module M10)."""

import sys
import streamlit as st
import rdkit
import sklearn
from src.ui.components import render_header, render_metric_card


def render_settings_view():
    """Render the System Settings & Platform Telemetry view."""
    render_header(
        "System Settings & Telemetry",
        "Platform health diagnostics, runtime environment specifications, and cache lifecycle controls.",
        badge_text="System Diagnostics",
        badge_variant="info",
    )

    st.markdown("### Runtime & Environment Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Python Runtime", sys.version.split()[0], "CPython 64-bit", badge="Core", badge_variant="info")
    with col2:
        render_metric_card("RDKit Version", rdkit.__version__, "Cheminformatics Engine", badge="C++", badge_variant="teal")
    with col3:
        render_metric_card("Scikit-Learn", sklearn.__version__, "Ensemble Models", badge="ML", badge_variant="purple")
    with col4:
        render_metric_card("Streamlit", st.__version__, "Frontend UI Engine", badge="UI", badge_variant="info")

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    st.markdown("### Cache & Storage Lifecycle Controls")
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        if st.button("🔄 Clear Analysis Cache", use_container_width=True):
            st.session_state.pop("single_mol_result", None)
            st.session_state.pop("single_mol_analyzed_smiles", None)
            st.success("✓ Single molecule analysis cache cleared.")
    with c_btn2:
        if st.button("🗑️ Reset Batch Memory", use_container_width=True):
            st.session_state.pop("batch_input_df", None)
            st.session_state.pop("batch_screening_result", None)
            st.success("✓ Batch memory reset.")
    with c_btn3:
        if st.button("⚡ Reload Model Artifacts", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    st.markdown("### Platform Architecture Disclosures")
    st.markdown(
        """
        <div class="mol-card" style="padding: 1.15rem; font-size: 0.86rem; color: #CBD5E1; line-height: 1.6;">
            <div>• <strong>Theme Enforcement:</strong> Strict dark theme (Observatory Black <code>#080D16</code>, Midnight Navy <code>#0B1220</code>, Scientific Teal <code>#4F9D96</code>).</div>
            <div>• <strong>Scientific Boundaries:</strong> In silico computational screening for research use only. Predictions do not establish clinical safety or regulatory efficacy.</div>
            <div>• <strong>Model Artifacts:</strong> Verified via SHA-256 against release manifest (v5.0.0). Hydrated automatically on cold start.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
