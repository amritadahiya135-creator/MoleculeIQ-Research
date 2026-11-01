"""MoleculeIQ Landing Page & Research Architecture (Module M10)."""

import streamlit as st
from src.ui.brand import render_computational_network_svg


def render_overview():
    """Render the editorial MoleculeIQ product landing page in brand palette."""
    # Top Hero Composition (Editorial Layout)
    col_text, col_visual = st.columns([1.45, 1], gap="medium")

    with col_text:
        st.markdown(
            """
            <div style="padding-top: 0.5rem;">
                <div style="display: inline-block; padding: 0.2rem 0.65rem; background: rgba(39, 199, 184, 0.12); border: 1px solid rgba(39, 199, 184, 0.35); border-radius: 9999px; font-size: 0.72rem; font-weight: 700; color: #0F8075; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.85rem;">
                    AI-Assisted Molecular Intelligence
                </div>
                <h1 style="margin: 0 0 0.5rem 0; font-size: 2.65rem; font-weight: 800; color: #101B2D; letter-spacing: -0.03em; line-height: 1.15;">
                    From molecular structure to interpretable research insight.
                </h1>
                <p style="margin: 0 0 1.25rem 0; font-size: 1.02rem; color: #334155; line-height: 1.6; max-width: 620px;">
                    MoleculeIQ unifies calibrated clinical toxicity estimation, 12 in-vitro biological stress assays,
                    chemical reference neighborhood matching, and TreeSHAP atom-level explainability in an integrated research workspace.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def _nav_to_single():
            st.session_state["nav_mode"] = "Single Molecule Analysis"
            st.session_state["active_nav_selector"] = "◎ Analyze Molecule"

        def _nav_to_batch():
            st.session_state["nav_mode"] = "Batch Molecular Screening"
            st.session_state["active_nav_selector"] = "◫ Batch Screening"

        col_btn1, col_btn2 = st.columns([1, 1.2], gap="small")
        with col_btn1:
            st.button(
                "◎ Analyze a Molecule",
                type="primary",
                use_container_width=True,
                key="cta_analyze_single",
                on_click=_nav_to_single,
            )
        with col_btn2:
            st.button(
                "◫ Explore Batch Screening",
                use_container_width=True,
                key="cta_explore_batch",
                on_click=_nav_to_batch,
            )

        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.85rem;">
                Quick-start references: Aspirin, Caffeine, Atorvastatin, Metformin, Sorafenib.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_visual:
        st.markdown(
            f'<div class="mol-card-dark" style="padding: 1.25rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(16, 27, 45, 0.12);">{render_computational_network_svg(width=440, height=270)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # Core Research Capabilities (Editorial 3-Column Architecture)
    st.markdown("<h3 style='margin: 0 0 1.25rem 0; font-size: 1.25rem; color: #101B2D;'>Core Research Capabilities</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #0F8075; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
                    Predictive Toxicology
                </div>
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #101B2D;">Calibrated ClinTox & Tox21 Models</h4>
                <p style="font-size: 0.86rem; color: #475569; line-height: 1.55; margin: 0;">
                    Platt-calibrated Random Forest classification for clinical trial failure, paired with
                    12 independent Tox21 biological stress assays and strict applicability domain boundaries.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #0F8075; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
                    Cheminformatics & Conformation
                </div>
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #101B2D;">Chemical Space & Reference Similarity</h4>
                <p style="font-size: 0.86rem; color: #475569; line-height: 1.55; margin: 0;">
                    Interactive 3D molecular conformer visualization with MMFF94 force-field geometry, combined with
                    Tanimoto structural neighborhood matching against 1,396 FDA-approved reference drugs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #5B48DF; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
                    Explainability & Literature
                </div>
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #101B2D;">TreeSHAP & Semantic Retrieval</h4>
                <p style="font-size: 0.86rem; color: #475569; line-height: 1.55; margin: 0;">
                    Atom-level TreeSHAP feature attributions mapped from Morgan fingerprint bits onto 2D coordinates,
                    grounded by real-time NCBI PubMed literature search with concept-conjunction relevance ranking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # System & Engine Readiness Row
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; padding: 0.5rem 0; font-size: 0.82rem; color: #64748B;">
            <div>
                <strong style="color: #101B2D;">Engine Architecture:</strong>
                Scikit-Learn 1.9 &bull; RDKit (2D + 3D Conformer Engine) &bull; SHAP TreeExplainer &bull; NCBI E-utilities
            </div>
            <div style="display: flex; gap: 1.25rem;">
                <span><strong style="color: #065F46;">●</strong> ClinTox: 0.852 ROC-AUC</span>
                <span><strong style="color: #065F46;">●</strong> Tox21: 12 Classifiers</span>
                <span><strong style="color: #0F8075;">●</strong> ChEMBL: 1,396 References</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
