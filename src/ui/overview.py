"""MoleculeIQ — Command Centre Overview Dashboard (Module M10).

Executive research command centre providing telemetry, quick workspace actions,
model status, quick compound launchers, and architecture disclosures.
"""

import streamlit as st
from src.ui.brand import render_computational_network_svg


def render_overview():
    """Render the MoleculeIQ Command Centre Overview Dashboard in Observatory dark palette."""

    # Navigation callbacks
    def _go_to(mode: str, display: str, smiles: str = None):
        st.session_state["nav_mode"] = mode
        st.session_state["active_nav_selector"] = display
        if smiles:
            st.session_state["single_smiles_input"] = smiles

    # =========================================================================
    # 1. COMMAND CENTRE WELCOME & STATUS HERO
    # =========================================================================
    col_text, col_visual = st.columns([1.45, 1.0], gap="large")

    with col_text:
        st.markdown(
            """
            <div style="padding-top: 0.25rem;">
                <div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.2rem 0.65rem; background: rgba(79, 157, 150, 0.15); border: 1px solid rgba(79, 157, 150, 0.35); border-radius: 9999px; font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.85rem; font-family: 'IBM Plex Mono', monospace;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #57B88B; box-shadow: 0 0 8px #57B88B;"></span>
                    RESEARCH COMMAND CENTRE &bull; READY
                </div>
                <h1 style="margin: 0 0 0.5rem 0; font-size: 2.35rem; font-weight: 800; color: #F3F6FA; letter-spacing: -0.025em; line-height: 1.15; font-family: 'Manrope', sans-serif;">
                    MoleculeIQ Research Observatory
                </h1>
                <p style="margin: 0 0 1.25rem 0; font-size: 0.96rem; color: #AAB7C5; line-height: 1.6; max-width: 620px;">
                    Central intelligence command centre unifying calibrated clinical toxicity estimation,
                    12 in-vitro biological stress assays, chemical reference neighborhood matching, and
                    TreeSHAP atom-level explainability across active research workflows.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_btn1, col_btn2 = st.columns([1, 1.2], gap="small")
        with col_btn1:
            st.button(
                "◎ Analyze a Molecule",
                type="primary",
                use_container_width=True,
                key="ov_btn_single",
                on_click=lambda: _go_to("Single Molecule Analysis", "◎ Single Molecule Analysis"),
            )
        with col_btn2:
            st.button(
                "◫ Explore Batch Screening",
                use_container_width=True,
                key="ov_btn_batch",
                on_click=lambda: _go_to("Batch Molecular Screening", "◫ Batch Screening"),
            )

        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #8A99AD; margin-top: 0.85rem;">
                Active references: Aspirin, Caffeine, Paracetamol, Ibuprofen, Nicotine, Metformin.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_visual:
        st.markdown(
            f"""
            <div class="mol-card-dark" style="padding: 1.25rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);">
                {render_computational_network_svg(width=420, height=260)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 2. LIVE RESEARCH TELEMETRY (4 Cards)
    # =========================================================================
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                PLATFORM TELEMETRY & ENGINE READINESS
            </div>
            <div style="font-size: 0.74rem; color: #57B88B; font-weight: 600;">
                ● All Engines Hydrated
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.columns(4, gap="small")

    with t1:
        st.markdown(
            """
            <div class="mol-card" style="padding: 0.85rem 1rem;">
                <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">ClinTox Random Forest</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #4F9D96; margin: 0.2rem 0;">0.852 <span style="font-size: 0.75rem; color: #8A99AD;">ROC-AUC</span></div>
                <div style="font-size: 0.75rem; color: #AAB7C5;">Platt-Calibrated Posterior</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown(
            """
            <div class="mol-card" style="padding: 0.85rem 1rem;">
                <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Tox21 Assay Suite</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #39C6D8; margin: 0.2rem 0;">12 <span style="font-size: 0.75rem; color: #8A99AD;">Classifiers</span></div>
                <div style="font-size: 0.75rem; color: #AAB7C5;">Mean ROC-AUC: 0.767</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t3:
        st.markdown(
            """
            <div class="mol-card" style="padding: 0.85rem 1rem;">
                <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Approved Drug Cache</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #72A9E8; margin: 0.2rem 0;">1,396 <span style="font-size: 0.75rem; color: #8A99AD;">Drugs</span></div>
                <div style="font-size: 0.75rem; color: #AAB7C5;">ChEMBL Reference Slices</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t4:
        st.markdown(
            """
            <div class="mol-card" style="padding: 0.85rem 1rem;">
                <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Bibliographic Search</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #D9A441; margin: 0.2rem 0;">Live <span style="font-size: 0.75rem; color: #8A99AD;">E-Utils</span></div>
                <div style="font-size: 0.75rem; color: #AAB7C5;">NCBI PubMed Integration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. QUICK ACTION SHORTCUTS (5 Tiles)
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 0.85rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                WORKSPACE SHORTCUTS
            </div>
            <h3 style="margin: 0.2rem 0 0.5rem 0; font-size: 1.35rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Launch a Research Investigation
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4, q5 = st.columns(5, gap="small")
    with q1:
        st.button("◎ Single Molecule", use_container_width=True, key="sc_single", on_click=lambda: _go_to("Single Molecule Analysis", "◎ Single Molecule Analysis"))
    with q2:
        st.button("◫ Batch Screening", use_container_width=True, key="sc_batch", on_click=lambda: _go_to("Batch Molecular Screening", "◫ Batch Screening"))
    with q3:
        st.button("⌕ Search Literature", use_container_width=True, key="sc_lit", on_click=lambda: _go_to("Literature Intelligence", "⌕ Literature Research"))
    with q4:
        st.button("◉ Model Insights", use_container_width=True, key="sc_ins", on_click=lambda: _go_to("Model Insights & Metrics", "◉ Model Insights"))
    with q5:
        st.button("📋 Generate Report", use_container_width=True, key="sc_rep", on_click=lambda: _go_to("Export & Reporting", "📋 Export & Reporting"))

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 4. CORE RESEARCH CAPABILITIES (Mandatory: Chemical Space & Reference Similarity)
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 0.85rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                SCIENTIFIC FOUNDATIONS
            </div>
            <h3 style="margin: 0.2rem 0 0.75rem 0; font-size: 1.35rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Core Research Capabilities
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #4F9D96; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
                    Predictive Toxicology
                </div>
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #F3F6FA;">Calibrated ClinTox & Tox21 Models</h4>
                <p style="font-size: 0.84rem; color: #AAB7C5; line-height: 1.55; margin: 0;">
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
                <div style="font-size: 0.74rem; font-weight: 700; color: #39C6D8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
                    Cheminformatics & Conformation
                </div>
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #F3F6FA;">Chemical Space & Reference Similarity</h4>
                <p style="font-size: 0.84rem; color: #AAB7C5; line-height: 1.55; margin: 0;">
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
                <div style="font-size: 0.74rem; font-weight: 700; color: #72A9E8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
                    Explainability & Literature
                </div>
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #F3F6FA;">TreeSHAP & Semantic Retrieval</h4>
                <p style="font-size: 0.84rem; color: #AAB7C5; line-height: 1.55; margin: 0;">
                    Atom-level TreeSHAP feature attributions mapped from Morgan fingerprint bits onto 2D coordinates,
                    grounded by real-time NCBI PubMed literature search with concept-conjunction relevance ranking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 5. QUICK MOLECULE LAUNCHER (6 Pills)
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 0.85rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                QUICK COMPOUND LAUNCHER
            </div>
            <h3 style="margin: 0.2rem 0 0.4rem 0; font-size: 1.25rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Load Reference Molecule Directly
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mol_pills = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Nicotine", "CN1CCC[C@H]1c2cccnc2"),
        ("Metformin", "CN(C)C(=N)N=C(N)N"),
    ]

    p_cols = st.columns(6, gap="small")
    for col, (name, smiles) in zip(p_cols, mol_pills):
        with col:
            st.button(
                f"◎ {name}",
                key=f"ov_pill_{name}",
                use_container_width=True,
                on_click=lambda s=smiles: _go_to("Single Molecule Analysis", "◎ Single Molecule Analysis", s),
            )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 6. ENGINE ARCHITECTURE FOOTER
    # =========================================================================
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; padding: 0.5rem 0; font-size: 0.8rem; color: #8A99AD;">
            <div>
                <strong style="color: #F3F6FA;">Engine Architecture:</strong>
                Scikit-Learn 1.9 &bull; RDKit (2D + 3D Conformer Engine) &bull; SHAP TreeExplainer &bull; NCBI E-utilities
            </div>
            <div style="display: flex; gap: 1.25rem;">
                <span><strong style="color: #57B88B;">●</strong> ClinTox: 0.852 ROC-AUC</span>
                <span><strong style="color: #57B88B;">●</strong> Tox21: 12 Classifiers</span>
                <span><strong style="color: #39C6D8;">●</strong> ChEMBL: 1,396 References</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
