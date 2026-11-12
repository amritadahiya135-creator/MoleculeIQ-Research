"""MoleculeIQ — Molecular Observatory Landing Page (Module M10).

Unifies the Molecular Universe (cinematic, memorable) and Molecular Laboratory
(precise, trustworthy computational research instrument) in a 10-section dark-themed experience.
"""

import base64
import re
import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from src.ui.brand import render_computational_network_svg, render_brand_header


def _render_dark_rdkit_svg(smiles: str, width: int = 380, height: int = 240) -> str:
    """Generate high-contrast transparent 2D vector depiction of molecule for dark backgrounds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = d2d.drawOptions()
    opts.clearBackground = False
    opts.setSymbolColour((243 / 255, 246 / 255, 250 / 255, 1.0))
    opts.updateAtomPalette({
        6: (243 / 255, 246 / 255, 250 / 255),    # C: Off-white
        8: (224 / 255, 120 / 255, 120 / 255),    # O: Coral red
        7: (114 / 255, 169 / 255, 232 / 255),    # N: Molecular blue
        16: (217 / 255, 164 / 255, 65 / 255),    # S: Amber yellow
        9: (87 / 255, 184 / 255, 139 / 255),     # F: Green
        17: (87 / 255, 184 / 255, 139 / 255),    # Cl: Green
        35: (217 / 255, 164 / 255, 65 / 255),    # Br: Amber
        53: (142 / 255, 124 / 255, 255 / 255),   # I: Purple
    })
    opts.padding = 0.08
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    txt = d2d.GetDrawingText()
    txt = re.sub(r"<\?xml.*?\?>", "", txt)
    txt = re.sub(r"<!--[\s\S]*?-->", "", txt)
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    cleaned = " ".join(lines)
    b64 = base64.b64encode(cleaned.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" height="{height}" style="max-width: 100%; height: auto; display: block; margin: 0 auto;" alt="2D Molecular Structure">'


def render_landing_page():
    """Render the 10-section Molecular Observatory landing page."""

    # Callback helpers for navigation
    def _go_to_single(smiles: str = "CC(=O)Oc1ccccc1C(=O)O"):
        st.session_state["nav_mode"] = "Single Molecule Analysis"
        st.session_state["active_nav_selector"] = "◎ Single Molecule Analysis"
        st.session_state["single_smiles_input"] = smiles

    def _go_to_batch():
        st.session_state["nav_mode"] = "Batch Molecular Screening"
        st.session_state["active_nav_selector"] = "◫ Batch Screening"

    def _go_to_workspace(mode: str, display: str):
        st.session_state["nav_mode"] = mode
        st.session_state["active_nav_selector"] = display

    # =========================================================================
    # SECTION 1: CINEMATIC HERO
    # =========================================================================
    hero_col_text, hero_col_visual = st.columns([1.35, 1.0], gap="large")

    with hero_col_text:
        st.markdown(
            """
            <div style="padding-top: 0.5rem;">
                <div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0.75rem; background: rgba(79, 157, 150, 0.12); border: 1px solid rgba(79, 157, 150, 0.35); border-radius: 9999px; font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1rem; font-family: 'IBM Plex Mono', monospace;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #39C6D8; box-shadow: 0 0 8px #39C6D8;"></span>
                    MOLECULAR OBSERVATORY &bull; AI-ASSISTED CHEMINFORMATICS
                </div>
                <h1 style="margin: 0 0 0.65rem 0; font-size: 2.85rem; font-weight: 800; color: #F3F6FA; letter-spacing: -0.03em; line-height: 1.12; font-family: 'Manrope', sans-serif;">
                    Every Molecule Has a Story.<br>
                    <span style="color: #4F9D96;">Discover What Lies Within.</span>
                </h1>
                <div style="font-size: 1.08rem; font-weight: 600; color: #39C6D8; margin-bottom: 0.65rem; font-family: 'Manrope', sans-serif;">
                    Explore Molecules. Understand Evidence.
                </div>
                <p style="margin: 0 0 1.5rem 0; font-size: 0.98rem; color: #AAB7C5; line-height: 1.65; max-width: 620px;">
                    MoleculeIQ unifies calibrated clinical toxicity estimation, 12 in-vitro biological stress assays,
                    chemical reference neighborhood matching, and TreeSHAP atom-level explainability in an integrated
                    computational research observatory.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_btn1, c_btn2 = st.columns([1, 1.15], gap="small")
        with c_btn1:
            st.button(
                "◎ Start Molecular Analysis",
                type="primary",
                use_container_width=True,
                key="hero_start_analysis",
                on_click=_go_to_single,
            )
        with c_btn2:
            st.button(
                "◫ Explore Batch Screening",
                use_container_width=True,
                key="hero_explore_platform",
                on_click=_go_to_batch,
            )

        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 1.5rem; margin-top: 1.25rem; font-size: 0.78rem; color: #8A99AD;">
                <span><strong style="color: #57B88B;">●</strong> Calibrated Random Forest</span>
                <span><strong style="color: #39C6D8;">●</strong> 12 Tox21 Assays</span>
                <span><strong style="color: #72A9E8;">●</strong> 1,396 FDA References</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_col_visual:
        st.markdown(
            f"""
            <div class="mol-card-elevated" style="padding: 1.5rem; display: flex; flex-direction: column; align-items: center; justify-content: center; background: radial-gradient(circle at center, #142236 0%, #0B1220 100%); border: 1px solid #1E314B; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);">
                {render_computational_network_svg(width=440, height=270)}
                <div style="display: flex; justify-content: space-between; width: 100%; margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #1E314B; font-size: 0.72rem; color: #8A99AD; font-family: 'IBM Plex Mono', monospace;">
                    <span>TOPOLOGICAL GRAPH</span>
                    <span>MORGAN FP (r=2, 1024-bit)</span>
                    <span>TREE-SHAP EXPLAINER</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: RESEARCH CAPABILITY INTRODUCTION (5 Cards)
    # =========================================================================
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                RESEARCH SUITE ARCHITECTURE
            </div>
            <h2 style="margin: 0.25rem 0 0.5rem 0; font-size: 1.85rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Integrated Computational Capabilities
            </h2>
            <p style="color: #AAB7C5; font-size: 0.92rem; max-width: 680px; margin: 0 auto; line-height: 1.55;">
                Five specialized analytical workspaces engineered to investigate chemical structures across molecular, phenotypic, and bibliographic dimensions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cap_cols = st.columns(5, gap="small")
    capabilities = [
        (
            "◎ Single Molecule",
            "Deep 3-zone investigation with 2D/3D conformers, Lipinski Rule of 5 descriptors, calibrated ClinTox risk, and atom-level TreeSHAP attributions.",
            "Single Molecule Analysis",
            "◎ Single Molecule Analysis",
        ),
        (
            "◫ Batch Screening",
            "High-throughput chemical library screening. Upload CSV callsets, evaluate applicability domain conformance, filter by endpoints, and rank by Exploratory Triage Index.",
            "Batch Molecular Screening",
            "◫ Batch Screening",
        ),
        (
            "⌕ Literature Research",
            "Real-time NCBI PubMed E-utilities retrieval. Semantic conjunction matching, automated sentence relevance classification, and direct PMID publication links.",
            "Literature Intelligence",
            "⌕ Literature Research",
        ),
        (
            "◉ Model Insights",
            "Platt-calibrated Random Forest registry, 3×4 Tox21 assay explorer, ROC-AUC calibration curves, global SHAP feature importances, and applicability domain boundaries.",
            "Model Insights & Metrics",
            "◉ Model Insights",
        ),
        (
            "📋 Export & Reporting",
            "Publication-grade dossier generation. Export single-molecule profiles, batch screening audits, and toxicity dossiers in Markdown, HTML, and JSON formats.",
            "Export & Reporting",
            "📋 Export & Reporting",
        ),
    ]

    for col, (title, desc, mode, disp) in zip(cap_cols, capabilities):
        with col:
            st.markdown(
                f"""
                <div class="mol-card" style="height: 100%; min-width: 0; display: flex; flex-direction: column; justify-content: space-between; padding: 1.15rem 1rem;">
                    <div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #F3F6FA; margin-bottom: 0.45rem; font-family: 'Manrope', sans-serif;">
                            {title}
                        </div>
                        <p style="font-size: 0.8rem; color: #AAB7C5; line-height: 1.5; margin: 0 0 1rem 0;">
                            {desc}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                f"Launch →",
                key=f"launch_cap_{mode}",
                use_container_width=True,
                on_click=lambda m=mode, d=disp: _go_to_workspace(m, d),
            )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: EXPLORE A MOLECULE (Live RDKit Example)
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                INTERACTIVE REFERENCE
            </div>
            <h2 style="margin: 0.25rem 0 0.5rem 0; font-size: 1.75rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Explore a Molecule: Aspirin (Acetylsalicylic Acid)
            </h2>
            <p style="color: #AAB7C5; font-size: 0.92rem; margin: 0; line-height: 1.55;">
                Live RDKit topological vector depiction, real-time descriptor calculation, and multi-dimensional investigation preview.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mol_col_svg, mol_col_data = st.columns([1, 1.25], gap="large")

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    with mol_col_svg:
        aspirin_svg = _render_dark_rdkit_svg(aspirin_smiles, width=380, height=240)
        st.markdown(
            f"""
            <div class="mol-card-dark" style="padding: 1.25rem; text-align: center;">
                {aspirin_svg}
                <div style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #1E314B; font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #39C6D8; word-break: break-all;">
                    {aspirin_smiles}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mol_col_data:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div>
                        <span style="font-size: 1.15rem; font-weight: 800; color: #F3F6FA;">Aspirin</span>
                        <span style="font-size: 0.78rem; color: #8A99AD; margin-left: 0.5rem;">C9H8O4</span>
                    </div>
                    <span class="mol-badge mol-badge-success">Lipinski Compliant</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.65rem; margin-bottom: 1rem;">
                    <div style="background: #142236; padding: 0.6rem; border-radius: 6px; border: 1px solid #1E314B;">
                        <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 600;">MOL WEIGHT</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F3F6FA;">180.16 <span style="font-size: 0.7rem; color: #8A99AD;">g/mol</span></div>
                    </div>
                    <div style="background: #142236; padding: 0.6rem; border-radius: 6px; border: 1px solid #1E314B;">
                        <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 600;">LOGP</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F3F6FA;">1.19</div>
                    </div>
                    <div style="background: #142236; padding: 0.6rem; border-radius: 6px; border: 1px solid #1E314B;">
                        <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 600;">TPSA</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F3F6FA;">63.60 <span style="font-size: 0.7rem; color: #8A99AD;">Å²</span></div>
                    </div>
                    <div style="background: #142236; padding: 0.6rem; border-radius: 6px; border: 1px solid #1E314B;">
                        <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 600;">H-BOND DONORS</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F3F6FA;">1</div>
                    </div>
                    <div style="background: #142236; padding: 0.6rem; border-radius: 6px; border: 1px solid #1E314B;">
                        <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 600;">H-BOND ACCEPTORS</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F3F6FA;">3</div>
                    </div>
                    <div style="background: #142236; padding: 0.6rem; border-radius: 6px; border: 1px solid #1E314B;">
                        <div style="font-size: 0.7rem; color: #8A99AD; font-weight: 600;">ROTATABLE BONDS</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F3F6FA;">2</div>
                    </div>
                </div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.85rem;">
                    <span style="font-size: 0.72rem; padding: 0.2rem 0.5rem; background: #1B2B3D; border-radius: 4px; color: #CBD5E1;">1. Structure View (2D/3D)</span>
                    <span style="font-size: 0.72rem; padding: 0.2rem 0.5rem; background: #1B2B3D; border-radius: 4px; color: #CBD5E1;">2. Physicochemical Properties</span>
                    <span style="font-size: 0.72rem; padding: 0.2rem 0.5rem; background: #1B2B3D; border-radius: 4px; color: #CBD5E1;">3. ClinTox & Tox21 Toxicity</span>
                    <span style="font-size: 0.72rem; padding: 0.2rem 0.5rem; background: #1B2B3D; border-radius: 4px; color: #CBD5E1;">4. TreeSHAP Attributions</span>
                    <span style="font-size: 0.72rem; padding: 0.2rem 0.5rem; background: #1B2B3D; border-radius: 4px; color: #CBD5E1;">5. Reference Similarity</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "◎ Open Aspirin in Single Molecule Analysis →",
            key="btn_open_aspirin_live",
            type="primary",
            use_container_width=True,
            on_click=lambda: _go_to_single(aspirin_smiles),
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 4: MULTIPLE DIMENSIONS OF INVESTIGATION (5 Dimensions)
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                ANALYTICAL FRAMEWORK
            </div>
            <h2 style="margin: 0.25rem 0 0.5rem 0; font-size: 1.75rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Multiple Dimensions of Molecular Investigation
            </h2>
            <p style="color: #AAB7C5; font-size: 0.92rem; margin: 0; line-height: 1.55;">
                A rigorous multi-tiered evaluation spanning atomic topology, empirical properties, machine learning inference, and literature evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4, d5 = st.columns(5, gap="small")
    dims = [
        (
            "1. Physicochemical",
            "Lipinski Rule of Five parameters, rotatable bonds, topological polar surface area, and molecular weight distribution.",
            "#4F9D96",
        ),
        (
            "2. Toxicity Predictions",
            "Calibrated ClinTox clinical trial failure probability combined with 12 independent Tox21 cellular pathway stress assays.",
            "#E07878",
        ),
        (
            "3. Reference Similarity",
            "Tanimoto structural neighborhood matching against 1,396 FDA-approved reference drugs in the ChEMBL database.",
            "#39C6D8",
        ),
        (
            "4. TreeSHAP Attribution",
            "Shapley additive explanations decomposing predictions into atomic-environment fingerprint contributions.",
            "#72A9E8",
        ),
        (
            "5. Literature Evidence",
            "Real-time NCBI PubMed E-utilities integration querying biomedical literature and ranking sentence-level evidence.",
            "#D9A441",
        ),
    ]

    for col, (title, desc, accent) in zip([d1, d2, d3, d4, d5], dims):
        with col:
            st.markdown(
                f"""
                <div class="mol-card" style="height: 100%; min-width: 0; border-top: 3px solid {accent}; padding: 1.15rem 1rem;">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #F3F6FA; margin-bottom: 0.45rem;">
                        {title}
                    </div>
                    <p style="font-size: 0.8rem; color: #AAB7C5; line-height: 1.5; margin: 0;">
                        {desc}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 5: SCIENTIFIC MODEL INTERPRETATION (ClinTox, Tox21, SHAP)
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                MODEL ARCHITECTURE & INTERPRETATION
            </div>
            <h2 style="margin: 0.25rem 0 0.5rem 0; font-size: 1.75rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Calibrated Predictive Models & Explainability
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m_col1, m_col2, m_col3 = st.columns(3, gap="medium")

    with m_col1:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.74rem; font-weight: 700; color: #4F9D96; letter-spacing: 0.08em; text-transform: uppercase;">CLINICAL TOXICITY</span>
                    <span class="mol-badge mol-badge-teal">0.852 ROC-AUC</span>
                </div>
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #F3F6FA;">Platt-Calibrated ClinTox</h3>
                <p style="font-size: 0.84rem; color: #AAB7C5; line-height: 1.55; margin-bottom: 0.75rem;">
                    Random Forest ensemble trained on FDA clinical trial failure outcomes (CT_TOX). Calibrated via Platt sigmoid scaling
                    leaving ranking discrimination invariant while yielding reliable posterior probabilities.
                </p>
                <div style="background: #142236; padding: 0.5rem 0.75rem; border-radius: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #39C6D8; border: 1px solid #1E314B;">
                    P(y=1|f) = 1 / (1 + exp(a*f + b))
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.74rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.08em; text-transform: uppercase;">IN-VITRO SUITE</span>
                    <span class="mol-badge mol-badge-info">12 Assays</span>
                </div>
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #F3F6FA;">Tox21 Assay Battery</h3>
                <p style="font-size: 0.84rem; color: #AAB7C5; line-height: 1.55; margin-bottom: 0.75rem;">
                    Twelve dedicated classifiers predicting cellular stress response and nuclear receptor pathway activation:
                    NR-AR, NR-AhR, NR-ER, NR-Aromatase, NR-PPAR-γ, SR-p53, SR-MMP, SR-ATAD5, SR-HSE, and antioxidant response (ARE).
                </p>
                <div style="background: #142236; padding: 0.5rem 0.75rem; border-radius: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #39C6D8; border: 1px solid #1E314B;">
                    Mean ROC-AUC: 0.767 &bull; 12 Independent Endpoints
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.74rem; font-weight: 700; color: #72A9E8; letter-spacing: 0.08em; text-transform: uppercase;">LOCAL EXPLAINABILITY</span>
                    <span class="mol-badge mol-badge-info">TreeSHAP</span>
                </div>
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #F3F6FA;">Atom-Level Attributions</h3>
                <p style="font-size: 0.84rem; color: #AAB7C5; line-height: 1.55; margin-bottom: 0.75rem;">
                    Exact Shapley values computed via TreeExplainer on Morgan fingerprint bit substructures. Attributions describe model prediction contributions,
                    highlighting which fragments raise or lower estimated probability without asserting biological causation.
                </p>
                <div style="background: #142236; padding: 0.5rem 0.75rem; border-radius: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #72A9E8; border: 1px solid #1E314B;">
                    Additive Shapley Values &bull; Substructure Bit Mapping
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 6: LITERATURE INTELLIGENCE PREVIEW (Curated PubMed Cards)
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                BIBLIOGRAPHIC INTELLIGENCE
            </div>
            <h2 style="margin: 0.25rem 0 0.5rem 0; font-size: 1.75rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Evidence Grounding with NCBI PubMed
            </h2>
            <p style="color: #AAB7C5; font-size: 0.92rem; margin: 0; line-height: 1.55;">
                Automated concept-conjunction querying linking in silico predictions to peer-reviewed pharmacological and toxicological publications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lit1, lit2 = st.columns(2, gap="medium")

    with lit1:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
                    <span style="font-size: 0.72rem; color: #8A99AD; font-weight: 600;">NEW ENGL J MED &bull; 2018</span>
                    <span class="mol-badge mol-badge-teal">PMID: 30221597</span>
                </div>
                <h4 style="margin: 0 0 0.45rem 0; font-size: 0.96rem; color: #F3F6FA; line-height: 1.35;">
                    Effect of Aspirin on Cardiovascular Events and Bleeding in the Healthy Elderly.
                </h4>
                <p style="font-size: 0.82rem; color: #AAB7C5; line-height: 1.5; margin-bottom: 0.75rem;">
                    Aspirin use in healthy elderly persons did not prolong disability-free survival but led to a significantly higher rate of major hemorrhage.
                </p>
                <div style="font-size: 0.74rem; color: #39C6D8; border-top: 1px solid #1E314B; padding-top: 0.5rem;">
                    Extracted Concept: Prostaglandin G/H synthase inhibition &bull; Gastrointestinal bleeding risk
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lit2:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
                    <span style="font-size: 0.72rem; color: #8A99AD; font-weight: 600;">LANCET &bull; 2016</span>
                    <span class="mol-badge mol-badge-teal">PMID: 27150917</span>
                </div>
                <h4 style="margin: 0 0 0.45rem 0; font-size: 0.96rem; color: #F3F6FA; line-height: 1.35;">
                    Mechanisms of NSAID-induced Gastroduodenal Toxicity and Hepatocellular Perturbations.
                </h4>
                <p style="font-size: 0.82rem; color: #AAB7C5; line-height: 1.5; margin-bottom: 0.75rem;">
                    Topical and systemic mechanisms of cyclooxygenase-1 inhibition, uncoupling of mitochondrial oxidative phosphorylation, and mucosal barrier breakdown.
                </p>
                <div style="font-size: 0.74rem; color: #39C6D8; border-top: 1px solid #1E314B; padding-top: 0.5rem;">
                    Extracted Concept: Mitochondrial membrane permeability &bull; Hepatic metabolic stress
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 7: 5-STEP MOLECULEIQ WORKFLOW
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                RESEARCH WORKFLOW
            </div>
            <h2 style="margin: 0.25rem 0 0.5rem 0; font-size: 1.75rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                The 5-Step MoleculeIQ Investigation Pipeline
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    w1, w2, w3, w4, w5 = st.columns(5, gap="small")
    steps = [
        ("Step 1", "Input & Validation", "Enter SMILES string or upload library CSV; RDKit sanitizes coordinates and checks chemical valence."),
        ("Step 2", "Characterize", "Compute Lipinski Rule of 5 properties, TPSA, LogP, and generate MMFF94 force-field 3D conformers."),
        ("Step 3", "Model Inference", "Run calibrated ClinTox Random Forest, 12 Tox21 assays, and applicability domain boundary verification."),
        ("Step 4", "Contextualize", "Search 1,396 ChEMBL reference drugs via Tanimoto metric and retrieve live NCBI PubMed literature abstracts."),
        ("Step 5", "Document & Export", "Synthesize findings into an Exploratory Triage Index and export complete research dossiers in Markdown/HTML."),
    ]

    for col, (num, title, text) in zip([w1, w2, w3, w4, w5], steps):
        with col:
            st.markdown(
                f"""
                <div class="mol-card" style="height: 100%; min-width: 0; padding: 1.15rem 1rem;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #39C6D8; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.25rem;">
                        {num}
                    </div>
                    <div style="font-size: 0.92rem; font-weight: 700; color: #F3F6FA; margin-bottom: 0.4rem;">
                        {title}
                    </div>
                    <p style="font-size: 0.78rem; color: #AAB7C5; line-height: 1.45; margin: 0;">
                        {text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 8: TECHNOLOGY & METHODOLOGY STACK
    # =========================================================================
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                PROVENANCE & FOUNDATION
            </div>
            <h2 style="margin: 0.25rem 0 0.5rem 0; font-size: 1.75rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Scientific Technologies & Methodologies
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.columns(4, gap="medium")
    with t1:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <strong style="color: #4F9D96; font-size: 0.95rem;">RDKit</strong>
                <p style="font-size: 0.8rem; color: #AAB7C5; margin: 0.35rem 0 0 0; line-height: 1.5;">
                    Open-source cheminformatics toolkit powering 2D vector depiction, MMFF94 force-field 3D conformers, and Morgan circular fingerprints.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <strong style="color: #39C6D8; font-size: 0.95rem;">Scikit-Learn & TreeSHAP</strong>
                <p style="font-size: 0.8rem; color: #AAB7C5; margin: 0.35rem 0 0 0; line-height: 1.5;">
                    Calibrated Random Forest ensembles with Platt probability scaling and game-theoretic Shapley additive feature attributions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <strong style="color: #72A9E8; font-size: 0.95rem;">ChEMBL & ClinTox</strong>
                <p style="font-size: 0.8rem; color: #AAB7C5; margin: 0.35rem 0 0 0; line-height: 1.5;">
                    Curated reference libraries spanning 1,396 approved drugs and FDA clinical trial outcome datasets for domain-anchored benchmarking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t4:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <strong style="color: #D9A441; font-size: 0.95rem;">NCBI PubMed E-Utilities</strong>
                <p style="font-size: 0.8rem; color: #AAB7C5; margin: 0.35rem 0 0 0; line-height: 1.5;">
                    Automated Entrez programming utilities integrating real-time peer-reviewed biomedical literature and sentence semantic scoring.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 9: SCIENTIFIC TRANSPARENCY & NON-CLINICAL DISCLAIMER
    # =========================================================================
    st.markdown(
        """
        <div style="background: #0B1220; border-left: 3px solid #D9A441; border-top: 1px solid #1E314B; border-right: 1px solid #1E314B; border-bottom: 1px solid #1E314B; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.75rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.74rem; font-weight: 700; color: #D9A441; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'IBM Plex Mono', monospace;">
                    SCIENTIFIC GOVERNANCE & TRANSPARENCY STATEMENT
                </span>
                <span class="mol-badge mol-badge-warning">In Silico Research Use Only</span>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #F3F6FA;">
                Computational Predictions Require Experimental Verification
            </h3>
            <p style="margin: 0 0 0.65rem 0; font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
                MoleculeIQ is an in silico screening instrument designed exclusively for computational chemistry and academic research.
                Model outputs, including calibrated ClinTox probabilities and the Exploratory Triage Index, are statistical estimations
                and do not constitute clinical diagnostics, regulatory clearance, or pharmacological proof of safety or efficacy.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.75rem; padding-top: 0.65rem; border-top: 1px solid #1E314B; font-size: 0.8rem; color: #8A99AD;">
                <div><strong style="color: #F3F6FA;">Domain Extrapolation:</strong> Structures outside the applicability domain carry elevated uncertainty.</div>
                <div><strong style="color: #F3F6FA;">SHAP Explanations:</strong> Attributions describe model prediction contributions, not biological mechanisms.</div>
                <div><strong style="color: #F3F6FA;">Literature Mining:</strong> Extracted publication sentences are automated heuristic interpretations.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION 10: FINAL CALL TO ACTION (With Quick-Start Compound Pills)
    # =========================================================================
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 1.5rem; background: radial-gradient(circle at center, #142236 0%, #0B1220 100%); border: 1px solid #1E314B; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);">
            <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;">
                READY TO COMMENCE RESEARCH
            </div>
            <h2 style="margin: 0.35rem 0 0.65rem 0; font-size: 2.15rem; font-weight: 800; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                Begin Your Molecular Investigation
            </h2>
            <p style="color: #AAB7C5; font-size: 0.95rem; max-width: 600px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                Select a benchmark reference compound below or launch directly into Single Molecule Analysis with any valid SMILES representation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_cols = st.columns(6, gap="small")
    quick_compounds = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Nicotine", "CN1CCC[C@H]1c2cccnc2"),
        ("Metformin", "CN(C)C(=N)N=C(N)N"),
    ]

    for col, (name, smiles) in zip(cta_cols, quick_compounds):
        with col:
            st.button(
                f"◎ {name}",
                key=f"cta_pill_{name}",
                use_container_width=True,
                on_click=lambda s=smiles: _go_to_single(s),
            )
