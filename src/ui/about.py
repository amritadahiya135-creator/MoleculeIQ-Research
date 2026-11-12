"""MoleculeIQ — About & Scientific Architecture (Module M10)."""

import streamlit as st
from src.ui.components import render_header


def render_about_view():
    """Render the About & Scientific Architecture view."""
    render_header(
        "About MoleculeIQ",
        "Design philosophy, scientific foundations, peer-reviewed citations, and methodological governance.",
        badge_text="Molecular Observatory v5.0.0",
        badge_variant="teal",
    )

    st.markdown("### The Molecular Observatory")
    st.markdown(
        """
        <div class="mol-card" style="padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #F3F6FA; margin-bottom: 0.5rem; font-family: 'Manrope', sans-serif;">
                Every Molecule Has a Story. Discover What Lies Within.
            </div>
            <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.65; margin: 0 0 1rem 0;">
                MoleculeIQ — Molecular Observatory unifies two complementary perspectives: the <strong>Molecular Universe</strong>
                (a cinematic, memorable, scientifically grounded brand experience) and the <strong>Molecular Laboratory</strong>
                (a precise, practical, and trustworthy computational research instrument).
            </p>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.8rem; color: #8A99AD;">
                <span><strong>Concept:</strong> Cinematic Outside &bull; Scientific Inside</span>
                <span><strong>Design:</strong> Non-Negotiable Dark Theme</span>
                <span><strong>Release:</strong> v5.0.0 Public Distribution</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Scientific Technologies & Data Sources")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #39C6D8;">Cheminformatics & Machine Learning</h4>
                <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.6;">
                    <div>• <strong>RDKit (2024+):</strong> Open-source cheminformatics for structure sanitization, 2D vector drawing, MMFF94 3D conformers, and Morgan fingerprints.</div>
                    <div>• <strong>Scikit-Learn:</strong> Random Forest classifiers with Platt probability calibration ($P(y=1|f) = 1/(1+e^{af+b})$).</div>
                    <div>• <strong>TreeSHAP:</strong> Additive feature attribution decomposing model predictions to Morgan fingerprint sub-fragments.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="mol-card" style="height: 100%;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #4F9D96;">Benchmarks & Bibliographic Sources</h4>
                <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.6;">
                    <div>• <strong>ClinTox (MoleculeNet):</strong> FDA clinical trial outcomes benchmark (1,478 compounds).</div>
                    <div>• <strong>Tox21 Data Challenge:</strong> 12 cellular stress response and nuclear receptor assays.</div>
                    <div>• <strong>ChEMBL Reference Database:</strong> 1,396 approved small-molecule therapeutics.</div>
                    <div>• <strong>NCBI PubMed E-Utilities:</strong> Live automated Entrez API retrieval of peer-reviewed biomedical literature.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    st.markdown("### Key Scientific Citations")
    st.markdown(
        """
        <div class="mol-card" style="padding: 1.15rem; font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <div>1. <strong>Landrum, G.</strong> (2024). RDKit: Open-source cheminformatics. <em>https://www.rdkit.org</em></div>
            <div>2. <strong>Wu, Z., et al.</strong> (2018). MoleculeNet: A benchmark for molecular machine learning. <em>Chemical Science</em>, 9(2), 513-530.</div>
            <div>3. <strong>Lundberg, S. M., & Lee, S. I.</strong> (2017). A unified approach to interpreting model predictions. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 30.</div>
            <div>4. <strong>Platt, J.</strong> (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. <em>Advances in Large Margin Classifiers</em>, 10(3), 61-74.</div>
            <div>5. <strong>Sayers, E., et al.</strong> (2024). Database resources of the National Center for Biotechnology Information. <em>Nucleic Acids Research</em>, 52(D1), D20-D32.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background: #0B1220; border-left: 3px solid #D9A441; border-top: 1px solid #1E314B; border-right: 1px solid #1E314B; border-bottom: 1px solid #1E314B; border-radius: 8px; padding: 1.1rem 1.35rem; font-size: 0.82rem; color: #CBD5E1; line-height: 1.55;">
            <strong style="color: #F3F6FA;">Scientific & Regulatory Notice:</strong> Research use only. Model outputs and the Exploratory Triage Index are computational estimates, not determinations of clinical safety, efficacy, or suitability for human use. Applicability-domain warnings indicate structural distance from the model's training data and do not independently establish toxicity.
        </div>
        """,
        unsafe_allow_html=True,
    )
