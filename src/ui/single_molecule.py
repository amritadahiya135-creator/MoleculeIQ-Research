"""Single Molecule Analysis view (Workflow A)."""

import json
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from src.services.orchestration import ScreeningOrchestrator
from src.ui.components import render_header, render_metric_card

PRESET_MOLECULES = {
    "Aspirin (Analgesic)": "CC(=O)Oc1ccccc1C(=O)O",
    "Acetaminophen / Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine (Stimulant)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Atorvastatin (Lipitor - Statin)": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Metformin (Antidiabetic)": "CN(C)C(=N)NC(=N)N",
    "Ibuprofen (NSAID)": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
}


def render_molecule_svg(smiles: str, width: int = 350, height: int = 260) -> str:
    """Generate 2D SVG vector graphic for molecular structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d2d.drawOptions().padding = 0.08
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def render_single_molecule_view(orchestrator: ScreeningOrchestrator):
    """Render Workflow A: Single Molecule Analysis."""
    render_header(
        "Single Molecule Analysis",
        "Comprehensive in silico profiling: validation, descriptors, toxicity, similarity, and SHAP explainability.",
    )

    st.subheader("1. Molecular Input")

    col_presets, col_custom = st.columns([1, 2])
    with col_presets:
        selected_preset = st.selectbox("Quick-Load Reference Compounds:", ["-- Select Preset --"] + list(PRESET_MOLECULES.keys()))
        default_smiles = PRESET_MOLECULES[selected_preset] if selected_preset != "-- Select Preset --" else "CC(=O)Oc1ccccc1C(=O)O"

    with col_custom:
        smiles_input = st.text_input("Enter SMILES string:", value=default_smiles, help="Simplified Molecular Input Line Entry System")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_analysis = st.button("Run Full Analysis", type="primary", use_container_width=True)

    if smiles_input:
        with st.spinner("Executing screening pipelines and model inference..."):
            res = orchestrator.analyze_single_molecule(smiles_input, compute_explanation=True)

        if not res.is_valid:
            st.error(f"Validation Failure: {res.error_message}")
            return

        st.success(f"Validated Structure: Formula `{res.validation.formula}` | Canonical: `{res.validation.canonical_smiles}`")

        st.divider()

        # Row 1: 2D Structure & Physicochemical Descriptors
        col_img, col_desc = st.columns([1, 1.2])

        with col_img:
            st.markdown("### 2D Chemical Structure")
            svg_data = render_molecule_svg(res.validation.canonical_smiles)
            if svg_data:
                st.markdown(svg_data, unsafe_allow_html=True)

        with col_desc:
            st.markdown("### Physicochemical Properties (Lipinski)")
            if res.descriptors:
                d = res.descriptors
                lip_status = "Pass" if d.lipinski_pass else f"Fail ({d.lipinski_violations} violations)"
                lip_color = "#16a34a" if d.lipinski_pass else "#dc2626"

                desc_data = {
                    "Property": [
                        "Molecular Weight (MW)",
                        "LogP (Crippen)",
                        "H-Bond Donors (HBD)",
                        "H-Bond Acceptors (HBA)",
                        "Rotatable Bonds",
                        "Topological Polar Surface Area (TPSA)",
                        "Lipinski Rule of 5",
                    ],
                    "Value": [
                        f"{d.molecular_weight:.2f} Da",
                        f"{d.logp:.2f}",
                        f"{d.hbond_donors}",
                        f"{d.hbond_acceptors}",
                        f"{d.rotatable_bonds}",
                        f"{d.tpsa:.2f} Å²",
                        lip_status,
                    ],
                    "Threshold": ["<= 500 Da", "<= 5.0", "<= 5", "<= 10", "<= 10", "Optimal < 140 Å²", "0 Violations"],
                }
                st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)
                st.caption("ℹ️ **HBA Definition:** Evaluated using RDKit's physiological Lipinski rule (excludes resonance-delocalized amide nitrogens).")

        st.divider()

        # Row 2: Prioritization Score Summary
        st.markdown("### Candidate Prioritization Score")
        if res.prioritization and res.prioritization.is_complete:
            p = res.prioritization
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Overall Priority (R)", f"{p.overall_score:.1f}", "R = 0.5S + 0.3D + 0.2M", "#2563eb")
            with m2:
                render_metric_card("Safety Score (S)", f"{p.safety_score:.1f}", "S = (1 - P_tox) * 100", "#16a34a" if p.safety_score >= 50 else "#dc2626")
            with m3:
                render_metric_card("Drug-Likeness (D)", f"{p.druglikeness_score:.0f}", "100 if Lipinski else 40", "#0891b2")
            with m4:
                render_metric_card("Reference Similarity (M)", f"{p.similarity_score:.1f}", "100 * best Tanimoto", "#9333ea")

        st.divider()

        # Row 3: Toxicity Predictions (ClinTox & Tox21)
        col_clin, col_tox21 = st.columns([1, 1.2])

        with col_clin:
            st.markdown("### ClinTox Clinical Trial Toxicity")
            if res.clintox and res.clintox.status == "SUCCESS":
                p_tox = res.clintox.probability
                pred_label = "HIGH RISK (Toxic)" if res.clintox.is_toxic else "LOW RISK (Non-Toxic)"
                label_color = "#dc2626" if res.clintox.is_toxic else "#16a34a"

                st.markdown(f"**Predicted Class:** <span style='color:{label_color}; font-weight:700;'>{pred_label}</span>", unsafe_allow_html=True)
                st.progress(p_tox)
                st.caption(f"Estimated Probability of Trial Failure: **{p_tox*100:.1f}%** | Threshold: 50%")
                st.info(f"Target: `{res.clintox.task_definition}`")

        with col_tox21:
            st.markdown("### Tox21 12-Endpoint Assay Activation")
            if res.tox21 and res.tox21.status == "SUCCESS":
                t21 = res.tox21
                st.write(f"Active Assay Pathways Detected: **{t21.active_endpoints_count} / {t21.total_endpoints}**")

                ep_rows = []
                for ep, pred in t21.endpoints.items():
                    ep_rows.append({
                        "Assay Endpoint": ep,
                        "Pathway Type": "Nuclear Receptor (NR)" if ep.startswith("NR") else "Stress Response (SR)",
                        "Probability": f"{pred.probability*100:.1f}%",
                        "Predicted Activity": "Active" if pred.is_active else "Inactive",
                    })
                df_ep = pd.DataFrame(ep_rows)
                st.dataframe(df_ep, use_container_width=True, hide_index=True)

        st.divider()

        # Row 4: Chemical Similarity Search & Approved Analogues
        st.markdown("### Structural Similarity to Approved Small Molecules (ChEMBL Library)")
        if res.similarity and res.similarity.status == "SUCCESS":
            matches = res.similarity.matches
            if matches:
                sim_rows = []
                for m in matches:
                    sim_rows.append({
                        "Ref ID": m.reference_id,
                        "Drug Name": m.name,
                        "Tanimoto Similarity": f"{m.similarity:.4f}",
                        "Molecular Weight": f"{m.molecular_weight:.1f} Da" if m.molecular_weight else "N/A",
                        "Lipinski": "Pass" if m.lipinski_pass else "Fail",
                        "Structure": m.canonical_smiles,
                    })
                st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Row 5: Explainable AI (SHAP TreeExplainer & Substructure Highlighting)
        st.markdown("### Model Interpretability & Substructure Attribution (SHAP)")
        if res.explanation and res.explanation.status == "SUCCESS":
            x_col1, x_col2 = st.columns([1, 1.2])

            with x_col1:
                st.markdown("**Highlighted Atom Environments (Top Active Bits)**")
                if res.explanation.highlighted_svg:
                    st.markdown(res.explanation.highlighted_svg, unsafe_allow_html=True)

            with x_col2:
                st.markdown("**Top Influential Active Fingerprint Features**")
                feat_rows = []
                for f in res.explanation.top_features:
                    feat_rows.append({
                        "Morgan Bit ID": f.bit_id,
                        "SHAP Attribution": f"{f.shap_value:+.5f}",
                        "Influence on Toxicity Risk": "Increases Risk" if f.shap_value > 0 else "Decreases Risk",
                        "Radius (Bonds)": f.radius,
                        "Atoms Involved": str(f.atom_indices),
                    })
                st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

            st.caption(res.explanation.disclaimer)

        st.divider()

        # Export Analysis Section
        st.markdown("### Export Analysis Record")
        export_dict = {
            "smiles": res.validation.canonical_smiles,
            "formula": res.validation.formula,
            "descriptors": res.descriptors.__dict__ if res.descriptors else None,
            "clintox": res.clintox.__dict__ if res.clintox else None,
            "prioritization": res.prioritization.__dict__ if res.prioritization else None,
            "timestamp": res.analysis_timestamp,
        }
        st.download_button(
            label="Download Analysis JSON Record",
            data=json.dumps(export_dict, indent=2),
            file_name=f"moleculeiq_analysis_{res.validation.formula}.json",
            mime="application/json",
        )
