"""Single Molecule Analysis view (Workflow A) — Flagship Research Workspace."""

from dataclasses import asdict
import json
import re
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from src.services.orchestration import ScreeningOrchestrator
from src.ui.components import (
    render_header,
    render_triage_editorial_grid,
    render_risk_gauge,
    render_property_bar,
    render_executive_interpretation,
)


def generate_3d_mol_block(smiles: str) -> Optional[str]:
    """
    Generate an energy-minimized 3D conformer SDF MolBlock using RDKit ETKDGv3
    and MMFF94 force field optimization.
    Returns None if embedding fails or molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status != 0:
            embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
            if embed_status != 0:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def render_3dmol_viewer(mol_block: str, height: int = 310) -> None:
    """
    Render an interactive, brand-coordinated 3D molecular conformer viewer using 3Dmol.js.
    Features:
    - Orbit control (drag left mouse button to rotate)
    - Zoom (scroll wheel)
    - Pan (right mouse button / two-finger drag)
    - Spin toggle & Camera reset
    - Midnight Navy canvas background matching the MoleculeIQ palette
    """
    escaped_block = (
        mol_block.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
    )
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://3dmol.org/build/3Dmol-min.js"></script>
      <style>
        body, html {{
          margin: 0;
          padding: 0;
          width: 100%;
          height: 100%;
          overflow: hidden;
          background: #101B2D;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        #container {{
          position: relative;
          width: 100%;
          height: 100%;
        }}
        #viewer {{
          width: 100%;
          height: 100%;
        }}
        #toolbar {{
          position: absolute;
          top: 8px;
          right: 8px;
          display: flex;
          gap: 6px;
          z-index: 20;
        }}
        .btn-3d {{
          background: rgba(28, 43, 64, 0.90);
          color: #F4F7FB;
          border: 1px solid #27C7B8;
          border-radius: 4px;
          padding: 3px 8px;
          font-size: 11px;
          font-weight: 600;
          cursor: pointer;
          box-shadow: 0 1px 3px rgba(0,0,0,0.3);
          transition: all 0.15s ease-in-out;
        }}
        .btn-3d:hover {{
          background: #27C7B8;
          color: #101B2D;
        }}
        #attribution {{
          position: absolute;
          bottom: 6px;
          left: 10px;
          color: #94A3B8;
          font-size: 10px;
          pointer-events: none;
          letter-spacing: 0.04em;
        }}
      </style>
    </head>
    <body>
      <div id="container">
        <div id="viewer"></div>
        <div id="toolbar">
          <button id="spinBtn" class="btn-3d" onclick="toggleSpin()">Auto-Rotate: On</button>
          <button class="btn-3d" onclick="resetView()">Reset View</button>
        </div>
        <div id="attribution">3Dmol.js &bull; RDKit ETKDGv3 / MMFF94</div>
      </div>
      <script>
        var viewer = null;
        var spinning = true;
        var molData = `{escaped_block}`;
        document.addEventListener("DOMContentLoaded", function() {{
          var element = document.getElementById("viewer");
          var config = {{ backgroundColor: "#101B2D" }};
          viewer = $3Dmol.createViewer(element, config);
          viewer.addModel(molData, "sdf");
          viewer.setStyle({{}}, {{
            stick: {{ radius: 0.14, colorscheme: "Jmol" }},
            sphere: {{ scale: 0.28, colorscheme: "Jmol" }}
          }});
          viewer.zoomTo();
          viewer.render();
          viewer.spin(true);
        }});
        function toggleSpin() {{
          if (!viewer) return;
          spinning = !spinning;
          viewer.spin(spinning);
          document.getElementById("spinBtn").innerText = spinning ? "Auto-Rotate: On" : "Auto-Rotate: Off";
        }}
        function resetView() {{
          if (!viewer) return;
          viewer.zoomTo();
          viewer.render();
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)


PRESET_MOLECULES = {
    "Aspirin (Analgesic / Anti-inflammatory)": "CC(=O)Oc1ccccc1C(=O)O",
    "Acetaminophen / Paracetamol (Analgesic)": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine (CNS Stimulant)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Atorvastatin (Lipitor - Statin)": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Metformin (Biguanide Antidiabetic)": "CN(C)C(=N)NC(=N)N",
    "Ibuprofen (NSAID Analgesic)": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Sorafenib (Kinase Inhibitor Antineoplastic)": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Warfarin (Coumarin Anticoagulant)": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
}

TOX21_METADATA = {
    "NR-AR": {"name": "Androgen Receptor (Full Length)", "desc": "Endocrine signaling & male hormone regulation"},
    "NR-AR-LBD": {"name": "Androgen Receptor (Ligand Binding Domain)", "desc": "Competitive receptor ligand antagonism/agonism"},
    "NR-AhR": {"name": "Aryl Hydrocarbon Receptor", "desc": "Dioxin sensing & xenobiotic metabolism enzyme induction"},
    "NR-Aromatase": {"name": "Aromatase (CYP19A1)", "desc": "Estrogen biosynthesis rate-limiting enzyme"},
    "NR-ER": {"name": "Estrogen Receptor Alpha (Full Length)", "desc": "Estrogen hormonal signaling pathway"},
    "NR-ER-LBD": {"name": "Estrogen Receptor Alpha (LBD)", "desc": "Direct competitive binding to estrogen receptor"},
    "NR-PPAR-gamma": {"name": "PPAR-Gamma Receptor", "desc": "Adipogenesis, lipid storage & glucose homeostasis"},
    "SR-ARE": {"name": "Antioxidant Response Element (Nrf2)", "desc": "Oxidative stress sensing & detoxifying enzyme transcription"},
    "SR-ATAD5": {"name": "ATAD5 (DNA Damage Response)", "desc": "Genotoxic stress, DNA replication fork stall & repair"},
    "SR-HSE": {"name": "Heat Shock Response Element", "desc": "Proteotoxic stress & misfolded protein chaperone induction"},
    "SR-MMP": {"name": "Mitochondrial Membrane Potential", "desc": "Mitochondrial depolarization & early apoptotic cascade"},
    "SR-p53": {"name": "p53 Tumor Suppressor Pathway", "desc": "Genotoxic injury, DNA double-strand breaks & cell cycle arrest"},
}


def build_single_molecule_export_dict(res: Any) -> Dict[str, Any]:
    """
    Construct the machine-readable JSON export dictionary for Single Molecule Analysis.
    Includes complete Tox21 12-endpoint predictions alongside descriptors, ClinTox, and
    prioritization metadata while preserving raw vs. calibrated probability distinction.
    """
    tox21_obj = getattr(res, "tox21", None)
    return {
        "smiles": res.validation.canonical_smiles,
        "formula": res.validation.formula,
        "descriptors": res.descriptors.__dict__ if res.descriptors else None,
        "clintox": res.clintox.__dict__ if res.clintox else None,
        "tox21": asdict(tox21_obj) if tox21_obj is not None else None,
        "prioritization": res.prioritization.__dict__ if res.prioritization else None,
        "timestamp": res.analysis_timestamp,
    }


def build_tox21_endpoint_table_rows(t21: Any) -> List[Dict[str, Any]]:
    """
    Build structured display rows for the Single Molecule 12-Assay Endpoint Table.
    Preserves separate raw and calibrated probability columns, exposes endpoint-level
    calibration status, and keeps valid 0.0 ('0.0%') distinct from unavailable ('N/A').
    """
    ep_rows: List[Dict[str, Any]] = []
    if not t21 or not getattr(t21, "endpoints", None):
        return ep_rows
    for ep, pred in t21.endpoints.items():
        meta = TOX21_METADATA.get(ep, {"name": ep, "desc": ""})
        ad_score_str = f"{pred.applicability_score:.3f}" if pred.applicability_score is not None else "N/A"
        ad_stat = getattr(pred, "applicability_domain_status", "UNVALIDATED")
        cal_prob = getattr(pred, "calibrated_probability", None)
        cal_prob_str = f"{cal_prob * 100:.1f}%" if cal_prob is not None else "N/A"
        cal_meth = getattr(pred, "calibration_method", None) or "N/A"
        cal_stat = getattr(pred, "calibration_status", "UNAVAILABLE") or "UNAVAILABLE"
        ep_rows.append({
            "Assay Endpoint": ep,
            "Target Name": meta["name"],
            "Category": "Nuclear Receptor (NR)" if ep.startswith("NR") else "Stress Response (SR)",
            "Probability": f"{pred.probability * 100:.1f}%",
            "Calibrated Prob.": cal_prob_str,
            "Calibration Method": cal_meth,
            "Calibration Status": cal_stat,
            "Predicted Activity": "Active" if pred.is_active else "Inactive",
            "AD Score (Tanimoto)": ad_score_str,
            "AD Status": ad_stat,
        })
    return ep_rows


def render_molecule_svg(smiles: str, width: int = 420, height: int = 280) -> str:
    """Generate high-contrast 2D SVG vector graphic for molecular structure."""
    import base64
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d2d.drawOptions().padding = 0.08
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    txt = d2d.GetDrawingText()
    # Strip XML declaration, comments, and leading indentation to prevent markdown pre-block formatting
    txt = re.sub(r"<\?xml.*?\?>", "", txt)
    txt = re.sub(r"<!--[\s\S]*?-->", "", txt)
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    cleaned = " ".join(lines)
    b64 = base64.b64encode(cleaned.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" height="{height}" style="max-width: 100%; height: auto; display: block; margin: 0 auto;" alt="2D Molecular Structure">'


def render_empty_state():
    """Render an inviting, restrained empty state before molecule profiling."""
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1.5rem; margin-top: 1rem; border: 1px dashed #D9E2EF; border-radius: 12px; background: #FFFFFF; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
            <div style="font-size: 2.4rem; margin-bottom: 0.5rem;">🔬</div>
            <h3 style="margin: 0 0 0.4rem 0; font-size: 1.35rem; color: #101B2D; font-weight: 700;">
                No Molecule Analyzed
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; max-width: 620px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                Enter any valid SMILES notation above or select an approved reference compound (e.g. Aspirin, Caffeine, or Atorvastatin), then click
                <strong style="color: #27C7B8;">Run Full Analysis</strong> to begin multi-dimensional profiling.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; font-size: 0.82rem; color: #64748B;">
                <span>✓ Structure Validation</span>
                <span>✓ Lipinski Descriptors</span>
                <span>✓ ClinTox & 12 Tox21 Assays</span>
                <span>✓ TreeSHAP Explainability</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_molecule_view(orchestrator: ScreeningOrchestrator):
    """Render Workflow A: Single Molecule Research Workspace."""
    render_header(
        "Single Molecule Analysis",
        "Comprehensive in silico profiling: validation, descriptors, calibrated toxicity, similarity, and TreeSHAP explainability.",
        badge_text="Flagship Workspace",
        badge_variant="info",
    )

    # Chemical Structure Input Card (Toolbar)
    with st.container():
        st.markdown("""<div class="mol-card" style="padding: 1.15rem 1.35rem; margin-bottom: 1.25rem;">""", unsafe_allow_html=True)
        col_presets, col_custom = st.columns([1.1, 1.9])
        with col_presets:
            preset_options = ["-- Select Preset Compound --"] + list(PRESET_MOLECULES.keys())
            selected_preset = st.selectbox(
                "Reference Demonstration Compounds:",
                preset_options,
                index=0,
                help="Select an approved reference drug to load its canonical SMILES automatically.",
            )
            default_smiles = (
                PRESET_MOLECULES[selected_preset]
                if selected_preset != "-- Select Preset Compound --"
                else "CC(=O)Oc1ccccc1C(=O)O"
            )

        with col_custom:
            smiles_input = st.text_input(
                "SMILES Structure String:",
                value=default_smiles,
                help="Simplified Molecular Input Line Entry System (e.g. Aspirin: CC(=O)Oc1ccccc1C(=O)O)",
            )

        col_btn, col_info = st.columns([1, 2.8])
        with col_btn:
            run_analysis = st.button("Run Full Analysis", type="primary", use_container_width=True)
        with col_info:
            st.caption("ℹ️ Computes Lipinski descriptors, ClinTox probability, Tox21 assays, ChEMBL analogues, and TreeSHAP attributions.")
        st.markdown("""</div>""", unsafe_allow_html=True)

    clean_smiles = smiles_input.strip() if smiles_input else ""

    # Gated Execution & State Preservation
    if run_analysis:
        if not clean_smiles:
            st.session_state.pop("single_mol_result", None)
            st.session_state.pop("single_mol_analyzed_smiles", None)
            st.warning("⚠️ Please enter a valid SMILES string before running analysis.")
            return

        with st.spinner("Executing structure validation, feature extraction, ClinTox inference, Tox21 assays, ChEMBL search, and TreeSHAP explainability..."):
            res = orchestrator.analyze_single_molecule(clean_smiles, compute_explanation=True)

        if not res.is_valid:
            st.session_state.pop("single_mol_result", None)
            st.session_state.pop("single_mol_analyzed_smiles", None)
            st.error(f"❌ Chemical Structure Validation Failure: {res.error_message}")
            with st.expander("▸ Validation Advice"):
                st.markdown(
                    """
                    - Verify atom valences (e.g., carbon has 4 bonds, nitrogen has 3 or 4, oxygen has 2).
                    - Ensure all branch parentheses `()` and ring closure digits `1`, `2` are properly balanced.
                    - Try selecting an approved reference compound from the dropdown above.
                    """
                )
            return

        st.session_state["single_mol_result"] = res
        st.session_state["single_mol_analyzed_smiles"] = clean_smiles

    # Retrieve cached analysis result from session state if available
    saved_res = st.session_state.get("single_mol_result")
    analyzed_smiles = st.session_state.get("single_mol_analyzed_smiles")

    if saved_res is not None and analyzed_smiles is not None:
        if clean_smiles != analyzed_smiles:
            st.info("ℹ️ Molecular input has been modified. Click **'Run Full Analysis'** to profile this compound.")
            return
        res = saved_res
    else:
        render_empty_state()
        return

    # =========================================================================
    # MOLECULAR HERO (Focal Visual Area)
    # =========================================================================
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    col_hero_img, col_hero_id = st.columns([1.15, 1.25], gap="medium")

    with col_hero_img:
        view_mode = st.radio(
            "Structure View Selection:",
            ["Interactive 3D Conformer", "2D Connectivity Structure"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        if view_mode == "Interactive 3D Conformer":
            mol_block = generate_3d_mol_block(res.validation.canonical_smiles)
            if mol_block:
                st.markdown('<div class="mol-3d-canvas-wrapper">', unsafe_allow_html=True)
                render_3dmol_viewer(mol_block, height=295)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(
                    """<div style="text-align: center; margin-top: 0.4rem; font-size: 0.72rem; color: #64748B; line-height: 1.35;">
                    <strong>Interactive 3D Conformer:</strong> Drag to rotate &bull; Scroll to zoom &bull; Right-click to pan.<br>
                    <em>Scientific Notice: Computationally generated conformer (RDKit ETKDGv3 + MMFF94 force field). Energy-minimized in vacuo geometry; does not represent an experimentally confirmed in vivo biological binding conformation.</em>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                svg_data = render_molecule_svg(res.validation.canonical_smiles, width=420, height=270)
                if svg_data:
                    st.markdown(f'<div class="mol-structure-canvas">{svg_data}</div>', unsafe_allow_html=True)
                    st.markdown(
                        """<div style="text-align: center; margin-top: 0.35rem; font-size: 0.74rem; color: #64748B;">RDKit 2D Coordinate Vector Depiction (3D conformer embedding unavailable for this structure)</div>""",
                        unsafe_allow_html=True,
                    )
        else:
            svg_data = render_molecule_svg(res.validation.canonical_smiles, width=420, height=270)
            if svg_data:
                st.markdown(f'<div class="mol-structure-canvas">{svg_data}</div>', unsafe_allow_html=True)
                st.markdown(
                    """<div style="text-align: center; margin-top: 0.35rem; font-size: 0.74rem; color: #64748B;">RDKit 2D Coordinate Vector Depiction</div>""",
                    unsafe_allow_html=True,
                )

    with col_hero_id:
        d = res.descriptors
        mw_str = f"{d.molecular_weight:.2f} Da" if d else "N/A"

        # RDKit Heavy Atom count from actual validated molecule
        val = res.validation
        num_heavy_atoms = None
        if val and val.num_heavy_atoms is not None:
            num_heavy_atoms = val.num_heavy_atoms
        elif val and val.canonical_smiles:
            mol = Chem.MolFromSmiles(val.canonical_smiles)
            if mol is not None:
                num_heavy_atoms = mol.GetNumHeavyAtoms()
        elif d and hasattr(d, "num_heavy_atoms"):
            num_heavy_atoms = getattr(d, "num_heavy_atoms")

        heavy_str = str(num_heavy_atoms) if num_heavy_atoms is not None else "N/A"
        lip_badge = "mol-badge-success" if d and d.lipinski_pass else "mol-badge-danger"
        lip_text = "Lipinski Pass (0 Violations)" if d and d.lipinski_pass else f"Lipinski Fail ({d.lipinski_violations} Violations)"

        st.markdown(
            f"""
            <div style="padding: 0.25rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                    <span style="font-size: 0.74rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Structure Verified</span>
                    <span class="mol-badge {lip_badge}">{lip_text}</span>
                </div>
                <h2 style="margin: 0 0 0.25rem 0; font-size: 2.15rem; color: #101B2D; font-weight: 800; letter-spacing: -0.02em;">
                    {res.validation.formula}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Canonical SMILES (Click icon to copy):")
        st.code(res.validation.canonical_smiles, language="text")

        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #D9E2EF; font-size: 0.82rem; margin-top: 0.5rem;">
                <div><span style="color: #64748B;">Molecular Weight:</span><br><strong style="color: #101B2D; font-size: 0.95rem;">{mw_str}</strong></div>
                <div><span style="color: #64748B;">Heavy Atoms:</span><br><strong style="color: #101B2D; font-size: 0.95rem;">{heavy_str}</strong></div>
                <div><span style="color: #64748B;">LogP (Crippen):</span><br><strong style="color: #101B2D; font-size: 0.95rem;">{d.logp:.2f}</strong></div>
                <div><span style="color: #64748B;">Polar Surface Area:</span><br><strong style="color: #101B2D; font-size: 0.95rem;">{d.tpsa:.1f} Å²</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # WORKSPACE TABS
    # =========================================================================
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    tab_overview, tab_toxicology, tab_properties, tab_explain = st.tabs([
        "📊 Overview & Triage",
        "🛡️ Safety & Toxicology",
        "📐 Properties & Analogues",
        "💡 AI Explainability",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: OVERVIEW & TRIAGE (Anti-Card-Wall Editorial Design)
    # -------------------------------------------------------------------------
    with tab_overview:
        st.markdown("<h3 style='margin: 0 0 0.5rem 0; font-size: 1.25rem; color: #101B2D;'>Exploratory Triage Summary</h3>", unsafe_allow_html=True)

        # Render open editorial triage row (Zero Card Wall)
        render_triage_editorial_grid(res)

        # Executive Scientific Synthesis
        render_executive_interpretation(res)

        # Exploratory Triage Index Summary
        if res.prioritization and res.prioritization.is_complete:
            p = res.prioritization
            r_pct = max(0.0, min(100.0, p.overall_score))

            ad_status = getattr(res.clintox, "applicability_domain_status", "UNAVAILABLE") if res.clintox else "UNAVAILABLE"
            ad_score = getattr(res.clintox, "applicability_score", None) if res.clintox else None

            if ad_status == "OUTSIDE_DOMAIN":
                ad_badge_html = '<span class="mol-badge mol-badge-danger" style="font-size: 0.76rem;">OUTSIDE APPLICABILITY DOMAIN — PREDICTION MAY BE UNRELIABLE</span>'
            elif ad_status == "NEAR_BOUNDARY":
                score_info = f" (Tanimoto: {ad_score:.2f})" if ad_score is not None else ""
                ad_badge_html = f'<span class="mol-badge mol-badge-warning" style="font-size: 0.76rem;">NEAR APPLICABILITY DOMAIN BOUNDARY{score_info}</span>'
            elif ad_status == "WITHIN_DOMAIN":
                score_info = f" (Tanimoto: {ad_score:.2f} ≥ 0.40)" if ad_score is not None else ""
                ad_badge_html = f'<span class="mol-badge mol-badge-info" style="font-size: 0.76rem;">WITHIN APPLICABILITY DOMAIN{score_info}</span>'
            else:
                ad_badge_html = '<span class="mol-badge mol-badge-neutral" style="font-size: 0.76rem;">APPLICABILITY DOMAIN UNAVAILABLE</span>'

            st.markdown(
                f"""
                <div style="padding: 1rem 0; border-top: 1px solid #D9E2EF; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; flex-wrap: wrap; gap: 0.5rem;">
                        <div>
                            <span style="font-size: 0.74rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Exploratory Triage Index</span>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #27C7B8; letter-spacing: -0.02em;">
                                {p.overall_score:.1f} <span style="font-size: 1.05rem; color: #64748B; font-weight: 500;">/ 100</span>
                            </div>
                        </div>
                        <div>
                            {ad_badge_html}
                        </div>
                    </div>
                    <div style="width: 100%; height: 7px; background: #EDF2F7; border-radius: 4px; overflow: hidden; margin-bottom: 0.85rem;">
                        <div style="width: {r_pct}%; height: 100%; background: linear-gradient(90deg, #27C7B8, #1BA699); border-radius: 4px;"></div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; font-size: 0.82rem;">
                        <div><span style="color: #64748B;">Safety Component S (50%):</span> <strong style="color: {'#10B981' if p.safety_score >= 50 else '#EF4444'};">{p.safety_score:.1f}</strong></div>
                        <div><span style="color: #64748B;">Drug-Likeness Component D (30%):</span> <strong style="color: #27C7B8;">{p.druglikeness_score:.0f}</strong></div>
                        <div><span style="color: #64748B;">Reference Similarity M (20%):</span> <strong style="color: #8E7CFF;">{p.similarity_score:.1f}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ad_status == "OUTSIDE_DOMAIN":
                st.warning("⚠️ **Applicability Domain Warning:** OUTSIDE APPLICABILITY DOMAIN — PREDICTION MAY BE UNRELIABLE. Molecule chemotype or scaffold is unrepresented in training data (max Tanimoto < 0.25); predictions represent extrapolations.")
            elif ad_status == "NEAR_BOUNDARY":
                st.info("ℹ️ **Applicability Domain Advisory:** Molecule lies near the domain boundary (0.25 ≤ Tanimoto < 0.40). Prediction uncertainty is elevated due to few training analogues.")
            elif ad_status in ("UNAVAILABLE", "UNVALIDATED", "UNKNOWN"):
                st.info("ℹ️ **Applicability Domain Note:** Applicability domain status could not be computed for this structure.")

            with st.expander("📐 How is the Exploratory Triage Index calculated?"):
                st.markdown(
                    r"""
                    The **Exploratory Triage Index ($R$)** is a heuristic computational triage score unifying three screening dimensions into a standardized $0 \dots 100$ scale:

                    $$R = 0.5 \times S + 0.3 \times D + 0.2 \times M$$

                    - **Safety Component ($S = (1 - P_{\text{tox}}) \times 100$):** Primary weight ($50\%$) penalizes high clinical trial attrition risk based on calibrated ClinTox predictions.
                    - **Drug-Likeness Component ($D$):** $100$ if Lipinski compliant (0 violations), $40$ if violated ($30\%$ weight).
                    - **Reference Similarity Component ($M = 100 \times \text{Tanimoto}_{\max}$):** Topological proximity in Morgan fingerprint space to known approved small molecules in the reference library ($20\%$ weight).

                    *Scientific Notice: The Exploratory Triage Index is a heuristic research prioritization tool, not a validated measure of clinical safety, therapeutic efficacy, pharmacological drug-likeness, or probability of regulatory approval.*
                    """
                )

        # Scientific & Regulatory Disclaimer
        st.markdown(
            """
            <div style="background: #FFFFFF; border-left: 3px solid #E5A34D; border-top: 1px solid #D9E2EF; border-right: 1px solid #D9E2EF; border-bottom: 1px solid #D9E2EF; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 1rem; font-size: 0.8rem; color: #334155; line-height: 1.5; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                <strong>Scientific & Regulatory Disclaimer:</strong> Research use only. Model outputs and the Exploratory Triage Index are computational estimates, not determinations of clinical safety, efficacy, or suitability for human use. Applicability-domain warnings indicate structural distance from the model's training data and do not independently establish toxicity.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Next Research Actions
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #101B2D;'>Next Research Actions</h4>", unsafe_allow_html=True)
        act_col1, act_col2, act_col3 = st.columns(3)

        with act_col1:
            export_dict = build_single_molecule_export_dict(res)
            st.download_button(
                label="💾 Download JSON Record",
                data=json.dumps(export_dict, indent=2),
                file_name=f"moleculeiq_analysis_{res.validation.formula}.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Export full machine-readable profiling record.")

        with act_col2:
            if st.button("◫ Screen Libraries in Batch", use_container_width=True):
                st.session_state["nav_mode"] = "Batch Molecular Screening"
                st.rerun()
            st.caption("Transition to high-throughput batch screening mode.")

        with act_col3:
            if st.button("⌕ Search Literature for Compound", use_container_width=True):
                st.session_state["nav_mode"] = "Literature Intelligence"
                st.session_state["lit_query_prefill"] = f"{res.validation.formula} toxicity mechanism"
                st.rerun()
            st.caption("Retrieve PubMed peer-reviewed literature.")

    # -------------------------------------------------------------------------
    # TAB 2: SAFETY & TOXICOLOGY
    # -------------------------------------------------------------------------
    with tab_toxicology:
        st.markdown("<h3 style='margin: 0 0 0.5rem 0; font-size: 1.25rem; color: #101B2D;'>Clinical Trial Toxicity (ClinTox)</h3>", unsafe_allow_html=True)

        if res.clintox and res.clintox.status == "SUCCESS":
            p_tox = res.clintox.probability
            p_pct = p_tox * 100.0

            render_risk_gauge(p_tox, res.clintox.is_toxic, decision_threshold=0.50)

            uncal_str = ""
            if getattr(res.clintox, "uncalibrated_probability", None) is not None:
                uncal_str = f" &bull; Uncalibrated Tree Vote Index: {res.clintox.uncalibrated_probability*100:.1f}%"

            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #64748B; margin: 0.25rem 0 0.75rem 0;">
                    Calibrated ClinTox Probability: <strong style="color: #101B2D;">{p_pct:.1f}%</strong> {uncal_str} &bull; Decision Threshold: 50.0%
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Scientific Benchmark Interpretation Context
            st.markdown(
                """
                <div style="background: #FFFFFF; padding: 0.75rem 0.95rem; border-radius: 6px; border-left: 3px solid #27C7B8; border-top: 1px solid #D9E2EF; border-right: 1px solid #D9E2EF; border-bottom: 1px solid #D9E2EF; margin: 0.5rem 0; font-size: 0.84rem; color: #334155; line-height: 1.5; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                    <strong style="color: #0F8075;">Scientific Benchmark Context:</strong><br>
                    ClinTox models the probability of clinical trial failure attributed to toxicity (MoleculeNet benchmark).
                    A high prediction reflects clinical trial attrition risk in human cohorts (e.g. adverse gastrointestinal, hepatic, or cardiovascular termination criteria)
                    and does not imply that the compound is acutely toxic in everyday consumer contexts.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Applicability Domain Verification
            ad_stat = getattr(res.clintox, "applicability_domain_status", "WITHIN_DOMAIN")
            ad_score = getattr(res.clintox, "applicability_score", 1.0)
            ad_color = "#0F8075" if ad_stat == "WITHIN_DOMAIN" else ("#E5A34D" if ad_stat == "NEAR_BOUNDARY" else "#EF4444")
            ad_badge = "info" if ad_stat == "WITHIN_DOMAIN" else ("warning" if ad_stat == "NEAR_BOUNDARY" else "danger")

            st.markdown(
                f"""
                <div style="background: #FFFFFF; padding: 0.65rem 0.85rem; border-radius: 6px; border: 1px solid #D9E2EF; margin: 0.5rem 0; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                    <span style="font-size: 0.82rem; color: #475569;">Applicability Domain: <strong style="color: {ad_color};">{ad_stat}</strong> (Nearest Tanimoto: {ad_score:.2f} | Boundary: 0.40)</span>
                    <span class="mol-badge mol-badge-{ad_badge}">{ad_stat}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if getattr(res.clintox, "applicability_warning", None):
                st.warning(f"⚠️ {res.clintox.applicability_warning}")

            st.caption(f"Task Definition: `{res.clintox.task_definition}`")

        st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin: 0 0 0.5rem 0; font-size: 1.25rem; color: #101B2D;'>Tox21 12-Pathway Assay Activation</h3>", unsafe_allow_html=True)

        if res.tox21 and res.tox21.status == "SUCCESS":
            t21 = res.tox21
            active_cnt = t21.active_endpoints_count

            st.markdown(
                f"""
                <div style="font-size: 0.9rem; color: #334155; margin-bottom: 0.75rem;">
                    Active In-Vitro Pathways: <strong style="color: {'#10B981' if active_cnt == 0 else ('#E5A34D' if active_cnt <= 2 else '#EF4444')};">{active_cnt} / {t21.total_endpoints}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_nr, col_sr = st.columns(2, gap="medium")

            with col_nr:
                st.markdown("<h4 style='margin: 0 0 0.5rem 0; font-size: 1rem; color: #0F8075;'>Nuclear Receptor (NR) Pathways</h4>", unsafe_allow_html=True)
                nr_endpoints = [ep for ep in t21.endpoints.keys() if ep.startswith("NR")]
                for ep in nr_endpoints:
                    pred = t21.endpoints[ep]
                    meta = TOX21_METADATA.get(ep, {"name": ep, "desc": "Nuclear receptor signaling"})
                    badge_type = "danger" if pred.is_active else "neutral"
                    status_text = "Active" if pred.is_active else "Inactive"

                    st.markdown(
                        f"""
                        <div style="background: #FFFFFF; border: 1px solid #D9E2EF; border-radius: 6px; padding: 0.45rem 0.65rem; margin-bottom: 0.35rem; box-shadow: 0 1px 2px rgba(16, 27, 45, 0.03);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="font-size: 0.84rem; color: #101B2D;">{ep} &bull; {meta['name']}</strong>
                                <span class="mol-badge mol-badge-{badge_type}" style="font-size: 0.7rem;">{status_text} ({pred.probability*100:.0f}%)</span>
                            </div>
                            <div style="font-size: 0.74rem; color: #64748B; margin-top: 0.15rem;">{meta['desc']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_sr:
                st.markdown("<h4 style='margin: 0 0 0.5rem 0; font-size: 1rem; color: #5B48DF;'>Cellular Stress Response (SR) Pathways</h4>", unsafe_allow_html=True)
                sr_endpoints = [ep for ep in t21.endpoints.keys() if not ep.startswith("NR")]
                for ep in sr_endpoints:
                    pred = t21.endpoints[ep]
                    meta = TOX21_METADATA.get(ep, {"name": ep, "desc": "Cellular stress pathway"})
                    badge_type = "danger" if pred.is_active else "neutral"
                    status_text = "Active" if pred.is_active else "Inactive"

                    st.markdown(
                        f"""
                        <div style="background: #FFFFFF; border: 1px solid #D9E2EF; border-radius: 6px; padding: 0.45rem 0.65rem; margin-bottom: 0.35rem; box-shadow: 0 1px 2px rgba(16, 27, 45, 0.03);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="font-size: 0.84rem; color: #101B2D;">{ep} &bull; {meta['name']}</strong>
                                <span class="mol-badge mol-badge-{badge_type}" style="font-size: 0.7rem;">{status_text} ({pred.probability*100:.0f}%)</span>
                            </div>
                            <div style="font-size: 0.74rem; color: #64748B; margin-top: 0.15rem;">{meta['desc']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("📋 View Complete 12-Assay Endpoint Table"):
                ep_rows = build_tox21_endpoint_table_rows(t21)
                st.dataframe(pd.DataFrame(ep_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "ℹ️ **Calibration & Applicability Domain (AD) Note:** Default screening uses raw uncalibrated RF probabilities. "
                    "Calibrated probabilities are selected via 5-fold 3-way disjoint training cross-validation and fitted on 5-fold stratified "
                    "out-of-fold training predictions (Platt sigmoid); they reflect empirical in-vitro assay frequency under the 80/20 split, "
                    "not clinical risk. AD scores reflect maximum Tanimoto similarity against the endpoint's isolated training corpus "
                    "(Morgan r=2, 2048 bits; unvalidated thresholds)."
                )

    # -------------------------------------------------------------------------
    # TAB 3: PROPERTIES & ANALOGUES
    # -------------------------------------------------------------------------
    with tab_properties:
        st.markdown("<h3 style='margin: 0 0 0.5rem 0; font-size: 1.25rem; color: #f8fafc;'>Physicochemical Profile & Lipinski Rules</h3>", unsafe_allow_html=True)

        if res.descriptors:
            d = res.descriptors
            p_col1, p_col2 = st.columns([1.2, 1], gap="medium")

            with p_col1:
                render_property_bar("Molecular Weight (MW)", f"{d.molecular_weight:.1f} Da", min(1.0, d.molecular_weight / 500.0), "<= 500 Da", d.molecular_weight <= 500.0)
                render_property_bar("LogP (Crippen Lipophilicity)", f"{d.logp:.2f}", min(1.0, max(0.05, (d.logp + 2.0) / 7.0)), "<= 5.0", d.logp <= 5.0)
                render_property_bar("Hydrogen Bond Donors (HBD)", f"{d.hbond_donors}", min(1.0, d.hbond_donors / 5.0), "<= 5", d.hbond_donors <= 5)
                render_property_bar("Hydrogen Bond Acceptors (HBA)", f"{d.hbond_acceptors}", min(1.0, d.hbond_acceptors / 10.0), "<= 10", d.hbond_acceptors <= 10)
                render_property_bar("Rotatable Bonds", f"{d.rotatable_bonds}", min(1.0, d.rotatable_bonds / 10.0), "<= 10", d.rotatable_bonds <= 10)
                render_property_bar("Topological Polar Surface Area (TPSA)", f"{d.tpsa:.1f} Å²", min(1.0, d.tpsa / 140.0), "Optimal < 140 Å²", d.tpsa <= 140.0)

            with p_col2:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #D9E2EF; border-radius: 10px; padding: 1.15rem; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <h4 style="margin: 0; font-size: 1.05rem; color: #101B2D;">Lipinski Compliance</h4>
                            <span class="mol-badge {'mol-badge-success' if d.lipinski_pass else 'mol-badge-danger'}">
                                {'Compliant' if d.lipinski_pass else f'{d.lipinski_violations} Violations'}
                            </span>
                        </div>
                        <p style="font-size: 0.88rem; color: #334155; line-height: 1.55; margin: 0 0 0.75rem 0;">
                            {'The compound satisfies all criteria of Lipinski’s Rule of 5, meeting standard physicochemical threshold ranges for oral drug-likeness heuristics.' if d.lipinski_pass else 'The compound violates one or more Lipinski rules, exceeding standard physicochemical threshold ranges for oral absorption heuristics.'}
                        </p>
                        <div style="font-size: 0.78rem; color: #64748B; line-height: 1.45; border-top: 1px solid #D9E2EF; padding-top: 0.65rem;">
                            <strong style="color: #334155;">Cheminformatics Note:</strong> HBA count is calculated using RDKit's physiological Lipinski rule, strictly excluding resonance-delocalized amide nitrogens.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("📊 Complete Tabular Descriptors"):
                lip_status = "Pass" if d.lipinski_pass else f"Fail ({d.lipinski_violations} violations)"
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

        st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin: 0 0 0.5rem 0; font-size: 1.25rem; color: #101B2D;'>Chemical Neighborhood (ChEMBL Library)</h3>", unsafe_allow_html=True)

        if res.similarity and res.similarity.status == "SUCCESS":
            matches = res.similarity.matches
            if matches:
                top_match = matches[0]
                sim_tier = (
                    "VERY HIGH (Identical / Close Analogue)"
                    if top_match.similarity >= 0.70
                    else ("HIGH SIMILARITY" if top_match.similarity >= 0.50 else ("MODERATE SIMILARITY" if top_match.similarity >= 0.30 else "DISTANT"))
                )
                sim_badge = "success" if top_match.similarity >= 0.70 else ("warning" if top_match.similarity >= 0.40 else "neutral")

                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border-left: 3px solid #8E7CFF; border-top: 1px solid #D9E2EF; border-right: 1px solid #D9E2EF; border-bottom: 1px solid #D9E2EF; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                            <span style="font-size: 0.76rem; color: #5B48DF; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Nearest Approved Reference Drug</span>
                            <span class="mol-badge mol-badge-{sim_badge}">{sim_tier}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">
                            <h3 style="margin: 0; font-size: 1.35rem; color: #101B2D;">
                                {top_match.name} <span style="font-size: 0.85rem; color: #64748B; font-weight: 400;">({top_match.reference_id})</span>
                            </h3>
                            <div style="font-size: 1.45rem; font-weight: 800; color: #5B48DF;">
                                Tanimoto: {top_match.similarity:.4f}
                            </div>
                        </div>
                        <div style="font-size: 0.82rem; color: #475569; margin-top: 0.35rem;">
                            MW: <strong>{top_match.molecular_weight:.1f} Da</strong> &bull; Lipinski: <strong>{'Pass' if top_match.lipinski_pass else 'Fail'}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "⚠️ **Scientific Principle:** Structural similarity indicates chemotype proximity in Morgan fingerprint space, "
                    "but does NOT guarantee biological equivalence, clinical efficacy, or safety."
                )

                st.markdown(
                    """
                    <div style="background: #FFFFFF; border: 1px solid #D9E2EF; border-radius: 6px; padding: 0.65rem 0.85rem; margin: 0.5rem 0; font-size: 0.8rem; color: #475569; line-height: 1.5; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                        <strong style="color: #101B2D;">Reference Library Provenance & Methodological Disclosure:</strong><br>
                        • <strong>Composition:</strong> The reference library contains <strong>1,396</strong> small-molecule records, comprising <strong>1,384</strong> records derived from the ClinTox FDA-approved slice and <strong>12</strong> ChEMBL curated anchor compounds.<br>
                        • <strong>Structural Resemblance:</strong> Tanimoto similarity (Morgan radius 2, 2048 bits) measures topological resemblance in fingerprint space; it does <em>not</em> imply shared biological mechanism, clinical efficacy, or safety profile.<br>
                        • <strong>Self-Match Notice:</strong> Query molecules that exist within the reference library will produce an exact self-match (Tanimoto = 1.0000).<br>
                        • <strong>Evaluation Boundary:</strong> Retrospective similarity evaluations against this library share underlying compounds with the ClinTox benchmark, which limits the independence of retrospective validation.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("📚 View All Matched Approved Reference Analogues"):
                    sim_rows = []
                    for m in matches:
                        sim_rows.append({
                            "Reference ID": m.reference_id,
                            "Approved Drug Name": m.name,
                            "Tanimoto Similarity": f"{m.similarity:.4f}",
                            "Molecular Weight": f"{m.molecular_weight:.1f} Da" if m.molecular_weight else "N/A",
                            "Lipinski Status": "Pass" if m.lipinski_pass else "Fail",
                            "Canonical Structure": m.canonical_smiles,
                        })
                    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 4: AI EXPLAINABILITY
    # -------------------------------------------------------------------------
    with tab_explain:
        st.markdown("<h3 style='margin: 0 0 0.5rem 0; font-size: 1.25rem; color: #101B2D;'>Why did the model predict this?</h3>", unsafe_allow_html=True)

        if res.explanation and res.explanation.status == "SUCCESS":
            x_col1, x_col2 = st.columns([1, 1.2], gap="medium")

            with x_col1:
                st.markdown("<h4 style='margin: 0 0 0.4rem 0; font-size: 1rem; color: #101B2D;'>Highlighted Atomic Environments</h4>", unsafe_allow_html=True)
                st.caption("Atoms and radius bonds contributing most significantly to the model decision:")
                if res.explanation.highlighted_svg:
                    st.markdown(f"""<div class="mol-structure-canvas">{res.explanation.highlighted_svg}</div>""", unsafe_allow_html=True)

            with x_col2:
                st.markdown("<h4 style='margin: 0 0 0.4rem 0; font-size: 1rem; color: #101B2D;'>Directional Fingerprint Influences</h4>", unsafe_allow_html=True)
                st.caption("Active Morgan fingerprint bits and their directional impact on predicted ClinTox probability. Attributions describe contributions to the model prediction; they do not establish causal biological effects:")

                for f in res.explanation.top_features[:5]:
                    is_risk_increasing = f.shap_value > 0
                    bar_color = "#EF4444" if is_risk_increasing else "#10B981"
                    impact_label = "↑ Raises Predicted ClinTox Probability" if is_risk_increasing else "↓ Lowers Predicted ClinTox Probability"
                    badge_type = "danger" if is_risk_increasing else "success"
                    st.markdown(
                        f"""
                        <div style="background: #FFFFFF; border: 1px solid #D9E2EF; border-radius: 6px; padding: 0.45rem 0.65rem; margin-bottom: 0.4rem; box-shadow: 0 1px 2px rgba(16, 27, 45, 0.03);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.84rem; color: #101B2D; font-weight: 600;">Morgan Bit #{f.bit_id} <span style="font-size: 0.74rem; color: #64748B;">(Radius: {f.radius})</span></span>
                                <span class="mol-badge mol-badge-{badge_type}" style="font-size: 0.7rem;">{impact_label}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748B; margin-top: 0.2rem;">
                                <span>Attribution: <strong style="color: {bar_color};">{f.shap_value:+.5f}</strong></span>
                                <span>Atoms: {f.atom_indices}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("🔬 Complete Morgan Bit Feature Attribution Table"):
                feat_rows = []
                for f in res.explanation.top_features:
                    feat_rows.append({
                        "Morgan Bit ID": f.bit_id,
                        "SHAP Attribution": f"{f.shap_value:+.5f}",
                        "Prediction Impact": "Raises Predicted ClinTox Probability" if f.shap_value > 0 else "Lowers Predicted ClinTox Probability",
                        "Radius (Bonds)": f.radius,
                        "Atoms Involved": str(f.atom_indices),
                    })
                st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

            if hasattr(res.explanation, "disclaimer") and res.explanation.disclaimer:
                st.caption(f"ℹ️ {res.explanation.disclaimer}")
