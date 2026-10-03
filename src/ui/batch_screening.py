"""Batch Molecular Screening view (Workflow B)."""

import io
import streamlit as st
import pandas as pd

from src.services.orchestration import ScreeningOrchestrator
from src.ui.components import render_header, render_metric_card

SAMPLE_BATCH_CSV = """name,smiles
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Paracetamol,CC(=O)Nc1ccc(O)cc1
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Invalid_Compound_1,NOT_A_VALID_SMILES_STRING
Ibuprofen,CC(C)Cc1ccc(C(C)C(=O)O)cc1
Atorvastatin,CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O
Invalid_Compound_2,C1CC
Metformin,CN(C)C(=N)NC(=N)N
Sunset_Yellow,O=S(=O)(O)c1ccc(/N=N/c2c(O)ccc3cc(S(=O)(=O)O)ccc23)cc1
"""


def render_batch_screening_view(orchestrator: ScreeningOrchestrator):
    """Render Workflow B: Batch Molecular Screening."""
    render_header(
        "Batch Molecular Screening",
        "Upload compound libraries (CSV) for automated parallel profiling, triage ranking, and CSV export.",
    )

    st.subheader("1. Data Ingestion")
    col_upload, col_sample = st.columns([2, 1])

    with col_sample:
        st.markdown("**Sample Test Dataset**")
        st.download_button(
            label="Download Demo Batch CSV",
            data=SAMPLE_BATCH_CSV,
            file_name="moleculeiq_demo_batch.csv",
            mime="text/csv",
            help="Download a pre-formatted demonstration CSV containing valid and invalid compounds.",
        )
        load_demo = st.button("Load Demo Batch in Memory", use_container_width=True)

    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV Compound Library:", type=["csv"])

    df_input = None
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(df_input)} rows and columns: {list(df_input.columns)}")
        except Exception as exc:
            st.error(f"Failed to read CSV file: {exc}")
            return
    elif load_demo:
        df_input = pd.read_csv(io.StringIO(SAMPLE_BATCH_CSV))
        st.info(f"Loaded demo dataset with {len(df_input)} rows.")

    if df_input is not None:
        col_smiles_col, col_name_col, col_run = st.columns([1.5, 1.5, 1])
        with col_smiles_col:
            candidate_smiles_cols = [c for c in df_input.columns if "smiles" in c.lower()]
            default_smiles_col = candidate_smiles_cols[0] if candidate_smiles_cols else df_input.columns[0]
            smiles_col = st.selectbox("SMILES Structure Column:", df_input.columns, index=list(df_input.columns).index(default_smiles_col))

        with col_name_col:
            name_cols = [c for c in df_input.columns if "name" in c.lower() or "id" in c.lower()]
            default_name_col = name_cols[0] if name_cols else df_input.columns[0]
            name_col = st.selectbox("Identifier / Name Column (Optional):", ["-- None (Auto-Generate) --"] + list(df_input.columns))

        with col_run:
            st.write("")
            st.write("")
            start_btn = st.button("Start Batch Screening", type="primary", use_container_width=True)

        if start_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            smiles_list = df_input[smiles_col].astype(str).tolist()
            names_list = None
            if name_col != "-- None (Auto-Generate) --":
                names_list = df_input[name_col].astype(str).tolist()

            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Processing candidate {current} of {total}...")

            batch_res = orchestrator.screen_batch(smiles_list, names_list, progress_callback=update_progress)
            progress_bar.progress(1.0)
            status_text.text("Batch screening completed successfully!")

            st.divider()

            # Batch Summary Metrics
            st.markdown("### 2. Screening Summary & Fault Isolation")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Total Uploaded", str(batch_res.total_records), "Input rows processed")
            with m2:
                render_metric_card("Valid Compounds", str(batch_res.valid_records), "Passed RDKit validation", "#16a34a")
            with m3:
                render_metric_card("Invalid Records", str(batch_res.invalid_records), "Isolated without crashing", "#dc2626" if batch_res.invalid_records > 0 else "#64748b")
            with m4:
                top_score = batch_res.results_dataframe["Overall_Priority_Score"].dropna().max()
                render_metric_card("Top Priority Score", f"{top_score:.1f}" if pd.notna(top_score) else "N/A", "Highest ranked candidate", "#2563eb")

            st.divider()

            # Ranked Results Table
            st.markdown("### 3. Ranked Candidate Prioritization Results")
            st.caption("Candidates are sorted in descending order of composite priority score (R = 0.5S + 0.3D + 0.2M). Invalid rows are isolated at bottom.")

            df_display = batch_res.results_dataframe
            st.dataframe(
                df_display[[
                    "Rank", "Name", "Status", "Overall_Priority_Score", "Safety_Score",
                    "Druglikeness_Score", "Similarity_Score", "Top_Reference_Drug",
                    "MW", "LogP", "Lipinski_Pass", "ClinTox_Toxic", "Tox21_Active_Assays", "Canonical_SMILES"
                ]],
                use_container_width=True,
                hide_index=True,
            )

            # Export Button
            csv_export = batch_res.results_dataframe.to_csv(index=False)
            st.download_button(
                label="Download Full Enriched Screening Results (CSV)",
                data=csv_export,
                file_name="moleculeiq_screening_results.csv",
                mime="text/csv",
                type="primary",
            )

            # Fault Isolation Inspection
            if batch_res.invalid_records > 0:
                with st.expander(f"Inspect {batch_res.invalid_records} Invalid Structure Entries"):
                    invalid_df = df_display[df_display["Status"] == "Invalid"][["Row_ID", "Name", "Original_SMILES", "Error"]]
                    st.dataframe(invalid_df, use_container_width=True, hide_index=True)
