"""Batch Molecular Screening view (Workflow B) — High-Throughput Compound Triage."""

import io
import streamlit as st
import pandas as pd

from src.services.orchestration import ScreeningOrchestrator
from src.ui.components import render_header, render_metric_card
from src.utils.security import sanitize_dataframe_for_csv, MAX_BATCH_ROWS

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
        badge_text="High-Throughput Mode",
        badge_variant="purple",
    )

    # Ingestion & Library Configuration
    st.markdown("### Compound Library Ingestion")

    with st.container():
        col_upload, col_sample = st.columns([2, 1.2])

        with col_sample:
            st.markdown(
                """
                <div style="background: #FFFFFF; padding: 0.85rem 1rem; border-radius: 8px; border: 1px solid #D9E2EF; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                    <span style="font-size: 0.82rem; color: #101B2D; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;">Reference Demonstration Library</span>
                    <p style="font-size: 0.8rem; color: #64748B; margin: 0.3rem 0 0.6rem 0; line-height: 1.45;">Test automated fault isolation with a curated CSV containing valid drugs & malformed rows.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="📥 Download Demo Batch CSV",
                data=SAMPLE_BATCH_CSV,
                file_name="moleculeiq_demo_batch.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download pre-formatted demonstration CSV containing both valid and invalid compounds.",
            )
            load_demo = st.button("⚡ Load Demo Batch in Memory", use_container_width=True)

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Compound Library (CSV):",
                type=["csv"],
                help=f"Accepts CSV with SMILES string column. Max allowed rows: {MAX_BATCH_ROWS}.",
            )

    # State Persistence for Ingested Datasets
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            if len(df_uploaded) > MAX_BATCH_ROWS:
                st.error(
                    f"Uploaded file contains {len(df_uploaded)} rows, exceeding the maximum safe limit "
                    f"of {MAX_BATCH_ROWS} compounds. Please upload a smaller library batch."
                )
                return
            st.session_state["batch_input_df"] = df_uploaded
            st.session_state["batch_input_source"] = f"Uploaded File: {uploaded_file.name}"
            st.session_state.pop("batch_screening_result", None)
        except Exception as exc:
            st.error(f"Failed to read CSV file: {exc}")
            return
    elif load_demo:
        df_demo = pd.read_csv(io.StringIO(SAMPLE_BATCH_CSV))
        st.session_state["batch_input_df"] = df_demo
        st.session_state["batch_input_source"] = "Curated Reference Demo Batch (9 compounds)"
        st.session_state.pop("batch_screening_result", None)

    df_input = st.session_state.get("batch_input_df")
    input_source = st.session_state.get("batch_input_source", "Active Dataset")

    if df_input is not None:
        with st.container():
            col_info, col_reset = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #D9E2EF; border-radius: 6px; padding: 0.5rem 0.85rem; font-size: 0.88rem; color: #334155; margin-bottom: 0.5rem; box-shadow: 0 1px 2px rgba(16, 27, 45, 0.03);">
                        ✓ Active Dataset: <strong style="color: #0F8075;">{input_source}</strong> &nbsp;|&nbsp;
                        Total Compounds: <strong style="color: #10B981;">{len(df_input)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_reset:
                if st.button("✕ Reset Ingested Library", use_container_width=True):
                    st.session_state.pop("batch_input_df", None)
                    st.session_state.pop("batch_input_source", None)
                    st.session_state.pop("batch_screening_result", None)
                    st.rerun()

            col_smiles_col, col_name_col, col_run = st.columns([1.5, 1.5, 1.2])

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
            st.session_state["batch_screening_result"] = batch_res

    # Retrieve cached batch results
    batch_res = st.session_state.get("batch_screening_result")

    if batch_res is not None:
        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Screening Telemetry & Fault Isolation")

        # Summary Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            render_metric_card("Total Ingested", str(batch_res.total_records), "Input rows processed")
        with m2:
            render_metric_card("Valid Compounds", str(batch_res.valid_records), "Passed RDKit validation", "#10b981", badge="Passed", badge_variant="success")
        with m3:
            render_metric_card(
                "Invalid Records",
                str(batch_res.invalid_records),
                "Isolated without crashing",
                "#ef4444" if batch_res.invalid_records > 0 else "#64748b",
                badge=f"{batch_res.invalid_records} Errors" if batch_res.invalid_records > 0 else "0 Errors",
                badge_variant="danger" if batch_res.invalid_records > 0 else "neutral",
            )
        with m4:
            top_score = batch_res.results_dataframe["Overall_Priority_Score"].dropna().max()
            render_metric_card(
                "Top Triage Index (R)",
                f"{top_score:.1f}" if pd.notna(top_score) else "N/A",
                "Highest index in batch",
                "#27C7B8",
                badge="Top Index",
                badge_variant="info",
            )
        with m5:
            valid_df = batch_res.results_dataframe[batch_res.results_dataframe["Status"] == "Valid"]
            low_tox_count = len(valid_df[valid_df["ClinTox_Toxic"] == False]) if "ClinTox_Toxic" in valid_df.columns else 0
            render_metric_card(
                "Model-Predicted ClinTox-Negative",
                str(low_tox_count),
                "Estimated P(Tox) < 0.50",
                "#0F8075",
                badge="Predicted Negative",
                badge_variant="info",
            )

        # Batch Applicability Domain Alert
        outside_count = len(valid_df[valid_df["Applicability_Domain_Status"] == "OUTSIDE_DOMAIN"]) if "Applicability_Domain_Status" in valid_df.columns else 0
        unavail_count = len(valid_df[valid_df["Applicability_Domain_Status"].isin(["UNAVAILABLE", "UNVALIDATED", "UNKNOWN"])]) if "Applicability_Domain_Status" in valid_df.columns else 0

        if outside_count > 0 or unavail_count > 0:
            warn_parts = []
            if outside_count > 0:
                warn_parts.append(f"**{outside_count} candidate(s) OUTSIDE applicability domain** (predictions may be unreliable extrapolations)")
            if unavail_count > 0:
                warn_parts.append(f"**{unavail_count} candidate(s) with UNAVAILABLE domain status**")
            st.warning(f"⚠️ **Batch Applicability Domain Alert:** {' and '.join(warn_parts)}. Inspect the 'Applicability_Domain_Status' column before prioritizing candidates.")

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Ranked Candidate Prioritization")
        st.caption("Candidates are sorted in descending order of Exploratory Triage Index ($R = 0.5S + 0.3D + 0.2M$). Invalid rows are isolated at bottom.")

        # Interactive Filtering Toolbar
        with st.expander("🔍 Filter & Sieve Screening Results", expanded=False):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                min_score = st.slider("Minimum Triage Index (R):", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
            with f_col2:
                filter_lipinski = st.checkbox("Only Lipinski Compliant Compounds", value=False)
            with f_col3:
                filter_low_tox = st.checkbox("Only Model-Predicted ClinTox-Negative Compounds", value=False)

        # Apply filtering
        df_display = batch_res.results_dataframe.copy()
        if min_score > 0.0 and "Overall_Priority_Score" in df_display.columns:
            df_display = df_display[(df_display["Overall_Priority_Score"] >= min_score) | (df_display["Status"] == "Invalid")]
        if filter_lipinski and "Lipinski_Pass" in df_display.columns:
            df_display = df_display[(df_display["Lipinski_Pass"] == True) | (df_display["Status"] == "Invalid")]
        if filter_low_tox and "ClinTox_Toxic" in df_display.columns:
            df_display = df_display[(df_display["ClinTox_Toxic"] == False) | (df_display["Status"] == "Invalid")]

        if "Rank" in df_display.columns:
            df_display["Rank"] = df_display["Rank"].astype(str)

        st.dataframe(
            df_display[[
                "Rank", "Name", "Status", "Overall_Priority_Score", "Safety_Score",
                "Druglikeness_Score", "Similarity_Score", "Top_Reference_Drug",
                "MW", "LogP", "Lipinski_Pass", "ClinTox_Toxic",
                "Applicability_Domain_Status", "Applicability_Warning",
                "Tox21_Active_Assays", "Canonical_SMILES"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        # Scientific & Regulatory Disclaimer
        st.markdown(
            """
            <div style="background: #FFFFFF; border-left: 3px solid #E5A34D; border-top: 1px solid #D9E2EF; border-right: 1px solid #D9E2EF; border-bottom: 1px solid #D9E2EF; border-radius: 6px; padding: 0.65rem 0.95rem; margin: 0.75rem 0; font-size: 0.8rem; color: #334155; line-height: 1.5; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
                <strong>Scientific & Regulatory Disclaimer:</strong> Research use only. Model outputs and the Exploratory Triage Index are computational estimates, not determinations of clinical safety, efficacy, or suitability for human use. Applicability-domain warnings indicate structural distance from the model's training data and do not independently establish toxicity.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_dl, _ = st.columns([1.5, 2])
        with col_dl:
            # Export Button (Sanitized against formula injection CWE-1236)
            sanitized_df = sanitize_dataframe_for_csv(batch_res.results_dataframe)
            csv_export = sanitized_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Full Enriched Results (CSV)",
                data=csv_export,
                file_name="moleculeiq_screening_results.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

        # Fault Isolation Inspection
        if batch_res.invalid_records > 0:
            with st.expander(f"⚠️ Inspect {batch_res.invalid_records} Invalid Structure Entries (Fault-Isolated)"):
                invalid_df = batch_res.results_dataframe[batch_res.results_dataframe["Status"] == "Invalid"][["Row_ID", "Name", "Original_SMILES", "Error"]]
                st.dataframe(invalid_df, use_container_width=True, hide_index=True)
                st.caption("ℹ️ These compounds failed chemical parsing or valence sanitization and were safely isolated to prevent pipeline termination.")
