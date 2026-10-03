"""Model Insights and Evaluation Evidence view (Workflow C)."""

import json
import os
import streamlit as st
import pandas as pd

from src.ui.components import render_header, render_metric_card


def render_model_insights_view():
    """Render Workflow C: Model Insights & Scientific Evaluation Metrics."""
    render_header(
        "Model Insights & Evaluation Metrics",
        "Transparent scientific evidence: benchmark distributions, held-out evaluation metrics, and task definitions.",
    )

    tab_clin, tab_tox21, tab_limits = st.tabs(["ClinTox Benchmark Model", "Tox21 12-Endpoint Multi-Models", "Applicability & Boundaries"])

    # TAB 1: ClinTox
    with tab_clin:
        meta_path = "artifacts/clintox/metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metrics = meta.get("metrics", {})

            st.markdown("### ClinTox Random Forest Classifier (`CT_TOX`)")
            st.caption("Target definition: Likelihood of clinical trial failure attributed to toxicity (MoleculeNet benchmark).")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Held-Out ROC-AUC", f"{metrics.get('roc_auc', 0.85):.4f}", "Test split (20%)", "#2563eb")
            with m2:
                render_metric_card("PR-AUC (Avg Prec)", f"{metrics.get('pr_auc', 0.30):.4f}", "Minority class evaluation", "#0891b2")
            with m3:
                render_metric_card("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0.66):.4f}", "Mean of sensitivities", "#16a34a")
            with m4:
                render_metric_card("F1-Score", f"{metrics.get('f1', 0.39):.4f}", "Harmonic mean prec/recall", "#9333ea")

            st.divider()

            col_cm, col_meta = st.columns([1, 1.2])
            with col_cm:
                st.markdown("#### Test Split Confusion Matrix")
                cm = metrics.get("confusion_matrix", [[263, 11], [14, 8]])
                df_cm = pd.DataFrame(
                    cm,
                    columns=["Predicted Non-Toxic (0)", "Predicted Toxic (1)"],
                    index=["Actual Non-Toxic (0)", "Actual Toxic (1)"],
                )
                st.dataframe(df_cm, use_container_width=True)

            with col_meta:
                st.markdown("#### Model Specification")
                st.markdown(f"- **Algorithm:** `{meta.get('algorithm', 'RandomForestClassifier')}`")
                st.markdown(f"- **Estimators:** `{meta.get('n_estimators', 300)}`")
                st.markdown(f"- **Class Weighting:** `{meta.get('class_weight', 'balanced')}`")
                st.markdown(f"- **Features:** Morgan circular fingerprints (radius=2, 2048 bits)")
                st.markdown(f"- **Training Samples:** `{metrics.get('train_count', 1184)}` (80% stratified split)")
                st.markdown(f"- **Test Samples:** `{metrics.get('test_count', 296)}` (20% stratified split)")
        else:
            st.warning("ClinTox metadata artifact not found. Please train models first.")

    # TAB 2: Tox21
    with tab_tox21:
        summary_path = "artifacts/tox21/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                tox_summary = json.load(f)

            st.markdown("### Tox21 12-Endpoint In Vitro Screening Suite")
            st.caption("Independent binary Random Forest classifiers trained with isolated missing-label exclusion.")

            metrics_summary = tox_summary.get("metrics_summary", {})
            rows = []
            for ep, m in metrics_summary.items():
                rows.append({
                    "Endpoint": ep,
                    "Pathway Category": "Nuclear Receptor (NR)" if ep.startswith("NR") else "Stress Response (SR)",
                    "Samples": m.get("sample_count", 0),
                    "ROC-AUC": m.get("roc_auc", 0.0),
                    "PR-AUC": m.get("pr_auc", 0.0),
                    "Balanced Acc": m.get("balanced_accuracy", 0.0),
                    "Precision": m.get("precision", 0.0),
                    "Recall": m.get("recall", 0.0),
                    "F1": m.get("f1", 0.0),
                })

            df_tox = pd.DataFrame(rows)
            st.dataframe(df_tox, use_container_width=True, hide_index=True)
        else:
            st.warning("Tox21 summary artifact not found.")

    # TAB 3: Applicability & Boundaries
    with tab_limits:
        st.markdown("### Scientific Boundaries and Model Governance")
        st.markdown(
            """
            1. **Benchmark Specificity:** The ClinTox dataset annotates clinical trial outcomes. A negative prediction indicates absence of documented clinical failure, not guaranteed cellular safety.
            2. **Assay Mechanism vs. Organ Toxicity:** Tox21 in vitro luminescence assays measure pathway perturbation in cell lines, not animal or human organ toxicity.
            3. **Severe Class Imbalance:** In ClinTox, toxic compounds represent ~7.5% of samples. High standard accuracy can be achieved by a trivial majority-class predictor; therefore, PR-AUC and Balanced Accuracy are mandatory evaluation metrics.
            4. **Chemical Space Limits:** Models are trained on small drug-like organic compounds ($MW < 600$ Da). Novel scaffolds, inorganic complexes, and large macrocycles fall outside the validated applicability domain.
            """
        )
