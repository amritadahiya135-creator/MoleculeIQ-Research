"""Literature Intelligence and Semantic Retrieval view (Workflow D)."""

import streamlit as st
import pandas as pd

from src.services.orchestration import ScreeningOrchestrator
from src.ui.components import render_header, render_metric_card

SAMPLE_QUERIES = [
    "aspirin cardiotoxicity",
    "caffeine adenosine receptor",
    "atorvastatin hepatotoxicity and liver safety",
    "metformin mitochondrial complex I",
    "paracetamol acetaminophen liver necrosis",
]


def render_literature_assistant_view(orchestrator: ScreeningOrchestrator):
    """Render Workflow D: Literature Assistant & Semantic Retrieval."""
    render_header(
        "Literature Intelligence Assistant",
        "Semantic biomedical retrieval connecting molecular queries with PubMed abstracts and cited statements.",
    )

    st.subheader("1. Scientific Query")

    col_q, col_opts = st.columns([3, 1])

    with col_opts:
        selected_sample = st.selectbox("Sample Research Topics:", ["-- Select Sample --"] + SAMPLE_QUERIES)
        max_articles = st.slider("Max PubMed Articles:", min_value=2, max_value=10, value=5)

    with col_q:
        default_query = selected_sample if selected_sample != "-- Select Sample --" else "aspirin cardiotoxicity"
        query_input = st.text_input("Enter research topic or chemical adverse event query:", value=default_query)

    search_btn = st.button("Retrieve & Rank Scientific Literature", type="primary")

    if search_btn and query_input:
        with st.spinner("Querying NCBI PubMed and ranking abstract statements..."):
            lit_res = orchestrator.search_literature(query_input, max_articles=max_articles)

        if lit_res.status != "SUCCESS":
            st.warning(f"Retrieval Status: {lit_res.status} — {lit_res.error_message}")
            return

        st.success(f"Retrieved {lit_res.articles_retrieved} articles, parsed and ranked {lit_res.total_sentences_analyzed} sentences using `{lit_res.engine_used}`.")

        st.divider()

        # Summary Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Articles Analyzed", str(lit_res.articles_retrieved), "PubMed records retrieved")
        with m2:
            render_metric_card("Sentences Evaluated", str(lit_res.total_sentences_analyzed), "Segmented abstract sentences")
        with m3:
            top_score = lit_res.ranked_sentences[0].relevance_score if lit_res.ranked_sentences else 0.0
            render_metric_card("Top Semantic Match", f"{top_score:.3f}", "Cosine relevance to query", "#2563eb")

        st.divider()

        st.markdown("### 2. Ranked Scientific Statements with Citation Traceability")

        for idx, sent in enumerate(lit_res.ranked_sentences, start=1):
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: white; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 700; color: #1e3a8a;">Statement #{idx}</span>
                            <span style="background: #eff6ff; color: #1d4ed8; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.85rem;">
                                Relevance Score: {sent.relevance_score:.3f}
                            </span>
                        </div>
                        <p style="font-size: 1.02rem; color: #1e293b; line-height: 1.5; font-style: italic;">
                            "{sent.sentence}"
                        </p>
                        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.6rem;">
                            <strong>Source Article:</strong> {sent.article_title}<br>
                            <strong>Authors:</strong> {sent.authors} ({sent.year}) | 
                            <strong>PMID:</strong> <a href="{sent.source_url}" target="_blank" style="color: #2563eb; text-decoration: underline;">{sent.pmid} ↗</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(lit_res.disclaimer)
