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
        "Semantic biomedical retrieval connecting molecular queries with peer-reviewed PubMed abstracts and cited statements.",
        badge_text="Live NCBI API",
        badge_variant="info",
    )

    st.markdown("### Scientific Query & PubMed Retrieval")

    # Check for pre-filled query from Single Molecule view
    prefilled_query = st.session_state.pop("lit_query_prefill", None)

    with st.container():
        st.markdown("""<div class="mol-card" style="padding: 1.25rem 1.5rem;">""", unsafe_allow_html=True)
        col_q, col_opts = st.columns([2.5, 1.2])

        with col_opts:
            selected_sample = st.selectbox("Preset Research Topics:", ["-- Select Sample --"] + SAMPLE_QUERIES)
            max_articles = st.slider("Max PubMed Articles:", min_value=2, max_value=10, value=5)

        with col_q:
            if prefilled_query:
                default_query = prefilled_query
            elif selected_sample != "-- Select Sample --":
                default_query = selected_sample
            else:
                default_query = "aspirin cardiotoxicity"

            query_input = st.text_input("Enter research topic, chemical target, or adverse event:", value=default_query)

        col_btn, _ = st.columns([1, 2.5])
        with col_btn:
            search_btn = st.button("Retrieve & Rank Literature", type="primary", use_container_width=True)

        st.markdown("""</div>""", unsafe_allow_html=True)

    if search_btn and query_input:
        with st.spinner("Querying NCBI PubMed and ranking abstract statements via semantic cosine similarity..."):
            lit_res = orchestrator.search_literature(query_input, max_articles=max_articles)
            st.session_state["lit_search_result"] = lit_res
            st.session_state["lit_search_query"] = query_input

    # Retrieve cached literature results
    lit_res = st.session_state.get("lit_search_result")
    active_query = st.session_state.get("lit_search_query")

    if lit_res is not None:
        if lit_res.status != "SUCCESS":
            if lit_res.status == "API_ERROR":
                st.error(f"❌ **Literature Retrieval Error:** {lit_res.error_message}")
            elif lit_res.status in ("NO_ARTICLES_FOUND", "NO_RELEVANT_STATEMENTS"):
                st.info(f"ℹ️ **No Matches:** {lit_res.error_message}")
            elif lit_res.status == "NO_ABSTRACTS_AVAILABLE":
                st.warning(f"⚠️ **Abstract Text Unavailable:** {lit_res.error_message}")
            else:
                st.warning(f"⚠️ **Retrieval Status ({lit_res.status}):** {lit_res.error_message}")
            return

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.success(f"✓ Retrieved **{lit_res.articles_retrieved}** PubMed articles, evaluated **{lit_res.total_sentences_analyzed}** abstract sentences, and ranked **{len(lit_res.ranked_sentences)}** unique statements using `{lit_res.engine_used}`.")

        # Summary Metrics
        st.markdown("### Retrieval & Semantic Summary")
        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Articles Analyzed", str(lit_res.articles_retrieved), "PubMed records retrieved", badge="PubMed", badge_variant="info")
        with m2:
            render_metric_card("Sentences Evaluated", str(lit_res.total_sentences_analyzed), "Segmented abstract sentences", badge="TF-IDF", badge_variant="purple")
        with m3:
            top_sent = lit_res.ranked_sentences[0] if lit_res.ranked_sentences else None
            top_score = top_sent.relevance_score if top_sent else 0.0
            top_bonus = getattr(top_sent, "conjunction_bonus", 0.0) or 0.0
            has_bonus = bool(top_sent and top_bonus > 0.0)
            metric_subtitle = "TF-IDF cosine + conjunction bonus" if has_bonus else "TF-IDF vector cosine"
            metric_badge = "Composite Metric" if has_bonus else "Cosine Metric"
            render_metric_card(
                "Peak Ranking Score",
                f"{top_score:.3f}",
                metric_subtitle,
                "#27C7B8",
                badge=metric_badge,
                badge_variant="info",
            )

        if top_score < 0.10:
            st.info("ℹ️ **Information Retrieval Notice:** Retrieved statements exhibit low textual overlap (TF-IDF cosine < 0.10) with the query. Review the linked PubMed abstracts for broader scientific context.")

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Ranked Scientific Statements with Citation Traceability")

        col_stream, col_inspector = st.columns([1.75, 1.15], gap="large")

        with col_stream:
            for idx, sent in enumerate(lit_res.ranked_sentences, start=1):
                rel_color = "#57B88B" if sent.relevance_score >= 0.45 else ("#4F9D96" if sent.relevance_score >= 0.25 else "#8A99AD")
                score_val = sent.relevance_score
                bon_val = getattr(sent, "conjunction_bonus", 0.0) or 0.0
                cos_val = getattr(sent, "cosine_score", None)
                if cos_val is None:
                    cos_val = round(score_val - bon_val, 4)

                if bon_val > 0.0:
                    badge_text = f"Score: {score_val:.3f} (Cosine: {cos_val:.3f} + Conjunction: +{bon_val:.3f})"
                else:
                    badge_text = f"TF-IDF Cosine: {score_val:.3f}"

                pol_label = getattr(sent, "claim_polarity", None) or getattr(sent, "polarity", None) or "UNCERTAIN"
                pol_exp = getattr(sent, "polarity_explanation", "") or ""

                if pol_label == "AFFIRMATIVE":
                    pol_badge_text = "Possible affirmative claim"
                    pol_badge_class = "mol-badge-info"
                elif pol_label == "NEGATED":
                    pol_badge_text = "Negated wording"
                    pol_badge_class = "mol-badge-purple"
                else:
                    pol_badge_text = "Polarity uncertain"
                    pol_badge_class = "mol-badge-neutral"

                polarity_badge_html = f'<span class="mol-badge {pol_badge_class}" style="font-size: 0.75rem; margin-right: 0.35rem;" title="{pol_exp}">{pol_badge_text}</span>'

                st.markdown(
                    f"""
                    <div class="mol-card-elevated" style="border-left: 4px solid {rel_color}; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.4rem;">
                            <span style="font-weight: 700; color: #8A99AD; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Statement #{idx}</span>
                            <div style="display: flex; align-items: center; gap: 0.35rem; flex-wrap: wrap;">
                                {polarity_badge_html}
                                <span class="mol-badge mol-badge-info" style="font-size: 0.78rem;">
                                    {badge_text}
                                </span>
                            </div>
                        </div>
                        <p style="font-size: 0.96rem; color: #F3F6FA; line-height: 1.6; font-style: italic; margin: 0.4rem 0 0.8rem 0;">
                            "{sent.sentence}"
                        </p>
                        <div style="background: #142236; padding: 0.65rem 0.85rem; border-radius: 6px; border: 1px solid #1E314B; font-size: 0.82rem; color: #AAB7C5;">
                            <div style="color: #F3F6FA; font-weight: 600; margin-bottom: 0.25rem;">📄 {sent.article_title}</div>
                            <div>
                                <span><strong>Authors:</strong> {sent.authors} ({sent.year})</span> &nbsp;|&nbsp;
                                <span><strong>PMID:</strong> <a href="{sent.source_url}" target="_blank" style="color: #39C6D8; text-decoration: underline; font-weight: 600;">{sent.pmid} ↗</a></span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_inspector:
            st.markdown(
                """
                <div style="position: sticky; top: 1rem;">
                    <div class="mol-card" style="padding: 1.15rem; margin-bottom: 1rem;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #39C6D8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; font-family: 'IBM Plex Mono', monospace;">
                            PUBLICATION INSPECTOR
                        </div>
                        <h4 style="margin: 0 0 0.65rem 0; font-size: 1.1rem; color: #F3F6FA; font-family: 'Manrope', sans-serif;">
                            Evidence & Source Verification
                        </h4>
                        <p style="font-size: 0.82rem; color: #AAB7C5; line-height: 1.5; margin-bottom: 0.85rem;">
                            Statements are retrieved directly from the peer-reviewed National Center for Biotechnology Information (NCBI) PubMed index.
                        </p>
                        <div style="background: #142236; border: 1px solid #1E314B; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.75rem; font-size: 0.8rem; color: #CBD5E1;">
                            <div style="color: #8A99AD; font-size: 0.72rem; text-transform: uppercase; font-weight: 700; margin-bottom: 0.2rem;">AUTOMATED CLASSIFICATION NOTICE</div>
                            Linguistic polarity labels ("Possible affirmative claim", "Negated wording", "Polarity uncertain") are heuristic text-mining annotations. They do not constitute pharmacological proof and require human source verification.
                        </div>
                        <div style="font-size: 0.78rem; color: #8A99AD; line-height: 1.45;">
                            Click any PMID link on the left to inspect the primary publication in its entirety on NCBI PubMed.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if hasattr(lit_res, "disclaimer") and lit_res.disclaimer:
            st.caption(f"ℹ️ {lit_res.disclaimer}")
        st.caption("ℹ️ **Metric Note:** TF-IDF cosine similarity measures textual term overlap between query keywords and abstract sentences. It serves as an information retrieval ranking aid and does not constitute biological validation or evidence confidence.")
        st.caption("ℹ️ **Claim Polarity Notice:** Linguistic polarity badges reflect rule-based heuristic text annotations of abstract wording ('Possible affirmative claim', 'Negated wording', 'Polarity uncertain'). They describe phrasing only and do not establish biological validity, clinical truth, causal relationships, or safety proof.")
