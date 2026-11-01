"""Reusable UI presentation components, disclaimers, and metric displays (Module M10)."""

from typing import Optional, List, Dict, Any
import streamlit as st
from src.services.contracts import MoleculeAnalysisResult


def render_header(
    title: str,
    subtitle: str,
    badge_text: Optional[str] = None,
    badge_variant: str = "teal",
):
    """Render a clean, modern scientific header with optional badge."""
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="mol-badge mol-badge-{badge_variant}" style="margin-left: 0.75rem; vertical-align: middle;">{badge_text}</span>'

    header_html = (
        '<div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #D9E2EF; margin-bottom: 1.25rem;">'
        '<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">'
        f'<h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #101B2D; letter-spacing: -0.02em;">{title} {badge_html}</h1>'
        '</div>'
        f'<p style="margin: 0.35rem 0 0 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">{subtitle}</p>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_scientific_boundary_banner():
    """Render mandatory non-clinical, non-regulatory research disclaimer banner."""
    banner_html = (
        '<div style="background: #FFFFFF; border-left: 3px solid #27C7B8; border-top: 1px solid #D9E2EF; border-right: 1px solid #D9E2EF; border-bottom: 1px solid #D9E2EF; padding: 0.65rem 0.95rem; border-radius: 6px; margin-bottom: 1.15rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">'
        '<div style="font-size: 0.84rem; color: #334155;">'
        '<strong style="color: #0F8075; letter-spacing: 0.02em; margin-right: 0.35rem;">RESEARCH & EDUCATIONAL USE ONLY:</strong>'
        '<span>MoleculeIQ is an in silico screening instrument. Predictions do not constitute clinical diagnostics, regulatory clearance, or pharmacological proof of safety or efficacy.</span>'
        '</div>'
        '<div style="font-size: 0.7rem; color: #101B2D; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; background: rgba(39, 199, 184, 0.15); border: 1px solid rgba(39, 199, 184, 0.35); padding: 0.2rem 0.55rem; border-radius: 4px;">'
        'In Silico Triage Mode'
        '</div>'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    delta_color: str = "#27C7B8",
    badge: Optional[str] = None,
    badge_variant: str = "teal",
):
    """Render a clean, elevated scientific statistic card in brand styling."""
    badge_html = f'<span class="mol-badge mol-badge-{badge_variant}">{badge}</span>' if badge else ""

    card_html = (
        '<div class="mol-card" style="padding: 1rem 1.15rem; margin-bottom: 0.75rem; height: 100%;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">'
        f'<span style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">{title}</span>'
        f'{badge_html}'
        '</div>'
        f'<div style="font-size: 1.65rem; font-weight: 800; color: {delta_color}; letter-spacing: -0.02em; line-height: 1.15; margin: 0.2rem 0;">{value}</div>'
        f'<div style="font-size: 0.78rem; color: #64748B; line-height: 1.3;">{subtitle}</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_triage_editorial_grid(res: MoleculeAnalysisResult):
    """
    Render the editorial Research Triage summary row.
    Presents 5 core analytical dimensions in an open, aligned typographic grid.
    """
    # 1. Safety Dimension
    p_pct = res.clintox.probability * 100.0 if res.clintox else 0.0
    if p_pct >= 60.0:
        safety_status = "Higher Predicted ClinTox Probability"
        safety_color = "#EF4444"
        safety_badge = "danger"
    elif p_pct >= 35.0:
        safety_status = "Moderate Predicted ClinTox Probability"
        safety_color = "#E5A34D"
        safety_badge = "warning"
    else:
        safety_status = "Lower Predicted ClinTox Probability"
        safety_color = "#0F8075"
        safety_badge = "teal"

    # 2. Priority Score
    score_val = f"{res.prioritization.overall_score:.1f}" if res.prioritization else "N/A"

    # 3. Drug-Likeness
    lip_pass = res.descriptors.lipinski_pass if res.descriptors else False
    lip_label = "Lipinski Compliant" if lip_pass else "Non-Compliant"
    lip_color = "#065F46" if lip_pass else "#991B1B"
    lip_badge = "success" if lip_pass else "danger"
    lip_viol = f"{res.descriptors.lipinski_violations} Violations" if not lip_pass else "0 Violations"

    # 4. Applicability Domain
    ad_stat = getattr(res.clintox, "applicability_domain_status", "UNAVAILABLE") if res.clintox else "UNAVAILABLE"
    ad_score = getattr(res.clintox, "applicability_score", None) if res.clintox else None
    ad_score_str = f"{ad_score:.2f}" if ad_score is not None else "N/A"
    if ad_stat == "WITHIN_DOMAIN":
        ad_label = "Within Domain"
        ad_color = "#0F8075"
        ad_badge = "info"
    elif ad_stat == "NEAR_BOUNDARY":
        ad_label = "Boundary"
        ad_color = "#9A6010"
        ad_badge = "warning"
    elif ad_stat == "OUTSIDE_DOMAIN":
        ad_label = "Outside Domain"
        ad_color = "#991B1B"
        ad_badge = "danger"
    else:
        ad_label = "Unavailable"
        ad_color = "#64748B"
        ad_badge = "neutral"

    # 5. Reference Similarity
    top_sim = res.similarity.best_similarity if res.similarity else 0.0
    top_name = res.similarity.matches[0].name if res.similarity and res.similarity.matches else "None"
    sim_label = "High Structural Resemblance" if top_sim >= 0.50 else ("Moderate Structural Resemblance" if top_sim >= 0.30 else "Distant Chemotype")
    sim_color = "#0F8075" if top_sim >= 0.50 else "#5B48DF"

    st.markdown(
        f"""
        <div class="miq-triage-grid">
            <div class="miq-triage-item">
                <div class="miq-triage-label">Exploratory Triage Index</div>
                <div class="miq-triage-value" style="color: #27C7B8;">{score_val} <span style="font-size: 0.95rem; color: #64748B; font-weight: 500;">/ 100</span></div>
                <div class="miq-triage-desc">Heuristic multi-parameter rank</div>
            </div>
            <div class="miq-triage-item">
                <div class="miq-triage-label">Estimated P(Tox)</div>
                <div class="miq-triage-value" style="color: {safety_color};">{p_pct:.1f}%</div>
                <div class="miq-triage-desc"><span class="mol-badge mol-badge-{safety_badge}">{safety_status}</span></div>
            </div>
            <div class="miq-triage-item">
                <div class="miq-triage-label">Drug-Likeness</div>
                <div class="miq-triage-value" style="color: {lip_color};">{lip_label}</div>
                <div class="miq-triage-desc"><span class="mol-badge mol-badge-{lip_badge}">{lip_viol}</span></div>
            </div>
            <div class="miq-triage-item">
                <div class="miq-triage-label">Applicability Domain</div>
                <div class="miq-triage-value" style="color: {ad_color};">{ad_score_str}</div>
                <div class="miq-triage-desc"><span class="mol-badge mol-badge-{ad_badge}">{ad_label}</span></div>
            </div>
            <div class="miq-triage-item">
                <div class="miq-triage-label">Reference Similarity</div>
                <div class="miq-triage-value" style="color: {sim_color};">{sim_label}</div>
                <div class="miq-triage-desc">{top_name} (Tanimoto: {top_sim:.3f})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_gauge(probability: float, is_toxic: bool, decision_threshold: float = 0.5):
    """Render a clean, continuous multi-tier visual risk gauge."""
    pct = max(0.0, min(100.0, probability * 100.0))
    if pct < 35.0:
        tier_label = "LOWER PREDICTED CLINTOX PROBABILITY"
        badge_variant = "info"
    elif pct < 60.0:
        tier_label = "MODERATE PREDICTED CLINTOX PROBABILITY"
        badge_variant = "warning"
    else:
        tier_label = "HIGHER PREDICTED CLINTOX PROBABILITY"
        badge_variant = "danger"

    marker_left = max(2.0, min(98.0, pct))

    st.markdown(
        f"""
        <div style="margin: 0.65rem 0 0.85rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">
                <span style="font-size: 0.86rem; color: #101B2D; font-weight: 600;">Calibrated ClinTox Probability:</span>
                <span class="mol-badge mol-badge-{badge_variant}">{tier_label} ({pct:.1f}%)</span>
            </div>
            <div style="position: relative; width: 100%; height: 8px; background: #EDF2F7; border: 1px solid #D9E2EF; border-radius: 4px; overflow: visible;">
                <div style="width: 100%; height: 100%; border-radius: 4px; background: linear-gradient(to right, #10B981 0%, #10B981 35%, #E5A34D 35%, #E5A34D 60%, #EF4444 60%, #EF4444 100%); opacity: 0.85;"></div>
                <div style="position: absolute; top: -3px; left: {marker_left}%; width: 5px; height: 14px; background: #101B2D; border: 1px solid #FFFFFF; border-radius: 2px; box-shadow: 0 1px 4px rgba(16,27,45,0.4); transform: translateX(-50%);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748B; margin-top: 0.35rem;">
                <span>0% (Minimal ClinTox Probability)</span>
                <span>Threshold: {decision_threshold*100:.0f}%</span>
                <span>100% (High ClinTox Probability)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_property_bar(label: str, value_str: str, fraction: float, threshold_desc: str, is_pass: bool = True):
    """Render a clean range indicator for a molecular descriptor."""
    fill_pct = max(3.0, min(100.0, fraction * 100.0))
    bar_color = "#27C7B8" if is_pass else "#E5A34D"
    status_icon = "✓" if is_pass else "⚠"

    st.markdown(
        f"""
        <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; font-size: 0.84rem;">
                <span style="color: #334155; font-weight: 500;">{label}</span>
                <span style="font-weight: 700; color: #101B2D;">{value_str} <span style="color: {bar_color}; font-size: 0.74rem;">{status_icon}</span></span>
            </div>
            <div style="width: 100%; height: 6px; background: #EDF2F7; border: 1px solid #D9E2EF; border-radius: 3px; margin: 0.25rem 0;">
                <div style="width: {fill_pct}%; height: 100%; background: {bar_color}; border-radius: 3px;"></div>
            </div>
            <div style="font-size: 0.72rem; color: #64748B;">Target: {threshold_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_interpretation(res: MoleculeAnalysisResult):
    """
    Synthesize an executive scientific interpretation based strictly on actual model outputs.
    Ensures non-diagnostic, scientifically defensible language.
    """
    if not res.is_valid:
        return

    p_tox = res.clintox.probability if res.clintox else 0.0
    if p_tox >= 0.60:
        tox_tone = "Higher Predicted ClinTox Probability"
        tox_desc = f"The calibrated ClinTox model estimates a {p_tox*100:.1f}% probability of trial failure due to toxicity under a 50% decision threshold."
        tox_color = "#EF4444"
    elif p_tox >= 0.35:
        tox_tone = "Moderate Predicted ClinTox Probability"
        tox_desc = f"The calibrated ClinTox model estimates a {p_tox*100:.1f}% probability of trial failure due to toxicity, near the 50% decision boundary."
        tox_color = "#E5A34D"
    else:
        tox_tone = "Lower Predicted ClinTox Probability"
        tox_desc = f"The calibrated ClinTox model estimates a {p_tox*100:.1f}% probability of trial failure due to toxicity."
        tox_color = "#27C7B8"

    ad_status = getattr(res.clintox, "applicability_domain_status", "UNAVAILABLE")
    ad_score = getattr(res.clintox, "applicability_score", None)
    ad_score_str = f"{ad_score:.2f}" if ad_score is not None else "N/A"
    if ad_status == "WITHIN_DOMAIN":
        domain_desc = f"The structure meets the configured domain criterion (maximum Tanimoto: {ad_score_str} ≥ 0.40)."
        domain_badge = '<span class="mol-badge mol-badge-info">Within Domain</span>'
    elif ad_status == "NEAR_BOUNDARY":
        domain_desc = f"The structure lies near the domain boundary (Tanimoto: {ad_score_str}). Interpret predictions with moderate confidence."
        domain_badge = '<span class="mol-badge mol-badge-warning">Near Boundary</span>'
    elif ad_status == "OUTSIDE_DOMAIN":
        domain_desc = f"Extrapolation warning: Molecule chemotype is outside the training domain (Tanimoto: {ad_score_str} < 0.25). Prediction may be unreliable."
        domain_badge = '<span class="mol-badge mol-badge-danger">Outside Domain</span>'
    else:
        domain_desc = "Applicability domain status could not be assessed for this structure."
        domain_badge = '<span class="mol-badge mol-badge-neutral">Domain Unavailable</span>'

    tox21_active = res.tox21.active_endpoints_count if res.tox21 else 0
    tox21_total = res.tox21.total_endpoints if res.tox21 else 12
    tox21_desc = f"{tox21_active} of {tox21_total} cellular stress and nuclear receptor pathways active in in-vitro assays."

    lipinski_pass = res.descriptors.lipinski_pass if res.descriptors else False
    viol_count = res.descriptors.lipinski_violations if res.descriptors else 0
    if lipinski_pass:
        drug_desc = "Lipinski Rule of 5 compliant (0 violations); standard oral drug-likeness parameter range."
    else:
        drug_desc = f"Exhibits {viol_count} Lipinski Rule of 5 violations, which may limit oral absorption."

    top_name = "None"
    top_sim = 0.0
    if res.similarity and res.similarity.matches:
        top_name = res.similarity.matches[0].name
        top_sim = res.similarity.matches[0].similarity
    sim_desc = f"Nearest approved reference compound is <strong>{top_name}</strong> (Tanimoto: {top_sim:.3f}). Proximity in chemical space indicates structural resemblance only; does not establish efficacy, safety, or mechanism."

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border-left: 3px solid {tox_color}; border-top: 1px solid #D9E2EF; border-right: 1px solid #D9E2EF; border-bottom: 1px solid #D9E2EF; border-radius: 8px; padding: 1.1rem 1.35rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">
                <span style="font-size: 0.74rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em;">
                    Executive Scientific Synthesis
                </span>
                {domain_badge}
            </div>
            <h3 style="margin: 0 0 0.4rem 0; font-size: 1.15rem; font-weight: 700; color: #101B2D;">
                {tox_tone}
            </h3>
            <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #334155; line-height: 1.55;">
                {tox_desc} {domain_desc}
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.5rem; margin-top: 0.65rem; padding-top: 0.65rem; border-top: 1px solid #D9E2EF; font-size: 0.82rem; color: #64748B;">
                <div><strong style="color: #101B2D;">Tox21 Assays:</strong> {tox21_desc}</div>
                <div><strong style="color: #101B2D;">Drug-Likeness:</strong> {drug_desc}</div>
                <div><strong style="color: #101B2D;">Reference Similarity:</strong> {sim_desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
