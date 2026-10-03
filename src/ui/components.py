"""Reusable UI presentation components, disclaimers, and metric cards (Module M10)."""

import streamlit as st


def render_header(title: str, subtitle: str):
    """Render a clean, restrained scientific header."""
    st.markdown(
        f"""
        <div style="padding: 1rem 0; border-bottom: 1px solid #e0e0e0; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; color: #1e3a8a; font-size: 2.2rem; font-weight: 700;">{title}</h1>
            <p style="margin: 0.3rem 0 0 0; color: #475569; font-size: 1.05rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scientific_boundary_banner():
    """Render mandatory non-clinical, non-regulatory research disclaimer banner."""
    st.markdown(
        """
        <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 0.8rem 1rem; border-radius: 4px; margin-bottom: 1.5rem;">
            <strong style="color: #1e3a8a;">RESEARCH & EDUCATIONAL USE ONLY:</strong>
            <span style="color: #334155; font-size: 0.92rem;">
            MoleculeIQ is an in silico screening and computational triage prototype. It does not provide regulatory toxicology, clinical diagnostics, or experimental validation of biological safety or efficacy.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtitle: str = "", delta_color: str = "#0f172a"):
    """Render a clean scientific statistic card."""
    st.markdown(
        f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {delta_color}; margin: 0.3rem 0;">{value}</div>
            <div style="font-size: 0.82rem; color: #94a3b8;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
