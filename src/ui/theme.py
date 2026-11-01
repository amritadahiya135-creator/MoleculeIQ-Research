"""MoleculeIQ Design System — Tokens, Stylesheet, and Brand Identity.

Design Philosophy: Precision Scientific Research Instrument.
Brand Palette:
- Midnight Navy: #101B2D (Sidebar, dark surfaces, primary heading contrast)
- Dark Surface:  #1C2B40 (Sidebar cards, borders, 3D canvas backdrop)
- Scientific Teal: #27C7B8 (Primary actions, active states, key metrics)
- Ice White:     #F4F7FB (Main app background, clean workspace)
- Muted Violet:  #8E7CFF (Secondary model analysis, explainability, stress assays)
- Cool Gray:     #D9E2EF (Hairline borders, dividers, subtle neutral fills)
- Amber:         #E5A34D (Warnings, uncertainty, boundary zones)
"""

import streamlit as st

# Design Tokens (Single Source of Truth)
TOKENS = {
    "colors": {
        "midnight_navy": "#101B2D",
        "dark_surface": "#1C2B40",
        "scientific_teal": "#27C7B8",
        "scientific_teal_dark": "#1BA699",
        "ice_white": "#F4F7FB",
        "muted_violet": "#8E7CFF",
        "cool_gray": "#D9E2EF",
        "amber": "#E5A34D",
        # Semantic Mappings
        "bg_app": "#F4F7FB",
        "bg_surface": "#FFFFFF",
        "bg_surface_subtle": "#EDF2F7",
        "bg_sidebar": "#101B2D",
        "bg_sidebar_card": "#1C2B40",
        "border_subtle": "#D9E2EF",
        "border_medium": "#B8C9DF",
        "border_sidebar": "#2A3B52",
        "text_primary": "#101B2D",
        "text_secondary": "#334155",
        "text_muted": "#64748B",
        "text_sidebar_primary": "#F4F7FB",
        "text_sidebar_secondary": "#CBD5E1",
        "text_sidebar_muted": "#94A3B8",
        "status_success": "#10B981",
        "status_warning": "#E5A34D",
        "status_danger": "#EF4444",
        "status_info": "#27C7B8",
        "status_purple": "#8E7CFF",
    },
    "typography": {
        "font_family": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        "font_mono": 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    },
    "layout": {
        "max_width": "1240px",
        "radius_base": "6px",
        "radius_card": "8px",
    }
}

CUSTOM_CSS = """
<style>
/* Global App Background & Container Constraints */
.stApp {
    background-color: #F4F7FB;
    color: #101B2D;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.main .block-container {
    max-width: 1240px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
    margin: 0 auto;
}

/* Sidebar Branding & Navigation Shell (Midnight Navy) */
[data-testid="stSidebar"] {
    background-color: #101B2D !important;
    border-right: 1px solid #1C2B40;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1;
}

.sidebar-category {
    font-size: 0.7rem;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1.25rem;
    margin-bottom: 0.35rem;
    padding-left: 0.25rem;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
    font-weight: 500;
    color: #CBD5E1 !important;
    padding: 0.4rem 0.65rem;
    border-radius: 6px;
    transition: all 0.15s ease-in-out;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(28, 43, 64, 0.8);
    color: #F4F7FB !important;
}

[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
    color: #27C7B8 !important;
    font-weight: 700;
}

/* Typography Hierarchy in Main Workspace */
h1, h2, h3, h4, h5, h6 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #101B2D !important;
    letter-spacing: -0.02em;
    font-weight: 700;
}

p, span, label {
    color: #334155;
}

/* Scientific Triage Row (Anti-Card Wall Layout) */
.miq-triage-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    padding: 1rem 0;
    border-top: 1px solid #D9E2EF;
    border-bottom: 1px solid #D9E2EF;
    margin: 1.25rem 0;
}

.miq-triage-item {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);
}

.miq-triage-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.miq-triage-value {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.2rem;
    color: #101B2D;
}

.miq-triage-desc {
    font-size: 0.76rem;
    color: #64748B;
    line-height: 1.35;
}

/* Editorial Framing Panels */
.mol-card {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.15rem;
    box-shadow: 0 1px 3px rgba(16, 27, 45, 0.04);
}

.mol-card-elevated {
    background: #FFFFFF;
    border: 1px solid #B8C9DF;
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.15rem;
    box-shadow: 0 4px 12px rgba(16, 27, 45, 0.06);
}

.mol-card-dark {
    background: #101B2D;
    border: 1px solid #1C2B40;
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.15rem;
    color: #F4F7FB;
}

/* 2D Molecular Canvas */
.mol-structure-canvas {
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #D9E2EF;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(16, 27, 45, 0.04);
    transition: border-color 0.2s ease;
}

.mol-structure-canvas:hover {
    border-color: #27C7B8;
}

/* 3D Molecular Conformer Viewer Canvas */
.mol-3d-canvas-wrapper {
    background: #101B2D;
    border-radius: 8px;
    border: 1px solid #1C2B40;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(16, 27, 45, 0.12);
}

/* Badges */
.mol-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.mol-badge-success {
    background-color: rgba(16, 185, 129, 0.12);
    color: #065F46;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.mol-badge-warning {
    background-color: rgba(229, 163, 77, 0.15);
    color: #9A6010;
    border: 1px solid rgba(229, 163, 77, 0.4);
}

.mol-badge-danger {
    background-color: rgba(239, 68, 68, 0.12);
    color: #991B1B;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.mol-badge-info, .mol-badge-teal {
    background-color: rgba(39, 199, 184, 0.15);
    color: #0F8075;
    border: 1px solid rgba(39, 199, 184, 0.4);
}

.mol-badge-purple, .mol-badge-violet {
    background-color: rgba(142, 124, 255, 0.15);
    color: #5B48DF;
    border: 1px solid rgba(142, 124, 255, 0.35);
}

.mol-badge-neutral {
    background-color: #EDF2F7;
    color: #475569;
    border: 1px solid #D9E2EF;
}

/* Custom Tab Bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background-color: #EDF2F7;
    padding: 0.3rem 0.4rem;
    border-radius: 8px;
    border: 1px solid #D9E2EF;
    margin-bottom: 1.25rem;
}

.stTabs [data-baseweb="tab"] {
    height: 2.5rem;
    border-radius: 6px;
    color: #475569;
    font-weight: 600;
    font-size: 0.90rem;
    padding: 0 1.15rem;
    border: none;
    background-color: transparent;
    transition: all 0.15s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #101B2D;
    background-color: rgba(255, 255, 255, 0.6);
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #101B2D !important;
    border: 1px solid #D9E2EF !important;
    border-bottom: 2px solid #27C7B8 !important;
    box-shadow: 0 1px 4px rgba(16, 27, 45, 0.05);
}

/* Button Refinement */
.stButton > button {
    font-weight: 600;
    border-radius: 6px;
    padding: 0.45rem 0.9rem;
    transition: all 0.15s ease;
    border: 1px solid #D9E2EF;
    background-color: #FFFFFF;
    color: #101B2D;
    white-space: nowrap;
}

.stButton > button:hover {
    border-color: #27C7B8;
    color: #0F8075;
}

.stButton > button[kind="primary"] {
    background: #27C7B8;
    border: 1px solid #1BA699;
    color: #101B2D !important;
    font-weight: 700;
    box-shadow: 0 2px 4px rgba(39, 199, 184, 0.25);
}

.stButton > button[kind="primary"]:hover {
    background: #1BA699;
    color: #FFFFFF !important;
    border-color: #16877C;
    transform: translateY(-1px);
}

/* Hairline Section Divider */
.miq-divider {
    border-top: 1px solid #D9E2EF;
    margin: 1.35rem 0;
}

/* Range Track Indicator */
.range-track-container {
    margin: 0.4rem 0 0.8rem 0;
}

.range-track-bar {
    width: 100%;
    height: 6px;
    background: #EDF2F7;
    border: 1px solid #D9E2EF;
    border-radius: 3px;
    position: relative;
}

.range-track-fill {
    height: 100%;
    border-radius: 3px;
}

/* Monospace SMILES Box */
.smiles-code-box {
    background: #EDF2F7;
    border: 1px solid #D9E2EF;
    border-radius: 6px;
    padding: 0.45rem 0.75rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.82rem;
    color: #101B2D;
    word-break: break-all;
}

/* Streamlit Inputs & DataFrames styling */
[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border-color: #D9E2EF !important;
}

[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: #F4F7FB;
}
::-webkit-scrollbar-thumb {
    background: #D9E2EF;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #27C7B8;
}
</style>
"""


def apply_theme():
    """Inject MoleculeIQ design system stylesheet into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
