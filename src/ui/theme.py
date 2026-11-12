"""MoleculeIQ — Molecular Observatory Design System.

Tokens, Stylesheet, and Visual Architecture.
Theme: Molecular Universe × Molecular Laboratory (Strict Non-Negotiable Dark Theme).
"""

import streamlit as st

# Design Tokens (Single Source of Truth)
TOKENS = {
    "colors": {
        # Core Observatory Palette
        "observatory_black": "#080D16",
        "midnight_navy": "#0B1220",
        "deep_slate": "#142236",
        "elevated_slate": "#1B2B3D",
        "border_slate": "#1E314B",
        "border_subtle": "#24364D",
        # Accents
        "scientific_teal": "#4F9D96",
        "scientific_teal_dark": "#3A7570",
        "molecular_cyan": "#39C6D8",
        # Neutrals
        "off_white": "#F3F6FA",
        "cool_grey": "#AAB7C5",
        "muted_grey": "#77889A",
        # Semantic
        "success": "#57B88B",
        "warning": "#D9A441",
        "error": "#E07878",
        "information": "#72A9E8",
        # Semantic Mappings
        "bg_app": "#080D16",
        "bg_surface": "#0B1220",
        "bg_surface_card": "#142236",
        "bg_surface_elevated": "#1B2B3D",
        "bg_sidebar": "#0B1220",
        "text_primary": "#F3F6FA",
        "text_secondary": "#AAB7C5",
        "text_muted": "#77889A",
    },
    "typography": {
        "font_heading": "'Manrope', -apple-system, BlinkMacSystemFont, sans-serif",
        "font_body": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        "font_mono": "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
    },
    "layout": {
        "max_width": "1280px",
        "radius_base": "8px",
        "radius_card": "12px",
        "radius_pill": "9999px",
        "spacing_unit": "8px",
    },
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

/* Box Sizing & Root Layout */
*, *::before, *::after {
    box-sizing: border-box !important;
}

/* Global Reset & Dark App Viewport (Observatory Black) */
html, body, .stApp {
    background-color: #080D16 !important;
    color: #F3F6FA !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    overflow-x: hidden !important;
    max-width: 100vw !important;
}

.main .block-container {
    max-width: 1280px;
    width: 100%;
    padding-top: 1.25rem;
    padding-bottom: 3.5rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    margin: 0 auto;
}

/* Eradicate unintended black rectangular backgrounds behind headings, markdown blocks, captions, and cards */
h1, h2, h3, h4, h5, h6,
.stHeading,
div[data-testid="stHeadingWithActionElements"],
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div,
.stMarkdown,
.stMarkdown > div,
.stCaption,
[data-testid="stCaptionContainer"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Card Heading & Content Background Isolation */
.mol-card, .mol-card-elevated, .mol-card-dark {
    box-sizing: border-box;
}

.mol-card h1, .mol-card h2, .mol-card h3, .mol-card h4, .mol-card h5, .mol-card h6,
.mol-card-elevated h1, .mol-card-elevated h2, .mol-card-elevated h3, .mol-card-elevated h4,
.mol-card-dark h1, .mol-card-dark h2, .mol-card-dark h3, .mol-card-dark h4 {
    background: transparent !important;
    background-color: transparent !important;
}

/* Headings with Manrope */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #F3F6FA !important;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.25;
}

h1 {
    font-weight: 800;
}

p, span, label, li {
    color: #AAB7C5;
    font-size: 0.95rem;
    line-height: 1.6;
}

strong, b {
    color: #F3F6FA;
}

/* Monospace Elements */
code, pre, .mono-text, .smiles-code-box {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}

/* Sidebar Shell (Midnight Navy) with Unwrapped Navigation */
[data-testid="stSidebar"] {
    background-color: #0B1220 !important;
    border-right: 1px solid #1E314B !important;
    width: 310px !important;
    min-width: 290px !important;
}

[data-testid="stSidebar"] * {
    color: #AAB7C5;
}

.sidebar-category {
    font-size: 0.68rem;
    font-weight: 700;
    color: #77889A;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1.25rem;
    margin-bottom: 0.45rem;
    padding-left: 0.25rem;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 0.25rem;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem;
    font-weight: 500;
    color: #AAB7C5 !important;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.15s ease-in-out;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background-color: #142236;
    color: #F3F6FA !important;
}

[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
    background-color: rgba(79, 157, 150, 0.15) !important;
    color: #39C6D8 !important;
    font-weight: 600;
    border: 1px solid rgba(79, 157, 150, 0.35);
}

/* Observatory Header Component */
.observatory-header-bar {
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #1E314B;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.observatory-breadcrumbs {
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #77889A;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.observatory-breadcrumbs a, .observatory-breadcrumbs span.active {
    color: #4F9D96;
}

/* Scientific Boundary Banner (Dark Theme) */
.observatory-boundary-banner {
    background: #0B1220;
    border-left: 3px solid #4F9D96;
    border-top: 1px solid #1E314B;
    border-right: 1px solid #1E314B;
    border-bottom: 1px solid #1E314B;
    padding: 0.75rem 1.15rem;
    border-radius: 8px;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.6rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

/* Editorial & Research Card Structures */
.mol-card {
    background: #142236;
    border: 1px solid #1E314B;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.15rem;
    color: #F3F6FA;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    transition: border-color 0.2s ease;
}

.mol-card:hover {
    border-color: #24364D;
}

.mol-card-elevated {
    background: #1B2B3D;
    border: 1px solid #24364D;
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.15rem;
    color: #F3F6FA;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

.mol-card-dark {
    background: #0B1220;
    border: 1px solid #1E314B;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.15rem;
    color: #F3F6FA;
}

/* Research Triage Row (Dark Observatory Grid) */
.miq-triage-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 0.85rem;
    padding: 0.85rem 0;
    border-top: 1px solid #1E314B;
    border-bottom: 1px solid #1E314B;
    margin: 1.25rem 0;
}

.miq-triage-item {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #142236;
    border: 1px solid #1E314B;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}

.miq-triage-label {
    font-size: 0.70rem;
    font-weight: 700;
    color: #77889A;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
    font-family: 'IBM Plex Mono', monospace;
}

.miq-triage-value {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.25rem;
    color: #F3F6FA;
    font-family: 'Manrope', sans-serif;
}

.miq-triage-desc {
    font-size: 0.74rem;
    color: #AAB7C5;
    line-height: 1.35;
}

/* 2D Molecular Structure Vector Canvas (Dark Slate) */
.mol-structure-canvas {
    background: #0B1220;
    border-radius: 12px;
    border: 1px solid #1E314B;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.25rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: border-color 0.2s ease;
}

.mol-structure-canvas:hover {
    border-color: #4F9D96;
}

/* 3D Molecular Conformer Viewer Canvas */
.mol-3d-canvas-wrapper {
    background: #080D16;
    border-radius: 12px;
    border: 1px solid #1E314B;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
}

/* Badges & Status Indicators */
.mol-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.60rem;
    border-radius: 9999px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

.mol-badge-success {
    background-color: rgba(87, 184, 139, 0.15);
    color: #57B88B;
    border: 1px solid rgba(87, 184, 139, 0.4);
}

.mol-badge-warning {
    background-color: rgba(217, 164, 65, 0.15);
    color: #D9A441;
    border: 1px solid rgba(217, 164, 65, 0.4);
}

.mol-badge-danger {
    background-color: rgba(224, 120, 120, 0.15);
    color: #E07878;
    border: 1px solid rgba(224, 120, 120, 0.4);
}

.mol-badge-info, .mol-badge-teal {
    background-color: rgba(79, 157, 150, 0.15);
    color: #4F9D96;
    border: 1px solid rgba(79, 157, 150, 0.4);
}

.mol-badge-cyan {
    background-color: rgba(57, 198, 216, 0.15);
    color: #39C6D8;
    border: 1px solid rgba(57, 198, 216, 0.4);
}

.mol-badge-purple {
    background-color: rgba(142, 124, 255, 0.15);
    color: #8E7CFF;
    border: 1px solid rgba(142, 124, 255, 0.4);
}

.mol-badge-neutral {
    background-color: #142236;
    color: #AAB7C5;
    border: 1px solid #1E314B;
}

/* Custom Streamlit Tabs (Observatory Dark) */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background-color: #0B1220 !important;
    padding: 0.35rem 0.45rem;
    border-radius: 10px;
    border: 1px solid #1E314B;
    margin-bottom: 1.35rem;
}

.stTabs [data-baseweb="tab"] {
    height: 2.6rem;
    border-radius: 8px;
    color: #AAB7C5 !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0 1.25rem;
    border: none;
    background-color: transparent;
    transition: all 0.15s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F3F6FA !important;
    background-color: rgba(20, 34, 54, 0.7);
}

.stTabs [aria-selected="true"] {
    background-color: #142236 !important;
    color: #39C6D8 !important;
    border: 1px solid #24364D !important;
    border-bottom: 2px solid #4F9D96 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

/* Primary & Secondary Buttons */
.stButton > button {
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 1.15rem;
    min-height: 2.5rem;
    transition: all 0.15s ease;
    border: 1px solid #1E314B;
    background-color: #142236;
    color: #F3F6FA;
    font-size: 0.90rem;
}

.stButton > button:hover {
    border-color: #4F9D96;
    color: #39C6D8;
    background-color: #1B2B3D;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F9D96 0%, #3A7570 100%) !important;
    border: 1px solid #4F9D96 !important;
    color: #080D16 !important;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(79, 157, 150, 0.35);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #39C6D8 0%, #4F9D96 100%) !important;
    color: #080D16 !important;
    border-color: #39C6D8 !important;
    box-shadow: 0 6px 18px rgba(57, 198, 216, 0.45);
    transform: translateY(-1px);
}

/* Hairline Section Divider */
.miq-divider {
    border-top: 1px solid #1E314B;
    margin: 1.75rem 0;
}

/* Monospace SMILES Box */
.smiles-code-box {
    background: #0B1220;
    border: 1px solid #1E314B;
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.84rem;
    color: #39C6D8;
    word-break: break-all;
}

/* Streamlit Inputs, Dropdowns & Textareas */
[data-baseweb="input"] {
    background-color: #0B1220 !important;
    border-color: #1E314B !important;
    color: #F3F6FA !important;
    border-radius: 8px !important;
}

[data-baseweb="input"] input {
    color: #F3F6FA !important;
}

[data-baseweb="select"] {
    background-color: #0B1220 !important;
    color: #F3F6FA !important;
}

[data-baseweb="select"] * {
    background-color: #0B1220 !important;
    color: #F3F6FA !important;
}

textarea {
    background-color: #0B1220 !important;
    color: #F3F6FA !important;
    border-color: #1E314B !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #0B1220 !important;
    border: 1px solid #1E314B !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] summary {
    color: #F3F6FA !important;
    font-weight: 600;
}

/* Alerts & Notifications */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    background-color: #142236 !important;
    border: 1px solid #1E314B !important;
    color: #F3F6FA !important;
}

/* Metric Display */
div[data-testid="stMetricValue"] {
    color: #F3F6FA !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
}

/* Tables and DataFrames (Prevent viewport overflow) */
[data-testid="stDataFrame"], .stTable {
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #080D16;
}
::-webkit-scrollbar-thumb {
    background: #1E314B;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4F9D96;
}

/* Responsive Breakpoints */
@media (max-width: 1024px) {
    .main .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }
}

@media (max-width: 768px) {
    .main .block-container {
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        padding-top: 0.75rem !important;
    }
    .miq-triage-grid {
        grid-template-columns: 1fr !important;
    }
    [data-testid="stSidebar"] {
        width: 100% !important;
    }
}
</style>
"""


def apply_theme():
    """Inject MoleculeIQ Molecular Observatory design system stylesheet."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
