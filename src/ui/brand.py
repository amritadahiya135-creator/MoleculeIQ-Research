"""MoleculeIQ Brand Identity and Visual Assets (Single Source of Truth)."""

import os
import re
import base64

BRAND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "brand")


def _clean_svg(svg_content: str) -> str:
    """Strip XML comments, XML declarations, and leading whitespace from all lines to prevent Markdown code block triggers."""
    if not svg_content:
        return ""
    cleaned = re.sub(r"<\?xml.*?\?>", "", svg_content)
    cleaned = re.sub(r"<!--[\s\S]*?-->", "", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return " ".join(lines)


def _svg_to_data_uri(svg_content: str) -> str:
    """Convert SVG content to an absolute Base64 data URI image."""
    if not svg_content:
        return ""
    b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def _load_asset(filename: str) -> str:
    path = os.path.join(BRAND_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return _clean_svg(f.read())
    return ""


LOGO_SVG = _load_asset("moleculeiq-logo.svg")
MARK_SVG = _load_asset("moleculeiq-mark.svg")
FAVICON_SVG = _load_asset("moleculeiq-favicon.svg")


def render_brand_header(width: int = 210) -> str:
    """Render the MoleculeIQ horizontal brand lockup."""
    uri = _svg_to_data_uri(LOGO_SVG)
    return f'<div style="width: {width}px; max-width: 100%; margin-bottom: 0.5rem; display: inline-block;"><img src="{uri}" width="{width}" style="max-width: 100%; height: auto; display: block;" alt="MoleculeIQ"></div>'


def render_brand_mark(size: int = 36) -> str:
    """Render the standalone MoleculeIQ precision mark."""
    uri = _svg_to_data_uri(MARK_SVG)
    return f'<div style="width: {size}px; height: {size}px; display: inline-flex; align-items: center; justify-content: center;"><img src="{uri}" width="{size}" height="{size}" style="display: block;" alt="MoleculeIQ Mark"></div>'


def render_computational_network_svg(width: int = 460, height: int = 280) -> str:
    """
    Generate an original computational molecular graph visual for the landing page hero.
    Visualizes molecular coordinates, topological bonds, and neural weight pathways in brand palette.
    """
    raw_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 280" width="{width}" height="{height}" fill="none" style="max-width: 100%; height: auto;">
<defs>
<linearGradient id="net-bond-p" x1="40" y1="40" x2="420" y2="240" gradientUnits="userSpaceOnUse">
<stop offset="0%" stop-color="#27C7B8" stop-opacity="0.85"/>
<stop offset="50%" stop-color="#8E7CFF" stop-opacity="0.6"/>
<stop offset="100%" stop-color="#27C7B8" stop-opacity="0.75"/>
</linearGradient>
<radialGradient id="net-glow" cx="50%" cy="50%" r="50%">
<stop offset="0%" stop-color="#27C7B8" stop-opacity="0.12"/>
<stop offset="100%" stop-color="#101B2D" stop-opacity="0"/>
</radialGradient>
</defs>
<circle cx="230" cy="140" r="130" fill="url(#net-glow)"/>
<g stroke="url(#net-bond-p)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<polygon points="170,90 220,60 270,90 270,150 220,180 170,150" stroke="#27C7B8" stroke-width="2.2" fill="rgba(39, 199, 184, 0.05)"/>
<polygon points="270,90 320,70 350,110 320,150 270,150" stroke="#8E7CFF" stroke-width="1.8" fill="rgba(142, 124, 255, 0.04)"/>
<line x1="170" y1="90" x2="110" y2="70" stroke-dasharray="3,3"/>
<line x1="110" y1="70" x2="70" y2="110"/>
<line x1="110" y1="70" x2="90" y2="30"/>
<line x1="170" y1="150" x2="130" y2="200"/>
<line x1="130" y1="200" x2="80" y2="210"/>
<line x1="220" y1="180" x2="220" y2="230"/>
<line x1="220" y1="230" x2="270" y2="250"/>
<line x1="320" y1="150" x2="370" y2="190"/>
<line x1="370" y1="190" x2="420" y2="180"/>
<line x1="370" y1="190" x2="380" y2="240"/>
<line x1="350" y1="110" x2="410" y2="90"/>
<line x1="220" y1="60" x2="220" y2="180" stroke="#8E7CFF" stroke-width="1.2" stroke-opacity="0.6"/>
<line x1="170" y1="150" x2="270" y2="90" stroke="#8E7CFF" stroke-width="1.2" stroke-opacity="0.6"/>
</g>
<circle cx="170" cy="90" r="5" fill="#F4F7FB" stroke="#27C7B8" stroke-width="2"/>
<circle cx="220" cy="60" r="5.5" fill="#27C7B8" stroke="#FFFFFF" stroke-width="1.5"/>
<circle cx="270" cy="90" r="5" fill="#F4F7FB" stroke="#8E7CFF" stroke-width="2"/>
<circle cx="270" cy="150" r="4.5" fill="#27C7B8"/>
<circle cx="220" cy="180" r="6" fill="#101B2D" stroke="#27C7B8" stroke-width="2.5"/>
<circle cx="220" cy="180" r="2.5" fill="#27C7B8"/>
<circle cx="170" cy="150" r="4.5" fill="#27C7B8"/>
<circle cx="320" cy="70" r="4" fill="#8E7CFF"/>
<circle cx="350" cy="110" r="5" fill="#F4F7FB" stroke="#8E7CFF" stroke-width="2"/>
<circle cx="320" cy="150" r="4" fill="#8E7CFF"/>
<circle cx="110" cy="70" r="4.5" fill="#27C7B8"/>
<circle cx="70" cy="110" r="3.5" fill="#D9E2EF"/>
<circle cx="90" cy="30" r="3.5" fill="#D9E2EF"/>
<circle cx="130" cy="200" r="4" fill="#E5A34D"/>
<circle cx="80" cy="210" r="3" fill="#D9E2EF"/>
<circle cx="220" cy="230" r="4" fill="#27C7B8"/>
<circle cx="270" cy="250" r="3" fill="#D9E2EF"/>
<circle cx="370" cy="190" r="4.5" fill="#8E7CFF"/>
<circle cx="420" cy="180" r="3" fill="#D9E2EF"/>
<circle cx="380" cy="240" r="3" fill="#D9E2EF"/>
<circle cx="410" cy="90" r="3.5" fill="#D9E2EF"/>
<circle cx="220" cy="60" r="11" fill="none" stroke="#27C7B8" stroke-width="1" stroke-dasharray="2,3" opacity="0.6"/>
<circle cx="350" cy="110" r="10" fill="none" stroke="#8E7CFF" stroke-width="1" stroke-dasharray="2,3" opacity="0.6"/>
</svg>"""
    cleaned = _clean_svg(raw_svg)
    uri = _svg_to_data_uri(cleaned)
    return f'<img src="{uri}" width="{width}" height="{height}" style="max-width: 100%; height: auto; display: block;" alt="Computational Network Graph">'
