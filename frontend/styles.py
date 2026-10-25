"""Frontend UI Styling and CSS Design System.

Provides modern styling tokens, responsive container layouts, glassmorphic card
treatments, and clean chat bubbles across light and dark Streamlit themes.
"""

from __future__ import annotations

import streamlit as st
from frontend.core.models import InstitutionUIConfig


def inject_custom_styles(config: InstitutionUIConfig) -> None:
    """Inject custom CSS rules customized for the given institution branding."""
    primary_color = config.primary_color or "#0F294A"
    accent_color = config.accent_color or "#2563EB"

    css = f"""
    <style>
    /* -------------------------------------------------------------
       Streamlit Global Adjustments
    ------------------------------------------------------------- */
    header[data-testid="stHeader"] {{
        background: transparent;
        visibility: visible;
        height: 2.5rem;
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 950px;
    }}

    /* Sidebar width and subtle border */
    section[data-testid="stSidebar"] {{
        width: 320px !important;
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }}

    /* -------------------------------------------------------------
       Welcome Hero Screen & Compact Header
    ------------------------------------------------------------- */
    .welcome-hero-card {{
        background: linear-gradient(135deg, rgba(15, 41, 74, 0.08) 0%, rgba(37, 99, 235, 0.06) 100%);
        border: 1px solid rgba(59, 130, 246, 0.22);
        border-radius: 16px;
        padding: 2.2rem 2.2rem 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    }}

    .welcome-badge {{
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        background: rgba(37, 99, 235, 0.12);
        color: #2563EB;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }}

    .welcome-hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0 0.8rem 0;
        letter-spacing: -0.02em;
        color: inherit;
    }}

    .welcome-hero-desc {{
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 700px;
        margin: 0 auto;
        opacity: 0.86;
        font-weight: 400;
    }}

    .welcome-section-header {{
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .brand-header-compact {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.65rem 1.2rem;
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.06) 0%, rgba(59, 130, 246, 0.03) 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }}

    .brand-header-title {{
        font-weight: 700;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .brand-header-badge {{
        font-size: 0.74rem;
        padding: 3px 10px;
        border-radius: 12px;
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}

    /* -------------------------------------------------------------
       Quick Action & Suggested Question Cards
    ------------------------------------------------------------- */
    .quick-actions-container {{
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }}

    div[data-testid="column"] .stButton > button {{
        min-height: 3.5rem;
        text-align: left;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background: rgba(128, 128, 128, 0.03);
        transition: all 0.2s ease-in-out;
        font-weight: 500;
        width: 100%;
    }}

    div[data-testid="column"] .stButton > button:hover {{
        border-color: #2563EB;
        background: rgba(37, 99, 235, 0.05);
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.12);
    }}

    .stButton > button {{
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
        font-weight: 500;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }}

    .stButton > button:hover {{
        border-color: {accent_color};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }}

    /* -------------------------------------------------------------
       Structured Cards (Navigation & Emergency)
    ------------------------------------------------------------- */
    .structured-card {{
        background: rgba(128, 128, 128, 0.05);
        border-left: 4px solid {accent_color};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0;
    }}

    .structured-card-title {{
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.4rem;
    }}

    /* -------------------------------------------------------------
       Source & Citation Expander
    ------------------------------------------------------------- */
    .citation-box {{
        background: rgba(128, 128, 128, 0.04);
        border-left: 3px solid #10B981;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }}

    .citation-source {{
        font-weight: 600;
        color: #10B981;
    }}

    /* -------------------------------------------------------------
       Metadata Badge
    ------------------------------------------------------------- */
    .meta-badge {{
        display: inline-block;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        border-radius: 12px;
        background: rgba(128, 128, 128, 0.12);
        opacity: 0.85;
        margin-top: 0.4rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


__all__ = ["inject_custom_styles"]
