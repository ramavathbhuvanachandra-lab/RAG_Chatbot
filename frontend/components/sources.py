"""Source and Citation Presentation Component.

Renders verified citation units transparently to provide auditability
without cluttering the main conversation message flow.
"""

from __future__ import annotations

from typing import Any
import streamlit as st


def render_sources_accordion(sources: list[dict[str, Any]]) -> None:
    """Render verified evidence sources in a collapsible citation drawer."""
    if not sources:
        return

    count = len(sources)
    with st.expander(f"📚 Supporting Citations ({count})", expanded=False):
        for i, src in enumerate(sources, 1):
            doc_name = src.get("source") or "Official Institute Document"
            heading = src.get("heading") or "General Section"
            excerpt = src.get("text") or ""

            st.markdown(
                f"""
                <div class="citation-box">
                    <span class="citation-source">[{i}] {doc_name}</span> &bull; <em>{heading}</em><br/>
                    <span style="opacity: 0.9;">"{excerpt.strip()}"</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


__all__ = ["render_sources_accordion"]
