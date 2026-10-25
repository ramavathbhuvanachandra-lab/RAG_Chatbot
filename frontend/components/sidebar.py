"""Sidebar Navigation and Conversation Switcher.

Provides new-chat instantiation, multi-chat conversation switching, supported
topics overview, and institutional tenant metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import streamlit as st

from frontend.core.session_manager import (
    reset_conversation,
)

if TYPE_CHECKING:
    from frontend.core.models import InstitutionUIConfig


def render_sidebar(config: InstitutionUIConfig) -> None:
    """Render the application sidebar with chat controls and institutional information."""
    with st.sidebar:
        # 1. Institutional Branding Header
        st.markdown(f"### {config.logo_icon} {config.display_name}")
        if config.tagline:
            st.caption(config.tagline)

        st.markdown("---")

        # 2. Single Continuous Conversation Reset Control
        if st.button(
            "🔄 Restart Conversation",
            use_container_width=True,
            key="btn_restart_chat",
            help="Start a fresh conversation and reset message history.",
        ):
            reset_conversation(config)
            st.rerun()

        st.markdown("---")

        # 4. Supported Topics & Knowledge Scope
        if config.supported_topics:
            with st.expander("ℹ️ Supported Topics", expanded=False):
                for topic in config.supported_topics:
                    st.markdown(f"• {topic}")

        # 5. Developer Mode Diagnostics Toggle
        st.markdown("---")
        st.markdown("##### ⚙️ Developer Tools")
        st.toggle(
            "Show Retrieval Diagnostics",
            key="show_retrieval_diagnostics",
            value=False,
            help="Expose Top-15 candidate rankings, verification decisions, and context packaging metrics."
        )

        st.markdown("---")

        # 6. Institutional Footer
        st.caption(config.footer_text)
        st.caption(f"Active Tenant: `{config.institution_id}`")


__all__ = ["render_sidebar"]
