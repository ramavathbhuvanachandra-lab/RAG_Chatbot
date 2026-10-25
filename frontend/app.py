"""Main Streamlit Application Entry Point for the Modular Frontend.

Loads the active institution configuration, establishes session state, and
renders the sidebar and chat view.
"""

from __future__ import annotations

import streamlit as st

from frontend.components.chat import render_chat_view
from frontend.components.sidebar import render_sidebar
from frontend.core.config_loader import (
    get_active_tenant_id,
    load_institution_ui_config,
)
from frontend.core.session_manager import init_session_state
from frontend.styles import inject_custom_styles


def run_app() -> None:
    """Initialize and render the modular Streamlit frontend."""
    tenant_id = get_active_tenant_id()
    config = load_institution_ui_config(tenant_id)

    # Set page configuration if not already configured by root entry
    try:
        st.set_page_config(
            page_title=f"{config.display_name} AI Assistant",
            page_icon=config.logo_icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # set_page_config may only be invoked once per script run
        pass

    # Inject curated styling tokens
    inject_custom_styles(config)

    # Initialize temporary conversation session state
    init_session_state(config)

    # Render sidebar and chat views
    render_sidebar(config)
    render_chat_view(config)


if __name__ == "__main__":
    run_app()
