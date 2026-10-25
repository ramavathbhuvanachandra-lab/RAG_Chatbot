"""Root Streamlit Application Launcher.

Provides the primary user entry point for 'streamlit run app.py'.
Configures the Streamlit page and delegates execution to the modular frontend.
"""

from __future__ import annotations

import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration (Must execute before any rendering commands)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IITJ Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Canonical Engine Import (Maintains canonical architecture contract)
# -----------------------------------------------------------------------------
from ai_platform.runtime.engine import get_app_graph as create_graph

# -----------------------------------------------------------------------------
# 3. Modular Frontend Entry Point
# -----------------------------------------------------------------------------
from frontend.app import run_app

# Execute the modular frontend
run_app()