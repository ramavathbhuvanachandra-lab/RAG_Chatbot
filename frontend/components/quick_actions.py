"""Quick Actions Component.

Renders configured suggested queries and optional interactive campus map triggers
when a conversation is fresh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import streamlit as st

if TYPE_CHECKING:
    from frontend.core.models import InstitutionUIConfig


def _handle_action_click(prompt: str) -> None:
    """Explicit Streamlit callback executed synchronously when a card is clicked."""
    st.session_state["pending_prompt"] = prompt
    st.session_state["_turn_in_progress"] = False


def render_quick_actions(config: InstitutionUIConfig) -> str | None:
    """Render clickable suggested-question cards in a responsive grid.

    Returns the selected prompt query if a card is clicked, or None.
    """
    st.markdown("### What would you like to know?")

    if not config.quick_actions:
        return None

    # Check if a prompt was queued via on_click callback
    selected_prompt: str | None = st.session_state.pop("pending_prompt", None)

    # Render configured quick-action cards in 2 responsive columns
    cols = st.columns(2)

    for i, action in enumerate(config.quick_actions):
        with cols[i % 2]:
            label = f"{action.icon}  {action.label}".strip()
            clicked = st.button(
                label,
                key=f"quick_action_{action.id}",
                use_container_width=True,
                help=action.prompt,
                on_click=_handle_action_click,
                args=(action.prompt,),
            )
            if clicked and not selected_prompt:
                selected_prompt = action.prompt

    # Optional Campus Map trigger button
    if config.has_campus_map:
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        if st.button(
            "🗺️ Campus Map & Location Directory",
            use_container_width=True,
            key="qa_campus_map_btn",
            help="Explore IIT Jodhpur campus buildings, lecture halls, hostels, and dining facilities.",
        ):
            st.session_state.show_campus_map = True
            st.rerun()

    return selected_prompt



__all__ = ["render_quick_actions"]
