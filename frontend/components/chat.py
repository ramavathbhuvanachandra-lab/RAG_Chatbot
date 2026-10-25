"""Chat View Layout and Turn Execution Component.

Coordinates single-page welcome screen presentation, suggested question cards,
continuous message history presentation, persistent chat input, and canonical
assistant service turn execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import streamlit as st

from frontend.clients.assistant_client import send_chat_turn
from frontend.components.diagnostics import render_developer_trace_inspector
from frontend.components.messages import render_chat_messages
from frontend.components.quick_actions import render_quick_actions
from frontend.components.structured_cards import render_campus_map_grid
from frontend.core.session_manager import (
    add_assistant_message,
    add_user_message,
    get_active_conversation,
    get_bounded_history,
)

if TYPE_CHECKING:
    from frontend.core.models import InstitutionUIConfig


def _process_user_turn(prompt: str, config: InstitutionUIConfig) -> None:
    """Execute a single conversational turn through the canonical assistant service."""
    clean_prompt = " ".join(str(prompt or "").strip().split())
    if not clean_prompt:
        return

    if st.session_state.get("_turn_in_progress", False):
        return
    st.session_state["_turn_in_progress"] = True

    try:
        conv = get_active_conversation()

        # 1. Capture history before appending current question
        history = get_bounded_history()

        # 2. Add user question to conversation state
        add_user_message(clean_prompt)
        with st.chat_message("user"):
            st.markdown(clean_prompt)

        # 3. Call canonical assistant service under processing spinner with terminal fallback
        response = None
        try:
            with st.chat_message("assistant"):
                with st.spinner(f"Consulting {config.display_name} knowledge base..."):
                    response = send_chat_turn(
                        question=clean_prompt,
                        history=history,
                        session_id=conv.id,
                        institution_id=config.institution_id,
                    )
        except Exception as turn_exc:
            from frontend.clients.assistant_client import AssistantResponse
            logger = logging.getLogger(__name__)
            logger.exception("Chat turn execution failed for question: %s", clean_prompt)
            err_msg = (
                "⚠️ An unexpected error occurred while processing your request. "
                "Please try rephrasing your question or submitting again."
            )
            response = AssistantResponse(
                answer=err_msg,
                formatted_response=err_msg,
                is_safe_refusal=True,
                evidence_status="exception",
            )

        # 4. Add assistant response to conversation state to guarantee terminal state
        if response is not None:
            add_assistant_message(response)
    finally:
        st.session_state["_turn_in_progress"] = False


def render_chat_view(config: InstitutionUIConfig) -> None:
    """Render the primary single-page chat view container."""
    conv = get_active_conversation()
    user_turns = [m for m in conv.messages if m.role == "user"]
    has_active_conversation = len(user_turns) > 0

    # 1. Campus Map Mode (if toggled)
    if getattr(st.session_state, "show_campus_map", False) and config.has_campus_map:
        st.markdown(
            f"""
            <div class="brand-header-compact">
                <span class="brand-header-title">🗺️ {config.display_name} Campus Map & Locations</span>
                <span class="brand-header-badge">Interactive Directory</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("← Back to Chat", key="btn_exit_map"):
            st.session_state.show_campus_map = False
            st.rerun()

        selected_loc_query = render_campus_map_grid(config.campus_map_locations)
        if selected_loc_query:
            st.session_state.show_campus_map = False
            _process_user_turn(selected_loc_query, config)
            st.rerun()
        return

    selected_prompt: str | None = None

    # 2. Main Content: Welcome Screen vs Active Continuous Chat
    if not has_active_conversation:
        # Welcome Screen
        st.markdown(
            f"""
            <div class="welcome-hero-card">
                <div class="welcome-badge">🎓 Official AI Assistant</div>
                <h1 class="welcome-hero-title">Welcome to {config.display_name}</h1>
                <p class="welcome-hero-desc">
                    The {config.display_name} AI Assistant helps students explore institute information,
                    academics, admissions, programmes, campus facilities, fees, research, and other
                    available institutional information.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested Questions Grid
        selected_prompt = render_quick_actions(config)

    else:
        # Continuous Chat Header
        st.markdown(
            f"""
            <div class="brand-header-compact">
                <span class="brand-header-title">{config.logo_icon} {config.display_name} AI Assistant</span>
                <span class="brand-header-badge">Continuous Session</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render Active Conversation Messages chronologically
        first_user_idx = next(i for i, m in enumerate(conv.messages) if m.role == "user")
        active_messages = conv.messages[first_user_idx:]
        clicked_followup = render_chat_messages(active_messages)
        if clicked_followup:
            selected_prompt = clicked_followup

    # 2.5. Developer Diagnostics Trace Inspector (when enabled)
    render_developer_trace_inspector()

    # 3. Persistent Chat Input Bar
    placeholder = f"Ask anything about {config.display_name} (admissions, academics, hostels, campus)..."
    prompt = st.chat_input(placeholder)

    # 4. Turn Execution
    query_to_process = prompt or selected_prompt
    if query_to_process:
        st.session_state.pop("pending_prompt", None)
        _process_user_turn(query_to_process, config)
        st.rerun()


__all__ = ["render_chat_view"]
