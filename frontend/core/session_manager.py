"""Frontend Session and Conversation Manager.

Manages temporary conversation lifecycles, chat switching, session ID generation,
and bounded context windowing in Streamlit session state without database dependencies.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import streamlit as st

from frontend.core.models import ChatMessage, Conversation

if TYPE_CHECKING:
    from frontend.clients.assistant_client import AssistantResponse
    from frontend.core.models import InstitutionUIConfig

MAX_HISTORY_TURNS = 8
MAX_EXECUTION_TRACES = 15


def init_session_state(config: InstitutionUIConfig) -> None:
    """Initialize or reset Streamlit session state for the active institution."""
    current_inst = getattr(st.session_state, "active_institution_id", None)
    if current_inst != config.institution_id:
        # Tenant changed: isolate state and create fresh conversation for new tenant
        st.session_state.active_institution_id = config.institution_id
        st.session_state.conversations = {}
        st.session_state.active_chat_id = None
        st.session_state.show_campus_map = False
        st.session_state.execution_traces = []
        st.session_state.selected_trace_id = None

    if "conversations" not in st.session_state or not st.session_state.conversations:
        st.session_state.conversations = {}
        create_new_conversation(config)

    if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.conversations:
        first_id = next(iter(st.session_state.conversations.keys()))
        st.session_state.active_chat_id = first_id

    if "show_campus_map" not in st.session_state:
        st.session_state.show_campus_map = False

    if "execution_traces" not in st.session_state:
        st.session_state.execution_traces = []

    if "selected_trace_id" not in st.session_state:
        st.session_state.selected_trace_id = None

    if "_turn_in_progress" not in st.session_state:
        st.session_state["_turn_in_progress"] = False

    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None


def create_new_conversation(
    config: InstitutionUIConfig,
    title: str = "New Chat",
) -> str:
    """Create a new independent conversation thread with a unique session ID."""
    conv_id = f"chat_{uuid.uuid4().hex[:10]}"
    welcome_text = config.welcome_message or (
        f"👋 Hello! I am the **{config.display_name}** AI Assistant.\n\n"
        "Ask me anything about admissions, academics, hostels, research, or campus facilities."
    )
    welcome_msg = ChatMessage(
        role="assistant",
        content=welcome_text,
    )
    conv = Conversation(
        id=conv_id,
        title=title,
        messages=[welcome_msg],
    )
    st.session_state.conversations[conv_id] = conv
    st.session_state.active_chat_id = conv_id
    st.session_state.show_campus_map = False
    st.session_state["_turn_in_progress"] = False
    st.session_state["pending_prompt"] = None
    return conv_id


def reset_conversation(config: InstitutionUIConfig) -> str:
    """Reset the current conversation to a fresh welcome state with a new session ID."""
    st.session_state.conversations = {}
    st.session_state.active_chat_id = None
    st.session_state.show_campus_map = False
    st.session_state.execution_traces = []
    st.session_state.selected_trace_id = None
    st.session_state["_turn_in_progress"] = False
    st.session_state["pending_prompt"] = None
    return create_new_conversation(config)



def get_active_conversation() -> Conversation:
    """Retrieve the currently active conversation object."""
    active_id = st.session_state.active_chat_id
    return st.session_state.conversations[active_id]


def switch_conversation(chat_id: str) -> None:
    """Switch active conversation to the specified chat ID."""
    if chat_id in st.session_state.conversations:
        st.session_state.active_chat_id = chat_id
        st.session_state.show_campus_map = False


def add_user_message(text: str) -> ChatMessage:
    """Record a user query into the active conversation."""
    conv = get_active_conversation()
    msg = ChatMessage(role="user", content=text)
    conv.add_message(msg)

    # Automatically set conversational title from first user turn
    user_turns = [m for m in conv.messages if m.role == "user"]
    if len(user_turns) == 1:
        clean_title = text.strip().replace("\n", " ")
        if len(clean_title) > 30:
            clean_title = clean_title[:27] + "..."
        conv.title = clean_title

    return msg


def add_assistant_message(response: AssistantResponse) -> ChatMessage:
    """Record the assistant's response into the active conversation and capture execution trace."""
    import datetime

    conv = get_active_conversation()
    diagnostics = getattr(response, "diagnostics", None)

    # Associate trace with turn metadata
    user_turns = [m for m in conv.messages if m.role == "user"]
    last_query = user_turns[-1].content if user_turns else "Initial Query"
    turn_num = len(user_turns)
    turn_id = (
        str(diagnostics.get("turn_id"))
        if diagnostics and isinstance(diagnostics, dict) and diagnostics.get("turn_id")
        else f"turn_{turn_num}_{uuid.uuid4().hex[:6]}"
    )
    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    if diagnostics and isinstance(diagnostics, dict):
        diagnostics["turn_id"] = turn_id
        diagnostics["turn_number"] = turn_num
        diagnostics["timestamp"] = now_str
        diagnostics["user_query"] = last_query
        terminal_status = str(
            diagnostics.get("terminal_status")
            or ("safe_refusal" if response.is_safe_refusal else "success")
        )

        trace_record = {
            "turn_id": turn_id,
            "turn_number": turn_num,
            "query": last_query,
            "query_preview": (last_query[:45] + "...") if len(last_query) > 45 else last_query,
            "timestamp": now_str,
            "diagnostics": diagnostics,
            "latency_seconds": response.latency_seconds,
            "session_id": conv.id,
            "terminal_status": terminal_status,
        }

        if "execution_traces" not in st.session_state:
            st.session_state.execution_traces = []

        st.session_state.execution_traces.append(trace_record)
        if len(st.session_state.execution_traces) > MAX_EXECUTION_TRACES:
            st.session_state.execution_traces = st.session_state.execution_traces[-MAX_EXECUTION_TRACES:]
        st.session_state.selected_trace_id = turn_id

    msg = ChatMessage(
        role="assistant",
        content=response.formatted_response,
        response_time=response.latency_seconds,
        structured_blocks=response.structured_blocks,
        sources=response.sources,
        is_safe_refusal=response.is_safe_refusal,
        evidence_status=response.evidence_status,
        suggested_questions=getattr(response, "suggested_questions", []),
        diagnostics=diagnostics,
    )
    conv.add_message(msg)
    return msg


def get_bounded_history(max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, str]]:
    """Extract bounded message history formatted for the canonical assistant service.

    Only includes messages from the active conversation, strictly isolating chats.
    """
    conv = get_active_conversation()
    # Exclude the very first greeting message if it is generic assistant welcome
    relevant_messages = conv.messages
    if relevant_messages and relevant_messages[0].role == "assistant" and len(relevant_messages) > 1:
        relevant_messages = relevant_messages[1:]

    max_messages = max_turns * 2
    window = relevant_messages[-max_messages:] if len(relevant_messages) > max_messages else relevant_messages

    return [{"role": m.role, "content": m.content} for m in window]


__all__ = [
    "MAX_HISTORY_TURNS",
    "init_session_state",
    "create_new_conversation",
    "reset_conversation",
    "get_active_conversation",
    "switch_conversation",
    "add_user_message",
    "add_assistant_message",
    "get_bounded_history",
]
