"""
IIT Jodhpur V1 — Chat UI

Purpose
-------
Manage Streamlit chat conversations while enforcing session-scoped
conversation memory.

Architecture
------------
Application chat
    -> persistent session_id
    -> session-specific persistent history
    -> bounded recent history
    -> structured chat_history
    -> existing graph
    -> grounded answer

Important invariants
--------------------
1. Each application chat owns its own persistent session_id.
2. A new chat never intentionally reuses another chat's session.
3. Conversation history is scoped to the active chat only.
4. The graph receives structured conversation messages, not a formatted
   string.
5. The current user question is not duplicated inside previous history.
6. Persistent history is authoritative when available.
7. Local chat history is used as a fallback if persistence is temporarily
   unavailable.
8. Conversation history is context only; retrieval evidence remains the
   source of factual answers.
9. Existing navigation and emergency routing continue to work.
10. No institution-specific conversational behavior is hardcoded here.
"""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Any

import streamlit as st

from backend.assistant_router import (
    assistant_router,
)

from backend.conversation_session import (
    ensure_chat_session,
    load_persistent_chat_history,
    recent_chat_history,
)


# =========================================================
# Configuration
# =========================================================

MAX_CONVERSATION_HISTORY = 8


# =========================================================
# Welcome message
# =========================================================

WELCOME_MESSAGE = (
    "👋 Hello! I am the IIT Jodhpur AI Assistant.\n\n"
    "Ask me anything about:\n"
    "• Admissions\n"
    "• Academics\n"
    "• Departments\n"
    "• Research\n"
    "• Hostel\n"
    "• Campus Facilities"
)


# =========================================================
# User identity
# =========================================================

def _get_current_user_id() -> str | None:
    """
    Resolve the application user identifier from Streamlit session state.

    Supports the known login-state key variants used by the application.
    """

    candidate_keys = (
        "user_id",
        "user_identifier",
        "logged_in_user_id",
    )

    for key in candidate_keys:

        value = st.session_state.get(
            key
        )

        if value:

            return str(
                value
            )

    return None


# =========================================================
# Persistent session binding
# =========================================================

def _ensure_chat_persistent_session(
    chat: dict[str, Any],
) -> str | None:
    """
    Ensure one application chat owns one persistent database session.

    Existing session_id:
        reused unchanged.

    Missing session_id:
        a new persistent session is created when a user_id is available.

    Persistence failure:
        returns None without breaking the UI.
    """

    user_id = _get_current_user_id()

    try:

        return ensure_chat_session(
            chat,
            user_id=user_id,
        )

    except Exception as exc:

        print(
            "Failed to ensure persistent chat session:",
            exc,
        )

        return None


# =========================================================
# Active chat helpers
# =========================================================

def _get_active_chat() -> dict[str, Any]:
    """
    Return the currently active application chat.
    """

    active_chat_id = (
        st.session_state.active_chat
    )

    return (
        st.session_state.conversations[
            active_chat_id
        ]
    )


def _get_local_previous_history(
    chat: dict[str, Any],
    current_question: str,
) -> list[dict[str, str]]:
    """
    Build previous conversation history from local Streamlit state.

    The current question is excluded because it has already been appended
    locally before this function runs.
    """

    messages = chat.get(
        "messages",
        [],
    )

    history: list[
        dict[str, str]
    ] = []

    for message in messages:

        if not isinstance(
            message,
            dict,
        ):
            continue

        role = str(
            message.get(
                "role",
                "",
            )
        ).strip().lower()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        if (
            role == "assistant"
            and
            "Hello! I am the IIT Jodhpur AI Assistant"
            in content
        ):
            continue

        history.append(
            {
                "role": role,
                "content": content,
            }
        )

    # -----------------------------------------------------
    # The current question has already been added locally.
    # Remove only the final matching user turn.
    # -----------------------------------------------------

    if history:

        last = history[-1]

        if (
            last["role"] == "user"
            and
            last["content"] == current_question
        ):

            history = history[
                :-1
            ]

    return history


def _select_previous_history(
    chat: dict[str, Any],
    session_id: str | None,
    current_question: str,
) -> list[dict[str, str]]:
    """
    Prefer persistent session history, with local chat state as a fallback.

    This makes the live UI resilient to transient persistence failures while
    preserving session isolation.
    """

    persistent_history: list[
        dict[str, str]
    ] = []

    if session_id:

        try:

            persistent_history = (
                load_persistent_chat_history(
                    session_id
                )
            )

        except Exception as exc:

            print(
                "Failed to load persistent history:",
                exc,
            )

            persistent_history = []

    # -----------------------------------------------------
    # Remove the current question if it was persisted before
    # graph execution.
    # -----------------------------------------------------

    if persistent_history:

        last = persistent_history[-1]

        if (
            last.get("role") == "user"
            and
            last.get("content") == current_question
        ):

            persistent_history = (
                persistent_history[:-1]
            )

    # -----------------------------------------------------
    # Persistent history is preferred when available.
    # -----------------------------------------------------

    if persistent_history:

        return recent_chat_history(
            persistent_history,
            max_messages=MAX_CONVERSATION_HISTORY,
        )

    # -----------------------------------------------------
    # Local fallback.
    # -----------------------------------------------------

    local_history = (
        _get_local_previous_history(
            chat,
            current_question,
        )
    )

    return recent_chat_history(
        local_history,
        max_messages=MAX_CONVERSATION_HISTORY,
    )


# =========================================================
# Initialize first chat
# =========================================================

def initialize_chat():
    """
    Initialize the first application chat.

    Each chat receives its own local chat ID and, when persistence is
    available, its own database session_id.
    """

    if (
        "conversations"
        in
        st.session_state
    ):
        return

    chat_id = str(
        uuid.uuid4()
    )

    chat = {
        "title": "New Chat",
        "session_id": None,
        "messages": [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ],
    }

    st.session_state.conversations = {
        chat_id: chat
    }

    st.session_state.active_chat = (
        chat_id
    )

    _ensure_chat_persistent_session(
        chat
    )


# =========================================================
# Create independent chat
# =========================================================

def create_new_chat():
    """
    Create a completely independent application chat.
    """

    chat_id = str(
        uuid.uuid4()
    )

    chat = {
        "title": "New Chat",
        "session_id": None,
        "messages": [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ],
    }

    st.session_state.conversations[
        chat_id
    ] = chat

    st.session_state.active_chat = (
        chat_id
    )

    _ensure_chat_persistent_session(
        chat
    )


# =========================================================
# Chat title
# =========================================================

def generate_chat_title(
    prompt: str,
) -> str:
    """
    Generate a compact title from the first user question.
    """

    title = str(
        prompt or ""
    ).strip()

    prefixes = (
        "tell me about",
        "can you tell me about",
        "what is",
        "what are",
        "give me",
        "explain",
        "explain about",
        "information about",
    )

    lower_title = title.lower()

    for prefix in prefixes:

        if lower_title.startswith(
            prefix
        ):

            title = title[
                len(prefix):
            ].strip()

            break

    title = title.strip(
        " ?!.,:"
    )

    if title:
        title = title.title()

    if len(title) > 30:
        title = title[:30] + "..."

    if not title:
        title = "New Chat"

    return title


# =========================================================
# Display active conversation
# =========================================================

def display_chat_history():
    """
    Render only the messages belonging to the active application chat.
    """

    chat = _get_active_chat()

    messages = chat.get(
        "messages",
        [],
    )

    for message in messages:

        if not isinstance(
            message,
            dict,
        ):
            continue

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(
            role
        ):

            st.markdown(
                content
            )

            response_time = (
                message.get(
                    "response_time"
                )
            )

            if (
                role == "assistant"
                and
                isinstance(
                    response_time,
                    (int, float),
                )
            ):

                st.caption(
                    "⏱️ Response Time: "
                    f"{response_time:.2f} seconds"
                )


# =========================================================
# Structured response rendering
# =========================================================

def _render_structured_response(
    route: dict[str, Any],
) -> str:
    """
    Render structured module output such as navigation and emergency data.
    """

    final_response = ""

    for item in route.get(
        "structured",
        [],
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        item_type = item.get(
            "type"
        )

        data = item.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        # -------------------------------------------------
        # Navigation
        # -------------------------------------------------

        if item_type == "navigation":

            name = data.get(
                "name"
            )

            if name:

                final_response += (
                    f"📍 **{name}**\n\n"
                )

            description = data.get(
                "description"
            )

            if description:

                final_response += (
                    f"{description}\n\n"
                )

            timings = data.get(
                "timings"
            )

            if timings:

                final_response += (
                    f"🕒 **Timings:** "
                    f"{timings}\n\n"
                )

            google_maps = data.get(
                "google_maps"
            )

            if google_maps:

                final_response += (
                    "🗺️ **Google Maps:**\n"
                    f"{google_maps}\n\n"
                )

        # -------------------------------------------------
        # Emergency
        # -------------------------------------------------

        elif item_type == "emergency":

            name = data.get(
                "name"
            )

            if name:

                final_response += (
                    f"📍 **{name}**\n\n"
                )

            description = data.get(
                "description"
            )

            if description:

                final_response += (
                    f"{description}\n\n"
                )

            timings = data.get(
                "timings"
            )

            if timings:

                final_response += (
                    f"🕒 **Timings:** "
                    f"{timings}\n\n"
                )

            google_maps = data.get(
                "google_maps"
            )

            if google_maps:

                final_response += (
                    "🗺️ **Google Maps:**\n"
                    f"{google_maps}\n\n"
                )

    return final_response.strip()


# =========================================================
# Generate backend response
# =========================================================

def get_response(
    question: str,
    graph,
):
    """
    Generate an answer for the currently active application chat.

    Critical Phase-7 contract:

        session
            -> structured previous history
            -> graph
            -> conversation resolver
            -> resolved question
            -> retrieval
            -> answer
    """

    try:

        chat = _get_active_chat()

        # -------------------------------------------------
        # Resolve this chat's persistent session.
        # -------------------------------------------------

        session_id = (
            _ensure_chat_persistent_session(
                chat
            )
        )

        # -------------------------------------------------
        # Load ONLY this chat/session's previous history.
        # -------------------------------------------------

        previous_history = (
            _select_previous_history(
                chat=chat,
                session_id=session_id,
                current_question=question,
            )
        )

        # -------------------------------------------------
        # Structured history is intentionally passed directly.
        #
        # DO NOT convert this into a string.
        # -------------------------------------------------

        route = assistant_router(
            question
        )

        final_response = (
            _render_structured_response(
                route
            )
        )

        total_time = 0.0

        # -------------------------------------------------
        # Existing RAG graph.
        # -------------------------------------------------

        if route.get(
            "need_rag",
            True,
        ):

            started = time.perf_counter()

            result = graph.invoke(
                {
                    "question": question,
                    "chat_history": previous_history,
                }
            )

            total_time = (
                time.perf_counter()
                - started
            )

            answer = str(
                result.get(
                    "answer",
                    "",
                )
            ).strip()

            if final_response:

                final_response += (
                    "\n\n---\n\n"
                )

            final_response += (
                answer
            )

        return (
            final_response.strip(),
            total_time,
        )

    except Exception as exc:

        traceback.print_exc()

        return (
            f"❌ Error:\n\n{exc}",
            0.0,
        )


# =========================================================
# User prompt handler
# =========================================================

def handle_user_prompt(
    prompt: str,
    graph,
):
    """
    Execute one complete conversational turn.

    Order:

        1. Identify active chat.
        2. Ensure its persistent session.
        3. Save the user message.
        4. Generate the answer using previous turns.
        5. Save the assistant answer to the same session.
    """

    prompt = str(
        prompt or ""
    ).strip()

    if not prompt:
        return

    # -----------------------------------------------------
    # Resolve active chat.
    # -----------------------------------------------------

    active_chat_id = (
        st.session_state.active_chat
    )

    chat = (
        st.session_state.conversations[
            active_chat_id
        ]
    )

    messages = chat.setdefault(
        "messages",
        [],
    )

    # -----------------------------------------------------
    # Resolve this chat's session.
    # -----------------------------------------------------

    session_id = (
        _ensure_chat_persistent_session(
            chat
        )
    )

    # -----------------------------------------------------
    # Save user message locally BEFORE graph execution.
    # -----------------------------------------------------

    user_message = {
        "role": "user",
        "content": prompt,
    }

    messages.append(
        user_message
    )

    # -----------------------------------------------------
    # Persist user message.
    # -----------------------------------------------------

    if session_id:

        try:

            from backend.persistence import (
                safe_save_message,
            )

            safe_save_message(
                session_id=session_id,
                role="user",
                message=prompt,
            )

        except Exception as exc:

            print(
                "Failed to persist user message:",
                exc,
            )

    # -----------------------------------------------------
    # Generate title from first user turn.
    # -----------------------------------------------------

    if (
        chat.get(
            "title"
        )
        == "New Chat"
    ):

        chat["title"] = (
            generate_chat_title(
                prompt
            )
        )

    # -----------------------------------------------------
    # Display user message.
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )

    # -----------------------------------------------------
    # Generate assistant response.
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            answer, response_time = (
                get_response(
                    question=prompt,
                    graph=graph,
                )
            )

        # -------------------------------------------------
        # Display answer.
        # -------------------------------------------------

        st.markdown(
            answer
        )

        st.caption(
            "⏱️ Response Time: "
            f"{response_time:.2f} seconds"
        )

        # -------------------------------------------------
        # Store assistant response locally.
        # -------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "response_time": response_time,
            "message_id": None,
        }

        messages.append(
            assistant_message
        )

        # -------------------------------------------------
        # Persist assistant response in SAME session.
        # -------------------------------------------------

        if session_id:

            try:

                from backend.persistence import (
                    safe_save_message,
                )

                saved_message = (
                    safe_save_message(
                        session_id=session_id,
                        role="assistant",
                        message=answer,
                        response_time=response_time,
                    )
                )

                if saved_message:

                    assistant_message[
                        "message_id"
                    ] = saved_message.get(
                        "id"
                    )

            except Exception as exc:

                print(
                    "Failed to persist assistant message:",
                    exc,
                )