"""
IIT Jodhpur V1 — Conversation Session Management

Purpose
-------
Provide the session-scoped conversation boundary between the Streamlit UI,
persistent message storage, and the conversational graph.

Architecture
------------
UI chat
    -> application chat_id
        -> persistent database session_id
            -> session-specific message history
                -> controlled recent history
                    -> conversation resolver / graph

Important invariants
--------------------
1. Every application chat owns at most one persistent session_id.
2. Different application chats must never intentionally share a session.
3. History loading is always scoped by session_id.
4. Long-term history remains in persistence; only a bounded recent window
   is passed into the conversational graph.
5. Persistence is best-effort and must never prevent the chatbot from
   answering.
6. Database modules are imported lazily so pure unit tests do not require
   Supabase environment variables.
7. This module contains no institution-specific knowledge.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


# =========================================================
# Configuration
# =========================================================

SESSION_ID_KEY = "session_id"

DEFAULT_HISTORY_LIMIT = 8


# =========================================================
# Persistent session creation
# =========================================================

def create_persistent_chat_session(
    user_id: str | None,
) -> str | None:
    """
    Create one persistent database session for an application chat.

    The database dependency is imported lazily so importing this module
    does not require a configured Supabase client.

    Returns:
        Application-level session ID when creation succeeds.
        None when persistence is unavailable.
    """

    if not user_id:
        logger.warning(
            "Cannot create persistent chat session without user_id."
        )

        return None

    try:
        # -------------------------------------------------
        # Lazy import:
        # prevents Supabase initialization during pure unit tests.
        # -------------------------------------------------

        from backend.persistence import (
            safe_create_session,
        )

        session = safe_create_session(
            user_id=str(user_id)
        )

    except Exception as exc:
        logger.warning(
            "Persistent chat session creation failed: %s",
            exc,
        )

        return None

    if not session:
        return None

    session_id = session.get(
        SESSION_ID_KEY
    )

    if not session_id:
        logger.warning(
            "Persistent session record did not contain session_id."
        )

        return None

    return str(
        session_id
    )


# =========================================================
# Application-chat session binding
# =========================================================

def ensure_chat_session(
    chat: dict[str, Any],
    *,
    user_id: str | None = None,
) -> str | None:
    """
    Ensure an application chat owns one persistent session.

    Existing session_id:
        reused unchanged.

    Missing session_id:
        creates a new persistent session when user_id is available.

    Persistence unavailable:
        returns None without breaking the chat.
    """

    if not isinstance(
        chat,
        dict,
    ):
        raise TypeError(
            "chat must be a dictionary"
        )

    existing_session_id = chat.get(
        SESSION_ID_KEY
    )

    if existing_session_id:
        return str(
            existing_session_id
        )

    if not user_id:
        return None

    session_id = (
        create_persistent_chat_session(
            user_id=str(user_id)
        )
    )

    if session_id:
        chat[
            SESSION_ID_KEY
        ] = session_id

    return session_id


# =========================================================
# Persistent history loading
# =========================================================

def load_persistent_chat_history(
    session_id: str | None,
) -> list[dict[str, str]]:
    """
    Load messages belonging only to one persistent session.

    Returned format:

        [
            {
                "role": "user",
                "content": "...",
            },
            {
                "role": "assistant",
                "content": "...",
            },
        ]

    Database access is lazy and failures return an empty history.
    """

    if not session_id:
        return []

    try:
        # -------------------------------------------------
        # Lazy import:
        # prevents database initialization during test collection.
        # -------------------------------------------------

        from backend.message_db import (
            get_chat_history,
        )

        rows = get_chat_history(
            str(session_id)
        )

    except Exception as exc:
        logger.warning(
            "Chat history load failed for session %s: %s",
            session_id,
            exc,
        )

        return []

    history: list[
        dict[str, str]
    ] = []

    for row in rows or []:

        if not isinstance(
            row,
            dict,
        ):
            continue

        role = str(
            row.get(
                "role",
                "",
            )
        ).strip().lower()

        content = str(
            row.get(
                "message",
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

        history.append(
            {
                "role": role,
                "content": content,
            }
        )

    return history


# =========================================================
# Controlled conversational window
# =========================================================

def recent_chat_history(
    history: list[dict[str, str]],
    *,
    max_messages: int = DEFAULT_HISTORY_LIMIT,
) -> list[dict[str, str]]:
    """
    Return only the latest bounded history window.

    The database retains the complete conversation.
    The graph receives only the recent context required for conversational
    resolution.
    """

    if not history:
        return []

    if max_messages <= 0:
        return []

    return list(
        history[
            -max_messages:
        ]
    )