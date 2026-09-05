"""
Supabase persistence integration tests.

These tests verify that the application's current database layer can:
1. Find/create a user.
2. Create a session.
3. Save user and assistant messages.
4. Read chat history.
5. Save feedback.

The test creates its own temporary test user/session so it does not
depend on an existing production record.
"""

import uuid

from backend.user_db import get_or_create_user
from backend.session_db import create_session
from backend.message_db import save_message, get_chat_history
from backend.feedback_db import save_feedback


def test_supabase_persistence_flow():
    unique_phone = f"999{uuid.uuid4().int % 10**7:07d}"

    # ---------------------------------------------------------
    # Create or retrieve test user
    # ---------------------------------------------------------
    user = get_or_create_user(
        name="IITJ DB Test User",
        phone=unique_phone,
    )

    assert user is not None
    assert user["user_id"]
    assert user["phone"] == unique_phone

    # ---------------------------------------------------------
    # Create test session
    # ---------------------------------------------------------
    session = create_session(
        user_id=user["user_id"]
    )

    assert session is not None
    assert session["session_id"]
    assert session["user_id"] == user["user_id"]

    session_id = session["session_id"]

    # ---------------------------------------------------------
    # Save user message
    # ---------------------------------------------------------
    user_message = save_message(
        session_id=session_id,
        role="user",
        message="TEST: What is IIT Jodhpur?",
    )

    assert user_message is not None
    assert user_message["id"]

    # ---------------------------------------------------------
    # Save assistant message
    # ---------------------------------------------------------
    assistant_message = save_message(
        session_id=session_id,
        role="assistant",
        message="TEST: IIT Jodhpur is an institute of national importance.",
        response_time=1.23,
    )

    assert assistant_message is not None
    assert assistant_message["id"]

    # ---------------------------------------------------------
    # Read chat history
    # ---------------------------------------------------------
    history = get_chat_history(
        session_id=session_id
    )

    assert history is not None
    assert len(history) >= 2

    messages = [row["message"] for row in history]

    assert "TEST: What is IIT Jodhpur?" in messages
    assert (
        "TEST: IIT Jodhpur is an institute of national importance."
        in messages
    )

    # ---------------------------------------------------------
    # Save feedback
    # ---------------------------------------------------------
    feedback = save_feedback(
        message_id=assistant_message["id"],
        session_id=session_id,
        feedback="up",
    )

    assert feedback is not None