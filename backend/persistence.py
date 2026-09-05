import logging

from backend.user_db import get_or_create_user
from backend.session_db import create_session
from backend.message_db import save_message
from backend.feedback_db import save_feedback


logger = logging.getLogger(__name__)


def safe_get_or_create_user(name: str, phone: str):
    """
    Best-effort user persistence.

    Returns:
        user record when Supabase is available.
        None when persistence is unavailable.

    The chatbot must not depend on this succeeding.
    """

    try:
        return get_or_create_user(
            name=name,
            phone=phone,
        )

    except Exception as exc:
        logger.warning(
            "User persistence unavailable: %s",
            exc,
        )

        return None


def safe_create_session(user_id: str):
    """
    Best-effort session persistence.

    Returns:
        session record when Supabase is available.
        None when persistence is unavailable.
    """

    try:
        return create_session(
            user_id=user_id,
        )

    except Exception as exc:
        logger.warning(
            "Session persistence unavailable: %s",
            exc,
        )

        return None


def safe_save_message(
    session_id: str,
    role: str,
    message: str,
    response_time: float = None,
):
    """
    Best-effort message persistence.

    Failure must never prevent the chatbot from returning
    an answer to the student.
    """

    try:
        return save_message(
            session_id=session_id,
            role=role,
            message=message,
            response_time=response_time,
        )

    except Exception as exc:
        logger.warning(
            "Message persistence unavailable: %s",
            exc,
        )

        return None


def safe_save_feedback(
    message_id: str,
    session_id: str,
    feedback: str,
):
    """
    Best-effort feedback persistence.

    Failure must never break the chat experience.
    """

    try:
        return save_feedback(
            message_id=message_id,
            session_id=session_id,
            feedback=feedback,
        )

    except Exception as exc:
        logger.warning(
            "Feedback persistence unavailable: %s",
            exc,
        )

        return None