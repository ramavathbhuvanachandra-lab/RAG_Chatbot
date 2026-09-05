import logging
import uuid
from typing import Dict, Any

from backend.persistence import (
    safe_get_or_create_user,
    safe_create_session,
)


logger = logging.getLogger(__name__)


def authenticate_user(
    name: str,
    phone: str,
) -> Dict[str, Any]:
    """
    Authenticate a demo user using phone number.

    Normal mode:
        Supabase creates/retrieves the real user and session.

    Degraded mode:
        If Supabase is unavailable, create a temporary local
        user/session so the chatbot can still operate.

    This function does not perform any Streamlit/UI operations.
    """

    # ---------------------------------------------------------
    # Persistent authentication
    # ---------------------------------------------------------

    user = safe_get_or_create_user(
        name=name,
        phone=phone,
    )

    if user is not None:

        session = safe_create_session(
            user_id=user["user_id"],
        )

        if session is not None:
            return {
                "authenticated": True,
                "user_id": user["user_id"],
                "user_name": user["name"],
                "session_id": session["session_id"],
                "persistence_available": True,
                "persistence_mode": "supabase",
            }

        logger.warning(
            "User persisted successfully, but session creation failed."
        )

    # ---------------------------------------------------------
    # Degraded/local mode
    # ---------------------------------------------------------

    logger.warning(
        "Persistent authentication unavailable. "
        "Creating temporary local session."
    )

    local_user_id = (
        f"LOCAL_USR_{uuid.uuid4().hex[:8].upper()}"
    )

    local_session_id = (
        f"LOCAL_SES_{uuid.uuid4().hex[:8].upper()}"
    )

    return {
        "authenticated": True,
        "user_id": local_user_id,
        "user_name": name,
        "session_id": local_session_id,
        "persistence_available": False,
        "persistence_mode": "local",
    }