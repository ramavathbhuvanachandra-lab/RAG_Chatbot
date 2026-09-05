from backend.database import supabase


def _get_session(session_id: str):
    """
    Resolve the application's session_id to the database session row.
    """

    response = (
        supabase
        .table("sessions")
        .select("id, session_id")
        .eq("session_id", session_id)
        .single()
        .execute()
    )

    session = response.data

    if not session:
        raise ValueError(
            f"Session not found for session_id={session_id}"
        )

    return session


def save_message(
    session_id: str,
    role: str,
    message: str,
    response_time: float = None,
):
    """
    Save a message during the database relationship transition.

    Both the legacy TEXT session_id and the new UUID session_uuid
    are populated until the legacy column is removed.
    """

    session = _get_session(session_id)

    response = (
        supabase
        .table("messages")
        .insert({
            # Legacy column — required during transition
            "session_id": session["session_id"],

            # New UUID relationship
            "session_uuid": session["id"],

            "role": role,
            "message": message,
            "response_time": response_time,
        })
        .execute()
    )

    return response.data[0]


def get_chat_history(session_id: str):
    """
    Retrieve chat history using the application session_id.

    During the transition, the legacy column is still used for reads.
    """

    response = (
        supabase
        .table("messages")
        .select("*")
        .eq("session_id", session_id)
        .order("timestamp")
        .execute()
    )

    return response.data