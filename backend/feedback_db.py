from backend.database import supabase


def _get_session_uuid(session_id: str) -> str:
    """
    Resolve application session_id to internal sessions.id UUID.
    """

    response = (
        supabase
        .table("sessions")
        .select("id")
        .eq("session_id", session_id)
        .single()
        .execute()
    )

    session = response.data

    if not session:
        raise ValueError(
            f"Session not found for session_id={session_id}"
        )

    return session["id"]


def save_feedback(
    message_id: str,
    session_id: str,
    feedback: str,
):
    """
    Save feedback using UUID foreign keys.
    """

    session_uuid = _get_session_uuid(session_id)

    response = (
        supabase
        .table("message_feedback")
        .insert({
            "message_id": message_id,
            "session_id": session_uuid,
            "feedback": feedback,
        })
        .execute()
    )

    return response.data[0]