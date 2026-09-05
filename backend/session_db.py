import uuid

from backend.database import supabase


def create_session(user_id: str):
    """
    Create a new session during the database transition.

    The application identifier (`user_id`) is retained in the legacy
    column for compatibility, while the new UUID relationship is also
    populated.

    Once the migration is finalized, the legacy column will be removed.
    """

    # ---------------------------------------------------------
    # Resolve application user_id -> internal users.id UUID
    # ---------------------------------------------------------

    user_response = (
        supabase
        .table("users")
        .select("id, user_id")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    user = user_response.data

    if not user:
        raise ValueError(
            f"User not found for user_id={user_id}"
        )

    # ---------------------------------------------------------
    # Generate application session identifier
    # ---------------------------------------------------------

    session_id = f"SES_{uuid.uuid4().hex[:8].upper()}"

    # ---------------------------------------------------------
    # Write BOTH old and new relationship columns
    # ---------------------------------------------------------

    response = (
        supabase
        .table("sessions")
        .insert({
            # Legacy column — required during transition
            "user_id": user["user_id"],

            # New UUID relationship
            "user_uuid": user["id"],

            # Application-level session identifier
            "session_id": session_id,
        })
        .execute()
    )

    return response.data[0]