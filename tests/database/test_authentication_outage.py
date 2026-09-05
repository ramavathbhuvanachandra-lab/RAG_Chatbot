from backend import authentication


def test_complete_supabase_outage(monkeypatch):
    """
    Simulate a complete Supabase outage.

    The chatbot login layer must still create a usable
    temporary identity and session.
    """

    def fail_user(*args, **kwargs):
        return None

    monkeypatch.setattr(
        authentication,
        "safe_get_or_create_user",
        fail_user,
    )

    result = authentication.authenticate_user(
        name="Bhuvan",
        phone="9701755232",
    )

    assert result["authenticated"] is True

    assert result["persistence_available"] is False
    assert result["persistence_mode"] == "local"

    assert result["user_id"].startswith("LOCAL_USR_")
    assert result["session_id"].startswith("LOCAL_SES_")

    assert result["user_name"] == "Bhuvan"