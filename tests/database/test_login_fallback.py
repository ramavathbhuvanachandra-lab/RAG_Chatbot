from backend import authentication


def test_login_uses_supabase_when_available(monkeypatch):
    """
    When persistence is available, authentication should use
    the real Supabase user and session.
    """

    expected_user = {
        "user_id": "USR_TEST1234",
        "name": "Test User",
        "phone": "9999999999",
    }

    expected_session = {
        "session_id": "SES_TEST1234",
    }

    monkeypatch.setattr(
        authentication,
        "safe_get_or_create_user",
        lambda name, phone: expected_user,
    )

    monkeypatch.setattr(
        authentication,
        "safe_create_session",
        lambda user_id: expected_session,
    )

    result = authentication.authenticate_user(
        name="Test User",
        phone="9999999999",
    )

    assert result["authenticated"] is True
    assert result["user_id"] == "USR_TEST1234"
    assert result["user_name"] == "Test User"
    assert result["session_id"] == "SES_TEST1234"
    assert result["persistence_available"] is True
    assert result["persistence_mode"] == "supabase"


def test_login_falls_back_when_supabase_is_unavailable(monkeypatch):
    """
    When Supabase is unavailable, authentication must still
    return a usable local session.
    """

    monkeypatch.setattr(
        authentication,
        "safe_get_or_create_user",
        lambda name, phone: None,
    )

    result = authentication.authenticate_user(
        name="Test User",
        phone="9999999999",
    )

    assert result["authenticated"] is True
    assert result["user_id"].startswith("LOCAL_USR_")
    assert result["session_id"].startswith("LOCAL_SES_")
    assert result["user_name"] == "Test User"
    assert result["persistence_available"] is False
    assert result["persistence_mode"] == "local"


def test_login_falls_back_when_session_creation_fails(monkeypatch):
    """
    If the user exists but session persistence fails,
    the chatbot should still receive a local session.
    """

    expected_user = {
        "user_id": "USR_TEST1234",
        "name": "Test User",
        "phone": "9999999999",
    }

    monkeypatch.setattr(
        authentication,
        "safe_get_or_create_user",
        lambda name, phone: expected_user,
    )

    monkeypatch.setattr(
        authentication,
        "safe_create_session",
        lambda user_id: None,
    )

    result = authentication.authenticate_user(
        name="Test User",
        phone="9999999999",
    )

    assert result["authenticated"] is True
    assert result["user_name"] == "Test User"
    assert result["user_id"].startswith("LOCAL_USR_")
    assert result["session_id"].startswith("LOCAL_SES_")
    assert result["persistence_available"] is False
    assert result["persistence_mode"] == "local"