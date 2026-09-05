from backend import persistence


def test_message_persistence_failure_is_safe(monkeypatch):
    """
    Supabase failure must not propagate through the
    persistence boundary.
    """

    def failing_save_message(*args, **kwargs):
        raise RuntimeError("Simulated Supabase outage")

    monkeypatch.setattr(
        persistence,
        "save_message",
        failing_save_message,
    )

    result = persistence.safe_save_message(
        session_id="SES_TEST",
        role="user",
        message="Test question",
    )

    assert result is None


def test_feedback_persistence_failure_is_safe(monkeypatch):
    """
    Feedback failure must not break the application.
    """

    def failing_save_feedback(*args, **kwargs):
        raise RuntimeError("Simulated Supabase outage")

    monkeypatch.setattr(
        persistence,
        "save_feedback",
        failing_save_feedback,
    )

    result = persistence.safe_save_feedback(
        message_id="MSG_TEST",
        session_id="SES_TEST",
        feedback="up",
    )

    assert result is None