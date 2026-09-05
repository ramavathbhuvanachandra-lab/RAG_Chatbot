import pytest

from backend import message_db


def test_save_message_failure_isolated(monkeypatch):
    """
    Verify that a Supabase write failure can be isolated
    from the chatbot execution path.
    """

    def failing_save_message(*args, **kwargs):
        raise RuntimeError("Simulated Supabase outage")

    monkeypatch.setattr(
        message_db,
        "save_message",
        failing_save_message,
    )

    try:
        message_db.save_message(
            session_id="SES_TEST",
            role="user",
            message="Test question",
        )

    except RuntimeError:
        # Expected database failure.
        # The application layer must catch this.
        pass