"""
Supabase connection smoke tests.

Purpose:
    Verify that the application can initialize the configured
    Supabase client and perform a lightweight read operation.

These tests are intentionally small. They do not test the
chatbot or application logic.
"""

from backend.database import supabase


def test_supabase_client_initializes():
    """Supabase client should initialize successfully."""
    assert supabase is not None


def test_supabase_can_read_users():
    """Supabase should be able to perform a lightweight users read."""
    response = (
        supabase
        .table("users")
        .select("user_id")
        .limit(1)
        .execute()
    )

    assert response is not None
    assert hasattr(response, "data")