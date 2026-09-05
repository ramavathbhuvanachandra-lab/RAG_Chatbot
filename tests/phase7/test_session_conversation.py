"""
Phase 7A — Session-scoped conversation memory tests.

These tests verify that conversational history is isolated by session.
"""

from backend.conversation_session import (
    recent_chat_history,
)


def test_recent_history_keeps_only_latest_messages():

    history = [
        {
            "role": "user",
            "content": "Q1",
        },
        {
            "role": "assistant",
            "content": "A1",
        },
        {
            "role": "user",
            "content": "Q2",
        },
        {
            "role": "assistant",
            "content": "A2",
        },
    ]

    result = recent_chat_history(
        history,
        max_messages=2,
    )

    assert result == [
        {
            "role": "user",
            "content": "Q2",
        },
        {
            "role": "assistant",
            "content": "A2",
        },
    ]


def test_empty_history_is_safe():

    assert recent_chat_history(
        [],
        max_messages=8,
    ) == []


def test_zero_history_limit_is_safe():

    history = [
        {
            "role": "user",
            "content": "Q1",
        }
    ]

    assert recent_chat_history(
        history,
        max_messages=0,
    ) == []