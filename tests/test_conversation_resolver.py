"""
Tests for the IIT Jodhpur V1 conversation resolver.

Coverage:
- Standalone questions bypass the resolver model.
- Follow-up questions can be resolved from recent context.
- Explicit topic switches pass through unchanged.
- Resolver failures fall back safely.
- Obviously ambiguous references are never guessed.
"""

from backend import conversation_resolver


# =========================================================
# Standalone
# =========================================================

def test_standalone_question_is_not_rewritten(monkeypatch):
    """
    Standalone questions must not call the resolver LLM.
    """

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Resolver model should not be called."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question="What is the B.Tech admission process?",
        chat_history=[],
    )

    assert result["mode"] == "standalone"

    assert result["resolved_question"] == (
        "What is the B.Tech admission process?"
    )

    assert result["active_topic"] == ""
    assert result["active_entity"] == ""


# =========================================================
# Follow-up
# =========================================================

def test_follow_up_can_be_resolved(monkeypatch):
    """
    A genuine follow-up should be converted into a
    self-contained retrieval question.
    """

    class FakeResponse:
        content = """
{
  "mode": "follow_up",
  "resolved_question": "What are the B.Tech admission fees?",
  "active_topic": "admission",
  "active_entity": "B.Tech"
}
"""

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        lambda prompt: FakeResponse(),
    )

    history = [
        {
            "role": "user",
            "content": (
                "What is the B.Tech admission process?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The B.Tech admission process is "
                "handled through the relevant admission route."
            ),
        },
    ]

    result = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"

    assert result["resolved_question"] == (
        "What are the B.Tech admission fees?"
    )

    assert result["active_topic"] == "admission"
    assert result["active_entity"] == "B.Tech"


# =========================================================
# Topic Switch
# =========================================================

def test_topic_switch_is_not_forced_through_resolver(
    monkeypatch,
):
    """
    An explicit new topic should not invoke the resolver LLM.
    """

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Topic switch should not call resolver model."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    history = [
        {
            "role": "user",
            "content": (
                "Tell me about hostel facilities."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "IIT Jodhpur hostels provide "
                "accommodation and related facilities."
            ),
        },
    ]

    result = conversation_resolver.resolve_conversation(
        question="Now tell me about M.Sc. admission.",
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert result["resolved_question"] == (
        "Now tell me about M.Sc. admission."
    )

    assert result["active_topic"] == ""
    assert result["active_entity"] == ""


# =========================================================
# Resolver Failure
# =========================================================

def test_resolver_failure_falls_back_safely(
    monkeypatch,
):
    """
    If the resolver model fails, the original question must
    be preserved rather than inventing context.
    """

    def failing_model(*args, **kwargs):
        raise RuntimeError(
            "Simulated resolver failure"
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        failing_model,
    )

    history = [
        {
            "role": "user",
            "content": "Tell me about hostels.",
        },
        {
            "role": "assistant",
            "content": (
                "Hostels provide accommodation and facilities."
            ),
        },
    ]

    result = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=history,
    )

    assert result["mode"] == "ambiguous"

    assert result["resolved_question"] == (
        "What about fees?"
    )

    assert result["active_topic"] == ""
    assert result["active_entity"] == ""


# =========================================================
# Ambiguous "that"
# =========================================================

def test_obviously_ambiguous_reference_is_not_guessed():
    """
    Do not invent what 'that' refers to.
    """

    result = conversation_resolver.resolve_conversation(
        question="What about that?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "Tell me about hostel facilities."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "IIT Jodhpur hostels provide "
                    "accommodation and related facilities."
                ),
            },
        ],
    )

    assert result["mode"] == "ambiguous"

    assert result["resolved_question"] == (
        "What about that?"
    )

    assert result["active_topic"] == ""
    assert result["active_entity"] == ""


# =========================================================
# Ambiguous "it"
# =========================================================

def test_obviously_ambiguous_it_reference_is_not_guessed():
    """
    Do not invent what 'it' refers to.
    """

    result = conversation_resolver.resolve_conversation(
        question="What about it?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "Tell me about hostel facilities."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "IIT Jodhpur hostels provide "
                    "accommodation and related facilities."
                ),
            },
        ],
    )

    assert result["mode"] == "ambiguous"

    assert result["resolved_question"] == (
        "What about it?"
    )

    assert result["active_topic"] == ""
    assert result["active_entity"] == ""