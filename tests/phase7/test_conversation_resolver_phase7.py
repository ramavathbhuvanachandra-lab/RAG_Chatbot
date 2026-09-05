"""
Phase 7A — Conversation Resolver Regression Tests

Focus
-----
Follow-up resolution, topic continuity, ambiguity safety, and resolver
failure fallback.

These tests do not require Supabase.
"""

from backend import conversation_resolver


def test_no_history_is_standalone():

    result = conversation_resolver.resolve_conversation(
        question="What are the hostel fees?",
        chat_history=[],
    )

    assert result["mode"] == "standalone"
    assert (
        result["resolved_question"]
        == "What are the hostel fees?"
    )


def test_standalone_question_does_not_call_llm(
    monkeypatch,
):

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
        question="What are the hostel fees?",
        chat_history=[
            {
                "role": "user",
                "content": "Tell me about B.Tech admission.",
            },
            {
                "role": "assistant",
                "content": "Admission details.",
            },
        ],
    )

    assert result["mode"] == "topic_switch"


def test_follow_up_is_resolved(
    monkeypatch,
):

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

    result = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=[
            {
                "role": "user",
                "content": "What is the B.Tech admission process?",
            },
            {
                "role": "assistant",
                "content": "The admission process is ...",
            },
        ],
    )

    assert result["mode"] == "follow_up"
    assert (
        result["resolved_question"]
        == "What are the B.Tech admission fees?"
    )
    assert result["active_topic"] == "admission"
    assert result["active_entity"] == "B.Tech"


def test_topic_switch_stays_new_topic(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Explicit topic switch should not require resolver LLM."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question="What are the hostel fees?",
        chat_history=[
            {
                "role": "user",
                "content": "Tell me about B.Tech admission.",
            },
            {
                "role": "assistant",
                "content": "Admission details.",
            },
        ],
    )

    assert result["mode"] == "topic_switch"
    assert (
        result["resolved_question"]
        == "What are the hostel fees?"
    )


def test_obviously_ambiguous_reference_is_not_guessed(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Obvious ambiguity should not require an LLM call."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question="What about that?",
        chat_history=[
            {
                "role": "user",
                "content": "Tell me about hostels.",
            },
            {
                "role": "assistant",
                "content": "Hostels provide ...",
            },
        ],
    )

    assert result["mode"] == "ambiguous"
    assert (
        result["resolved_question"]
        == "What about that?"
    )


def test_resolver_failure_falls_back_safely(
    monkeypatch,
):

    def failing_model(*args, **kwargs):
        raise RuntimeError(
            "simulated resolver failure"
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        failing_model,
    )

    result = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=[
            {
                "role": "user",
                "content": "Tell me about B.Tech admission.",
            },
            {
                "role": "assistant",
                "content": "Admission details.",
            },
        ],
    )

    assert result["mode"] == "ambiguous"
    assert (
        result["resolved_question"]
        == "What about fees?"
    )


def test_json_schema_output_is_parsed(
    monkeypatch,
):

    class FakeResponse:
        content = """
        {
          "mode": "follow_up",
          "resolved_question": "What are the hostel Wi-Fi facilities?",
          "active_topic": "hostel",
          "active_entity": "hostel"
        }
        """

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        lambda prompt: FakeResponse(),
    )

    result = conversation_resolver.resolve_conversation(
        question="What about Wi-Fi?",
        chat_history=[
            {
                "role": "user",
                "content": "Tell me about hostel facilities.",
            },
            {
                "role": "assistant",
                "content": "The hostel has several facilities.",
            },
        ],
    )

    assert result["mode"] == "follow_up"
    assert "hostel" in (
        result["resolved_question"].lower()
    )
    assert "wi-fi" in (
        result["resolved_question"].lower()
    )


def test_history_window_is_bounded():

    history = []

    for index in range(20):

        history.append(
            {
                "role": "user",
                "content": f"Question {index}",
            }
        )

    history_text = (
        conversation_resolver._build_history(
            history
        )
    )

    assert "Question 0" not in history_text
    assert "Question 11" not in history_text
    assert "Question 12" in history_text
    assert "Question 19" in history_text


def test_parent_route_context_is_preserved(
    monkeypatch,
):

    class FakeResponse:
        content = """
        {
          "mode": "follow_up",
          "resolved_question": "What percentage is required for the bachelor's-degree route for regular Ph.D. admission?",
          "active_topic": "admission",
          "active_entity": "Ph.D."
        }
        """

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        lambda prompt: FakeResponse(),
    )

    result = conversation_resolver.resolve_conversation(
        question="What percentage is needed for that route?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "What are the eligibility requirements "
                    "for regular Ph.D. admission?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "There are master's and four-year "
                    "bachelor's routes."
                ),
            },
        ],
    )

    assert result["mode"] == "follow_up"
    assert "bachelor" in (
        result["resolved_question"].lower()
    )
    assert "ph.d." in (
        result["resolved_question"].lower()
    )


def test_model_cannot_turn_explicit_new_topic_into_old_topic(
    monkeypatch,
):

    # Even if the model returns a follow-up classification, the current
    # deterministic classifier should never call it for an explicitly
    # self-contained question.
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Explicit new topic should remain deterministic."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question="What are the M.Sc. admission requirements?",
        chat_history=[
            {
                "role": "user",
                "content": "Tell me about hostel fees.",
            },
            {
                "role": "assistant",
                "content": "Hostel fee details.",
            },
        ],
    )

    assert result["mode"] == "topic_switch"
    assert (
        result["resolved_question"]
        == "What are the M.Sc. admission requirements?"
    )