"""
Phase 7D — Conversation Resolver Hardening

Purpose
-------
Verify that natural contextual questions reach the resolver while
self-contained questions remain cheap and explicit ambiguity remains safe.
"""

from backend import conversation_resolver


# =========================================================
# Helper
# =========================================================

def fake_response(
    mode,
    resolved_question,
    active_topic="",
    active_entity="",
):

    class FakeResponse:

        content = f"""
        {{
          "mode": "{mode}",
          "resolved_question": "{resolved_question}",
          "active_topic": "{active_topic}",
          "active_entity": "{active_entity}"
        }}
        """

    return FakeResponse()


# =========================================================
# Generic contextual question shapes
# =========================================================

def test_percentage_question_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "What percentage is needed?"
    )


def test_entrance_exam_question_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "What entrance exam is required?"
    )


def test_control_question_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "Which one is related to control?"
    )


def test_duration_question_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "How long is it?"
    )


def test_natural_hostel_continuation_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "How about a stay longer than ten days?"
    )


# =========================================================
# Real resolver routing
# =========================================================

def test_percentage_question_reaches_resolver(
    monkeypatch,
):

    calls = []

    def fake_invoke(prompt):

        calls.append(prompt)

        return fake_response(
            mode="follow_up",
            resolved_question=(
                "What percentage is required for regular Ph.D. admission?"
            ),
            active_topic="admission",
            active_entity="Ph.D.",
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fake_invoke,
    )

    result = conversation_resolver.resolve_conversation(
        question="What percentage is needed?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "What are the regular Ph.D. eligibility requirements?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "There are multiple eligibility routes."
                ),
            },
        ],
    )

    assert calls
    assert result["mode"] == "follow_up"
    assert result["resolver_called"] is True
    assert (
        result["diagnostic_reason"]
        == "resolver_success"
    )


def test_entrance_exam_question_reaches_resolver(
    monkeypatch,
):

    calls = []

    def fake_invoke(prompt):

        calls.append(prompt)

        return fake_response(
            mode="follow_up",
            resolved_question=(
                "What entrance exam is required for M.Sc. admission?"
            ),
            active_topic="admission",
            active_entity="M.Sc.",
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fake_invoke,
    )

    result = conversation_resolver.resolve_conversation(
        question="What entrance exam is required?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "Now tell me about M.Sc. programs."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Several M.Sc. programs are available."
                ),
            },
        ],
    )

    assert calls
    assert result["mode"] == "follow_up"
    assert "m.sc." in (
        result["resolved_question"].lower()
    )


def test_stay_duration_question_reaches_resolver(
    monkeypatch,
):

    calls = []

    def fake_invoke(prompt):

        calls.append(prompt)

        return fake_response(
            mode="follow_up",
            resolved_question=(
                "What hostel pricing category applies "
                "to a stay longer than ten days?"
            ),
            active_topic="hostel",
            active_entity="hostel",
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fake_invoke,
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "How about a stay longer than ten days?"
        ),
        chat_history=[
            {
                "role": "user",
                "content": (
                    "What are the short-term hostel rates?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Short-term rates apply to shorter stays."
                ),
            },
        ],
    )

    assert calls
    assert result["mode"] == "follow_up"
    assert "hostel" in (
        result["resolved_question"].lower()
    )


# =========================================================
# Self-contained questions remain cheap
# =========================================================

def test_explicit_new_subject_does_not_call_resolver(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):

        raise AssertionError(
            "Resolver should not be called."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "What are the hostel fees?"
        ),
        chat_history=[
            {
                "role": "user",
                "content": (
                    "Tell me about Ph.D. admissions."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Ph.D. admission information."
                ),
            },
        ],
    )

    assert result["mode"] == "topic_switch"
    assert result["resolver_called"] is False
    assert (
        result["diagnostic_reason"]
        == "clearly_self_contained"
    )


# =========================================================
# Ambiguous references remain safe
# =========================================================

def test_ambiguous_reference_does_not_call_resolver(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):

        raise AssertionError(
            "Ambiguous reference should not be guessed."
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
                "content": (
                    "We discussed Ph.D. admission and hostel fees."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Both subjects were discussed."
                ),
            },
        ],
    )

    assert result["mode"] == "ambiguous"
    assert result["resolver_called"] is False
    assert (
        result["diagnostic_reason"]
        == "obviously_ambiguous"
    )


# =========================================================
# Resolver failure remains safe
# =========================================================

def test_resolver_failure_is_diagnostic(
    monkeypatch,
):

    def fail(*args, **kwargs):

        raise RuntimeError(
            "simulated resolver outage"
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail,
    )

    result = conversation_resolver.resolve_conversation(
        question="What percentage is needed?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "What are the Ph.D. requirements?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Eligibility information."
                ),
            },
        ],
    )

    assert result["mode"] == "ambiguous"
    assert result["resolver_called"] is True
    assert (
        result["diagnostic_reason"]
        == "resolver_failure"
    )
    assert (
        result["resolved_question"]
        == "What percentage is needed?"
    )


# =========================================================
# Invalid model output is diagnostic
# =========================================================

def test_invalid_model_output_is_diagnostic(
    monkeypatch,
):

    class BadResponse:

        content = "not json"

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        lambda prompt: BadResponse(),
    )

    result = conversation_resolver.resolve_conversation(
        question="What percentage is needed?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "What are the regular Ph.D. requirements?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "There are several routes."
                ),
            },
        ],
    )

    assert result["mode"] == "ambiguous"
    assert result["resolver_called"] is True
    assert (
        result["diagnostic_reason"]
        == "invalid_model_output"
    )


# =========================================================
# History remains bounded
# =========================================================

def test_resolver_history_window_remains_bounded():

    history = []

    for index in range(20):

        history.append(
            {
                "role": "user",
                "content": (
                    f"Question {index}"
                ),
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": (
                    f"Answer {index}"
                ),
            }
        )

    rendered = (
        conversation_resolver._build_history(
            history
        )
    )

    assert "Question 0" not in rendered
    assert "Answer 0" not in rendered

    assert "Question 19" in rendered
    assert "Answer 19" in rendered