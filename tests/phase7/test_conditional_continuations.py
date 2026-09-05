"""
Phase 7E — Conditional / Alternative Conversation Continuations

Purpose
-------
Verify that conditional and alternative conversational forms are handled
generically without adding institution-specific rules.
"""

from backend import conversation_resolver


# =========================================================
# Helpers
# =========================================================

def fake_response(
    mode,
    resolved_question,
    active_topic="",
    active_entity="",
):

    class FakeResponse:

        content = (
            "{"
            f'"mode": "{mode}",'
            f'"resolved_question": "{resolved_question}",'
            f'"active_topic": "{active_topic}",'
            f'"active_entity": "{active_entity}"'
            "}"
        )

    return FakeResponse()


# =========================================================
# Candidate detection
# =========================================================

def test_what_if_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "What if I choose the other route?"
    )


def test_what_happens_if_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "What happens if I switch?"
    )


def test_instead_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "Instead, what about the other option?"
    )


def test_alternative_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "Would the alternative option work?"
    )


def test_another_route_is_context_candidate():

    assert conversation_resolver._is_follow_up_like(
        "Can I take another route?"
    )


# =========================================================
# Resolver integration
# =========================================================

def test_conditional_phd_follow_up_reaches_resolver(
    monkeypatch,
):

    calls = []

    def fake_invoke(prompt):

        calls.append(prompt)

        return fake_response(
            mode="follow_up",
            resolved_question=(
                "What are the eligibility requirements "
                "for the part-time Ph.D. route?"
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
        question=(
            "What if I choose the part-time route instead?"
        ),
        chat_history=[
            {
                "role": "user",
                "content": (
                    "What are the regular Ph.D. "
                    "eligibility requirements?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The institute publishes multiple "
                    "admission routes."
                ),
            },
        ],
    )

    assert calls
    assert result["mode"] == "follow_up"
    assert result["resolver_called"] is True
    assert (
        "part-time"
        in result["resolved_question"].lower()
    )


def test_conditional_hostel_follow_up_reaches_resolver(
    monkeypatch,
):

    calls = []

    def fake_invoke(prompt):

        calls.append(prompt)

        return fake_response(
            mode="follow_up",
            resolved_question=(
                "What hostel accommodation applies "
                "to a stay of several weeks?"
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
            "What if I stay for several weeks instead?"
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
    assert result["resolver_called"] is True
    assert "hostel" in (
        result["resolved_question"].lower()
    )


def test_conditional_research_follow_up_reaches_resolver(
    monkeypatch,
):

    calls = []

    def fake_invoke(prompt):

        calls.append(prompt)

        return fake_response(
            mode="follow_up",
            resolved_question=(
                "Which Electrical Engineering "
                "research area would fit a focus on control?"
            ),
            active_topic="research",
            active_entity="Electrical Engineering",
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fake_invoke,
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "What if I want to focus on control instead?"
        ),
        chat_history=[
            {
                "role": "user",
                "content": (
                    "What research areas are available "
                    "in Electrical Engineering?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Several research areas are available."
                ),
            },
        ],
    )

    assert calls
    assert result["mode"] == "follow_up"
    assert result["resolver_called"] is True
    assert "control" in (
        result["resolved_question"].lower()
    )


# =========================================================
# Explicit self-contained alternatives
# =========================================================

def test_explicit_new_subject_alternative_is_not_forced_into_history(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):

        raise AssertionError(
            "Self-contained question must not invoke resolver."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "What if I apply for M.Sc. admission instead?"
        ),
        chat_history=[
            {
                "role": "user",
                "content": (
                    "Tell me about regular Ph.D. admission."
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
        result["resolved_question"]
        == "What if I apply for M.Sc. admission instead?"
    )


# =========================================================
# Vague alternative reference
# =========================================================

def test_vague_alternative_reference_is_safe():

    result = conversation_resolver.resolve_conversation(
        question="Can I take that instead?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "We discussed two different admission routes."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Both routes were described."
                ),
            },
        ],
    )

    assert result["mode"] in {
        "ambiguous",
        "topic_switch",
    }

    assert result[
        "resolved_question"
    ] == "Can I take that instead?"