"""
Phase 7B — Multi-Turn Conversational Continuity

Purpose
-------
Stress the conversation resolver across realistic multi-turn
conversations.

Focus
-----
1. Topic continuity.
2. Follow-up chains.
3. Narrowing from parent topic to subtopic.
4. Explicit topic switches.
5. Returning to a previous topic.
6. Long conversations.
7. Preservation of program/organization boundaries.
8. Ambiguity safety.
9. Session history isolation at the resolver boundary.

Important
---------
These tests focus on conversation understanding.

They do not require:
    - Supabase
    - retrieval
    - answer generation
    - the production LLM

The resolver model is mocked so the tests remain deterministic.
"""

from __future__ import annotations

from backend import conversation_resolver


# =========================================================
# Helpers
# =========================================================

def fake_response(
    *,
    mode: str,
    resolved_question: str,
    active_topic: str = "",
    active_entity: str = "",
):
    """
    Build a fake resolver response in the exact JSON shape expected
    by the production resolver.
    """

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


def install_resolver_sequence(
    monkeypatch,
    responses,
):
    """
    Install deterministic LLM responses consumed sequentially.
    """

    queue = list(
        responses
    )

    calls = []

    def fake_invoke(prompt):

        calls.append(
            prompt
        )

        if not queue:
            raise AssertionError(
                "Resolver model called more times than expected."
            )

        return queue.pop(
            0
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fake_invoke,
    )

    return calls


# =========================================================
# 7B.1 — Three-turn continuity
# =========================================================

def test_three_turn_conversation_preserves_parent_topic(
    monkeypatch,
):

    history = []

    # -----------------------------------------------------
    # Turn 1
    # -----------------------------------------------------

    turn1 = conversation_resolver.resolve_conversation(
        question=(
            "What are the eligibility requirements "
            "for regular Ph.D. admission?"
        ),
        chat_history=history,
    )

    assert turn1["mode"] == "standalone"

    history.extend(
        [
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
                    "The requirements include multiple "
                    "qualifying routes."
                ),
            },
        ]
    )

    # -----------------------------------------------------
    # Turn 2
    # -----------------------------------------------------

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What is the four-year bachelor's-degree "
                    "route for regular Ph.D. admission?"
                ),
                active_topic="admission",
                active_entity="Ph.D.",
            )
        ],
    )

    turn2 = conversation_resolver.resolve_conversation(
        question=(
            "What about the bachelor's route?"
        ),
        chat_history=history,
    )

    assert turn2["mode"] == "follow_up"
    assert "bachelor" in (
        turn2["resolved_question"].lower()
    )
    assert "ph.d." in (
        turn2["resolved_question"].lower()
    )

    history.extend(
        [
            {
                "role": "user",
                "content": (
                    "What about the bachelor's route?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The bachelor's route is a separate "
                    "eligibility pathway."
                ),
            },
        ]
    )

    # -----------------------------------------------------
    # Turn 3
    # -----------------------------------------------------

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What percentage is required for the "
                    "four-year bachelor's route for regular Ph.D. admission?"
                ),
                active_topic="admission",
                active_entity="Ph.D.",
            )
        ],
    )

    turn3 = conversation_resolver.resolve_conversation(
        question=(
            "What percentage is needed?"
        ),
        chat_history=history,
    )

    assert turn3["mode"] == "follow_up"
    assert "percentage" in (
        turn3["resolved_question"].lower()
    )
    assert "bachelor" in (
        turn3["resolved_question"].lower()
    )
    assert "ph.d." in (
        turn3["resolved_question"].lower()
    )


# =========================================================
# 7B.2 — Five-turn chain
# =========================================================

def test_five_turn_conversation_keeps_current_subject(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "Tell me about regular Ph.D. admission."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Regular Ph.D. admission has several "
                "eligibility routes."
            ),
        },
        {
            "role": "user",
            "content": (
                "What about the bachelor's route?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The four-year bachelor's route is one "
                "of the eligibility pathways."
            ),
        },
    ]

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What percentage is required for the "
                    "four-year bachelor's route for regular Ph.D. admission?"
                ),
                active_topic="admission",
                active_entity="Ph.D.",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What percentage is required?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"
    assert "bachelor" in (
        result["resolved_question"].lower()
    )
    assert "ph.d." in (
        result["resolved_question"].lower()
    )

    history.extend(
        [
            {
                "role": "user",
                "content": "What percentage is required?",
            },
            {
                "role": "assistant",
                "content": "The published criteria specify a percentage threshold.",
            },
        ]
    )

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "Does the four-year bachelor's route for "
                    "regular Ph.D. admission require GATE?"
                ),
                active_topic="admission",
                active_entity="Ph.D.",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about GATE?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"
    assert "gate" in (
        result["resolved_question"].lower()
    )
    assert "ph.d." in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7B.3 — Explicit topic switch
# =========================================================

def test_explicit_topic_switch_replaces_previous_subject(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Explicit topic switch should not need resolver LLM."
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
                    "Tell me about regular Ph.D. admission."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Ph.D. admission details."
                ),
            },
        ],
    )

    assert result["mode"] == "topic_switch"
    assert (
        result["resolved_question"]
        == "What are the hostel fees?"
    )


# =========================================================
# 7B.4 — Topic switch followed by continuity
# =========================================================

def test_new_topic_then_follow_up_uses_new_topic(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "Tell me about regular Ph.D. admission."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Ph.D. admission details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Now tell me about hostel accommodation."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hostel accommodation details."
            ),
        },
    ]

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What are the hostel fees?"
                ),
                active_topic="hostel",
                active_entity="hostel",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"
    assert "hostel" in (
        result["resolved_question"].lower()
    )
    assert "fee" in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7B.5 — Return to previous topic explicitly
# =========================================================

def test_explicit_return_to_old_topic_is_new_topic(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "Tell me about Ph.D. admission."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Ph.D. admission details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Tell me about hostel fees."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hostel fee details."
            ),
        },
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Explicit question should not require resolver LLM."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "What are the Ph.D. admission requirements?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"
    assert "ph.d." in (
        result["resolved_question"].lower()
    )
    assert "admission" in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7B.6 — Pronoun chain
# =========================================================

def test_pronoun_chain_preserves_subject(
    monkeypatch,
):

    history = [
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
                "The department has several research areas."
            ),
        },
    ]

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "Which Electrical Engineering research area "
                    "includes control systems?"
                ),
                active_topic="research",
                active_entity="Electrical Engineering",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "Which one is related to control?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "follow_up"
    assert "electrical engineering" in (
        result["resolved_question"].lower()
    )
    assert "control" in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7B.7 — Parent program boundary
# =========================================================

def test_program_boundary_is_preserved(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "What are the M.Tech admission requirements?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The M.Tech admission requirements include "
                "academic and examination criteria."
            ),
        },
    ]

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What GATE requirement applies to "
                    "M.Tech admission?"
                ),
                active_topic="admission",
                active_entity="M.Tech",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about GATE?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"
    assert "m.tech" in (
        result["resolved_question"].lower()
    )
    assert "gate" in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7B.8 — Organizational boundary
# =========================================================

def test_school_boundary_is_preserved(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "What programs are available in the "
                "School of Artificial Intelligence and Data Science?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The school offers several technology-focused programs."
            ),
        },
    ]

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What research is being done in the "
                    "School of Artificial Intelligence and Data Science?"
                ),
                active_topic="research",
                active_entity=(
                    "School of Artificial Intelligence and Data Science"
                ),
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about research?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"
    assert (
        "school of artificial intelligence and data science"
        in result["resolved_question"].lower()
    )


# =========================================================
# 7B.9 — Eight-turn history stays bounded
# =========================================================

def test_long_history_uses_only_recent_window(
    monkeypatch,
):

    history = []

    for index in range(20):

        history.append(
            {
                "role": "user",
                "content": f"Old question {index}",
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": f"Old answer {index}",
            }
        )

    captured = install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What are the latest hostel fees "
                    "in the current conversation topic?"
                ),
                active_topic="hostel",
                active_entity="hostel",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"

    assert captured

    prompt = captured[0]

    # The resolver uses only the latest 8 messages.
    assert "Old question 0" not in prompt
    assert "Old answer 0" not in prompt

    assert "Old question 19" in prompt
    assert "Old answer 19" in prompt


# =========================================================
# 7B.10 — Genuine ambiguity remains ambiguous
# =========================================================

def test_multiple_possible_references_remain_safe(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Purely ambiguous reference should be rejected deterministically."
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
                    "We discussed Ph.D. admission and hostel facilities."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Both topics have different rules."
                ),
            },
        ],
    )

    assert result["mode"] == "ambiguous"
    assert (
        result["resolved_question"]
        == "What about that?"
    )


# =========================================================
# 7B.11 — Resolver failure does not break conversation
# =========================================================

def test_resolver_failure_returns_safe_fallback(
    monkeypatch,
):

    def failing_invoke(*args, **kwargs):
        raise RuntimeError(
            "simulated LLM outage"
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        failing_invoke,
    )

    result = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=[
            {
                "role": "user",
                "content": (
                    "Tell me about M.Tech admission."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "M.Tech admission details."
                ),
            },
        ],
    )

    assert result["mode"] == "ambiguous"
    assert (
        result["resolved_question"]
        == "What about fees?"
    )


# =========================================================
# 7B.12 — Two independent session histories
# =========================================================

def test_two_sessions_produce_independent_contexts(
    monkeypatch,
):

    session_a_history = [
        {
            "role": "user",
            "content": (
                "Tell me about B.Tech admission."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "B.Tech admission details."
            ),
        },
    ]

    session_b_history = [
        {
            "role": "user",
            "content": (
                "Tell me about hostel accommodation."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hostel accommodation details."
            ),
        },
    ]

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What are the B.Tech admission fees?"
                ),
                active_topic="admission",
                active_entity="B.Tech",
            ),
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What are the hostel fees?"
                ),
                active_topic="hostel",
                active_entity="hostel",
            ),
        ],
    )

    result_a = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=session_a_history,
    )

    result_b = conversation_resolver.resolve_conversation(
        question="What about fees?",
        chat_history=session_b_history,
    )

    assert "b.tech" in (
        result_a["resolved_question"].lower()
    )

    assert "hostel" in (
        result_b["resolved_question"].lower()
    )

    assert "hostel" not in (
        result_a["resolved_question"].lower()
    )

    assert "b.tech" not in (
        result_b["resolved_question"].lower()
    )


# =========================================================
# 7B.13 — Multi-step narrowing
# =========================================================

def test_multi_step_narrowing_keeps_full_parent_chain(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "What are the Ph.D. eligibility requirements?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The institute publishes multiple routes."
            ),
        },
        {
            "role": "user",
            "content": (
                "What about the four-year bachelor's route?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The four-year bachelor's route is one pathway."
            ),
        },
        {
            "role": "user",
            "content": (
                "What percentage is needed?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The published material gives percentage thresholds."
            ),
        },
    ]

    install_resolver_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "Is GATE required for the four-year bachelor's "
                    "route for regular Ph.D. admission?"
                ),
                active_topic="admission",
                active_entity="Ph.D.",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about GATE?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"

    resolved = (
        result["resolved_question"].lower()
    )

    assert "gate" in resolved
    assert "bachelor" in resolved
    assert "ph.d." in resolved


# =========================================================
# 7B.14 — Standalone explicit question after long history
# =========================================================

def test_explicit_question_after_long_history_is_not_forced_into_old_topic(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": "Tell me about hostel fees.",
        },
        {
            "role": "assistant",
            "content": "Hostel fee details.",
        },
        {
            "role": "user",
            "content": "What about single occupancy?",
        },
        {
            "role": "assistant",
            "content": "Single occupancy details.",
        },
        {
            "role": "user",
            "content": "Now tell me about M.Sc. admissions.",
        },
        {
            "role": "assistant",
            "content": "M.Sc. admission details.",
        },
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Self-contained topic switch must not invoke resolver."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "What entrance exam is required for M.Sc. admission?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert (
        result["resolved_question"]
        == "What entrance exam is required for M.Sc. admission?"
    )