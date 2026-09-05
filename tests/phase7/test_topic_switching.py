"""
Phase 7C — Conversational Topic Switching

Purpose
-------
Verify that conversational context changes correctly when the user
switches topics and later returns to an earlier topic.

Important invariants
--------------------
1. Explicitly named new topics replace the previous active topic.
2. Follow-ups after a topic switch use the NEW topic.
3. Returning explicitly to an old topic creates a new current context.
4. Old context must not contaminate the new topic.
5. Ambiguous references remain ambiguous.
6. No institution-specific topic logic is embedded in these tests.
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
    Create a deterministic fake LLM response.
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


def install_sequence(
    monkeypatch,
    responses,
):
    """
    Install a deterministic sequence of resolver responses.
    """

    queue = list(
        responses
    )

    calls = []

    def fake_invoke(prompt):
        calls.append(prompt)

        if not queue:
            raise AssertionError(
                "Resolver was called more times than expected."
            )

        return queue.pop(0)

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fake_invoke,
    )

    return calls


# =========================================================
# 7C.1 — Ph.D. → Hostel → Ph.D.
# =========================================================

def test_switch_from_phd_to_hostel_and_back_to_phd(
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
                "The four-year bachelor's route is one pathway."
            ),
        },
    ]

    # -----------------------------------------------------
    # Explicit switch → Hostel
    # -----------------------------------------------------

    result = conversation_resolver.resolve_conversation(
        question=(
            "Now tell me about hostel accommodation."
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert (
        result["resolved_question"]
        == "Now tell me about hostel accommodation."
    )

    history.extend(
        [
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
    )

    # -----------------------------------------------------
    # Follow-up must use Hostel, not Ph.D.
    # -----------------------------------------------------

    install_sequence(
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
    assert "ph.d." not in (
        result["resolved_question"].lower()
    )

    # -----------------------------------------------------
    # Explicit return → Ph.D.
    # -----------------------------------------------------

    history.extend(
        [
            {
                "role": "user",
                "content": "What about fees?",
            },
            {
                "role": "assistant",
                "content": "Hostel fee details.",
            },
        ]
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "Going back to regular Ph.D. admission, "
            "what about GATE?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert "ph.d." in (
        result["resolved_question"].lower()
    )

    assert "gate" in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7C.2 — A → B → C → A
# =========================================================

def test_three_topic_switches_do_not_leak_context(
    monkeypatch,
):

    history = [
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
        {
            "role": "user",
            "content": (
                "Now tell me about hostel facilities."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hostel facilities details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Now tell me about Electrical Engineering research."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Electrical Engineering research details."
            ),
        },
    ]

    # -----------------------------------------------------
    # Follow-up must use Research
    # -----------------------------------------------------

    install_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "Which Electrical Engineering research areas "
                    "are related to VLSI?"
                ),
                active_topic="research",
                active_entity="Electrical Engineering",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about VLSI?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"

    resolved = (
        result["resolved_question"].lower()
    )

    assert "electrical engineering" in resolved
    assert "vlsi" in resolved
    assert "hostel" not in resolved
    assert "m.tech" not in resolved

    # -----------------------------------------------------
    # Explicit return to M.Tech
    # -----------------------------------------------------

    history.extend(
        [
            {
                "role": "user",
                "content": "What about VLSI?",
            },
            {
                "role": "assistant",
                "content": "VLSI research details.",
            },
        ]
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "Now let's go back to M.Tech admission."
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert "m.tech" in (
        result["resolved_question"].lower()
    )

    assert "research" not in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7C.3 — Switch + follow-up chain
# =========================================================

def test_follow_up_chain_after_topic_switch(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "Tell me about Ph.D. eligibility."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Ph.D. eligibility details."
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

    # -----------------------------------------------------
    # Follow-up 1
    # -----------------------------------------------------

    install_sequence(
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

    history.extend(
        [
            {
                "role": "user",
                "content": "What about fees?",
            },
            {
                "role": "assistant",
                "content": "Hostel fee details.",
            },
        ]
    )

    # -----------------------------------------------------
    # Follow-up 2
    # -----------------------------------------------------

    install_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What is the single-occupancy hostel rate?"
                ),
                active_topic="hostel",
                active_entity="hostel",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about single occupancy?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"

    resolved = (
        result["resolved_question"].lower()
    )

    assert "single" in resolved
    assert "occupancy" in resolved
    assert "hostel" in resolved


# =========================================================
# 7C.4 — Ambiguity after multiple topics
# =========================================================

def test_ambiguous_reference_after_topic_switches_is_safe(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Pure ambiguity must not be guessed."
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
                "Now tell me about hostel facilities."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hostel facility details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Now tell me about Electrical Engineering research."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Electrical Engineering research details."
            ),
        },
    ]

    result = conversation_resolver.resolve_conversation(
        question="What about that?",
        chat_history=history,
    )

    assert result["mode"] == "ambiguous"

    assert (
        result["resolved_question"]
        == "What about that?"
    )

    assert result["active_topic"] == ""
    assert result["active_entity"] == ""


# =========================================================
# 7C.5 — New topic with explicit wording must win
# =========================================================

def test_explicit_new_subject_wins_over_recent_context(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Self-contained new subject should not invoke resolver."
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
                "Tell me about hostel fees."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hostel fee details."
            ),
        },
        {
            "role": "user",
            "content": (
                "What about single occupancy?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Single occupancy details."
            ),
        },
    ]

    result = conversation_resolver.resolve_conversation(
        question=(
            "What are the regular Ph.D. eligibility requirements?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert (
        result["resolved_question"]
        == "What are the regular Ph.D. eligibility requirements?"
    )


# =========================================================
# 7C.6 — Switch back and then continue
# =========================================================

def test_switch_back_then_follow_up_uses_returned_topic(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "Tell me about Electrical Engineering research."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Electrical Engineering research details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Now tell me about M.Sc. programs."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "M.Sc. program details."
            ),
        },
    ]

    # -----------------------------------------------------
    # Return to Electrical Engineering
    # -----------------------------------------------------

    result = conversation_resolver.resolve_conversation(
        question=(
            "Let's go back to Electrical Engineering."
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert "electrical engineering" in (
        result["resolved_question"].lower()
    )

    history.extend(
        [
            {
                "role": "user",
                "content": (
                    "Let's go back to Electrical Engineering."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Electrical Engineering details."
                ),
            },
        ]
    )

    # -----------------------------------------------------
    # Follow-up should now use Electrical Engineering
    # -----------------------------------------------------

    install_sequence(
        monkeypatch,
        [
            fake_response(
                mode="follow_up",
                resolved_question=(
                    "What Electrical Engineering research areas "
                    "are related to power systems?"
                ),
                active_topic="research",
                active_entity="Electrical Engineering",
            )
        ],
    )

    result = conversation_resolver.resolve_conversation(
        question="What about power systems?",
        chat_history=history,
    )

    assert result["mode"] == "follow_up"

    resolved = (
        result["resolved_question"].lower()
    )

    assert "electrical engineering" in resolved
    assert "power systems" in resolved
    assert "m.sc." not in resolved


# =========================================================
# 7C.7 — Explicit switch should not inherit old entity
# =========================================================

def test_explicit_switch_drops_old_entity(
    monkeypatch,
):

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Explicit self-contained switch should not call resolver."
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
                "What research areas are available "
                "in Electrical Engineering?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Electrical Engineering research details."
            ),
        },
    ]

    result = conversation_resolver.resolve_conversation(
        question=(
            "What are the hostel facilities?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert (
        result["resolved_question"]
        == "What are the hostel facilities?"
    )

# =========================================================
# 7C.8 — Rapid switching
# =========================================================

def test_rapid_topic_switching(
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
            "content": "Ph.D. admission.",
        },
    ]

    # A → Hostel
    result = conversation_resolver.resolve_conversation(
        question="Tell me about hostel facilities.",
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    history.extend(
        [
            {
                "role": "user",
                "content": "Tell me about hostel facilities.",
            },
            {
                "role": "assistant",
                "content": "Hostel facilities.",
            },
        ]
    )

    # B → Research
    result = conversation_resolver.resolve_conversation(
        question=(
            "What research areas are available in Electrical Engineering?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    history.extend(
        [
            {
                "role": "user",
                "content": (
                    "What research areas are available in Electrical Engineering?"
                ),
            },
            {
                "role": "assistant",
                "content": "Research areas.",
            },
        ]
    )

    # C → M.Sc.
    result = conversation_resolver.resolve_conversation(
        question=(
            "What are the M.Sc. admission requirements?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert "m.sc." in (
        result["resolved_question"].lower()
    )

    assert "hostel" not in (
        result["resolved_question"].lower()
    )

    assert "electrical engineering" not in (
        result["resolved_question"].lower()
    )


# =========================================================
# 7C.9 — Old answer must not become current topic by itself
# =========================================================

def test_old_assistant_answer_does_not_override_new_user_topic(
    monkeypatch,
):

    history = [
        {
            "role": "user",
            "content": (
                "Tell me about hostel fees."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Hostel charges depend on occupancy and duration."
            ),
        },
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Self-contained new question should not invoke resolver."
        )

    monkeypatch.setattr(
        conversation_resolver,
        "invoke_query_llm",
        fail_if_called,
    )

    result = conversation_resolver.resolve_conversation(
        question=(
            "What research areas are available in Electrical Engineering?"
        ),
        chat_history=history,
    )

    assert result["mode"] == "topic_switch"

    assert (
        "hostel"
        not in result["resolved_question"].lower()
    )

    assert (
        "electrical engineering"
        in result["resolved_question"].lower()
    )