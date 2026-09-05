"""
Phase 4 — Conversation Grounding Regression Tests

Purpose
-------
Validate that short follow-up questions preserve the active topic and
entity from the immediately preceding conversation.

These tests intentionally use natural wording variations rather than
one hardcoded sentence.
"""

from backend.conversation_resolver import (
    resolve_conversation,
)


# =========================================================
# Shared conversation context
# =========================================================

INITIAL_HISTORY = [
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
            "The regular Ph.D. admission requirements "
            "include master's and four-year bachelor's routes."
        ),
    },
]


# =========================================================
# Helper
# =========================================================

def resolve_followup(
    question: str,
):
    return resolve_conversation(
        question=question,
        chat_history=INITIAL_HISTORY,
    )


# =========================================================
# Core follow-up
# =========================================================

def test_bachelors_route_followup_preserves_phd_context():

    result = resolve_followup(
        "What about the bachelor's degree route?"
    )

    resolved = result[
        "resolved_question"
    ].lower()

    assert (
        "ph.d"
        in resolved
        or
        "phd"
        in resolved
    )

    assert (
        "bachelor"
        in resolved
    )


# =========================================================
# Natural wording variations
# =========================================================

def test_short_bachelors_followup():

    result = resolve_followup(
        "What about the bachelor's route?"
    )

    resolved = result[
        "resolved_question"
    ].lower()

    assert "bachelor" in resolved
    assert (
        "ph.d"
        in resolved
        or
        "phd"
        in resolved
    )


def test_elliptical_bachelors_followup():

    result = resolve_followup(
        "And the bachelor's option?"
    )

    resolved = result[
        "resolved_question"
    ].lower()

    assert "bachelor" in resolved
    assert (
        "ph.d"
        in resolved
        or
        "phd"
        in resolved
    )


def test_route_followup():

    result = resolve_followup(
        "What is the other degree route?"
    )

    resolved = result[
        "resolved_question"
    ].lower()

    assert (
        "ph.d"
        in resolved
        or
        "phd"
        in resolved
    )


# =========================================================
# Topic continuity
# =========================================================

def test_percentage_followup_preserves_admission_topic():

    result = resolve_followup(
        "What percentage is needed for that route?"
    )

    resolved = result[
        "resolved_question"
    ].lower()

    assert (
        "ph.d"
        in resolved
        or
        "phd"
        in resolved
    )

    assert (
        "percentage"
        in resolved
        or
        "marks"
        in resolved
        or
        "cgpa"
        in resolved
    )


# =========================================================
# Resolver should not fabricate a new topic
# =========================================================

def test_unrelated_new_question_can_stand_alone():

    result = resolve_followup(
        "What are the hostel fees?"
    )

    resolved = result[
        "resolved_question"
    ].lower()

    assert "hostel" in resolved
    assert (
        "ph.d"
        not in resolved
        and
        "phd"
        not in resolved
    )