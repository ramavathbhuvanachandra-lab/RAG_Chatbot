"""
Answer Scope / Grounding Tests

Purpose
-------
Validate that answers remain focused on the user's question and
do not introduce unsupported or unnecessarily broad claims.

These tests intentionally use the current corpus as a stress test.
The production behavior must remain college-agnostic.
"""

from backend.graph import create_graph


# =========================================================
# Helper
# =========================================================

def invoke(
    question: str,
):
    """
    Invoke the current production graph for one question.
    """

    graph = create_graph()

    return graph.invoke(
        {
            "question": question,
            "chat_history": [],
        }
    )


# =========================================================
# M.Tech eligibility
# =========================================================

def test_mtech_eligibility_answer_stays_scope_focused():
    """
    A regular M.Tech eligibility question should primarily answer
    eligibility/qualification requirements.
    """

    result = invoke(
        "What are the eligibility requirements "
        "for regular M.Tech. admission?"
    )

    answer = result["answer"].lower()

    assert "bachelor" in answer

    assert (
        "60%"
        in answer
        or "6.0"
        in answer
    )

    # Do not turn a direct eligibility question into an application
    # fee/process answer.
    assert "application fee" not in answer


# =========================================================
# Ph.D. eligibility
# =========================================================

def test_phd_eligibility_answer_contains_core_requirements():
    """
    The answer should contain the primary master's-degree route.
    """

    result = invoke(
        "What are the eligibility requirements "
        "for regular Ph.D. admission?"
    )

    answer = result["answer"].lower()

    assert "master" in answer

    assert (
        "60%"
        in answer
        or "6.0"
        in answer
    )


def test_phd_eligibility_answer_contains_supported_alternative_route():
    """
    The answer should represent the supported alternative bachelor's
    degree eligibility route.

    The test deliberately does not require one exact phrase.
    """

    result = invoke(
        "What are the eligibility requirements "
        "for regular Ph.D. admission?"
    )

    answer = result["answer"].lower()

    alternative_route_signals = (
        "bachelor",
        "70%",
        "7.0",
        "four-year",
    )

    assert any(
        signal in answer
        for signal in alternative_route_signals
    )


# =========================================================
# Hostel fees
# =========================================================

def test_hostel_fee_answer_contains_actual_hostel_rates():
    """
    The hostel-fee answer should contain the actual accommodation
    rates supported by the current corpus.
    """

    result = invoke(
        "What are the hostel fees?"
    )

    answer = result["answer"]

    assert (
        "₹300" in answer
        or "300" in answer
    )

    assert (
        "₹500" in answer
        or "500" in answer
    )

    assert (
        "₹2,175" in answer
        or "2175" in answer
    )


# =========================================================
# Electrical Engineering research
# =========================================================

def test_electrical_research_answer_stays_research_focused():
    """
    A research-area question should not drift into admission
    eligibility content.
    """

    result = invoke(
        "What research areas are available "
        "in Electrical Engineering?"
    )

    answer = result["answer"].lower()

    assert "research" in answer

    assert (
        "eligibility requirement"
        not in answer
    )


# =========================================================
# Internal retrieval leakage
# =========================================================

def test_internal_retrieval_labels_do_not_leak():
    """
    Internal debugging/retrieval terminology must never appear in
    the user-facing answer.
    """

    questions = [
        "What research areas are available in Electrical Engineering?",
        "What are the eligibility requirements for regular Ph.D. admission?",
        "What are the hostel fees?",
    ]

    forbidden_markers = (
        "document 1",
        "document 2",
        "retrieval rank",
        "rrf score",
        "chunk id",
    )

    for question in questions:

        result = invoke(
            question
        )

        answer = result["answer"].lower()

        for marker in forbidden_markers:

            assert marker not in answer