"""
Phase 5.5 — Real Graph Evidence Test

IMPORTANT:
-----------
This test does NOT modify backend/nodes.py or integrate 5.5.

It invokes the CURRENT production graph, captures:
    - final answer
    - actual answer context

Then it runs the standalone Phase-5.5 grounding detector against that
real output.

Purpose:
    Prove that 5.5 is relevant to real production failures before wiring
    it into generate_answer().
"""

from backend.graph import create_graph
from backend.core.answering.grounding import  assess_answer_grounding


def invoke_real_graph(question: str):
    graph = create_graph()

    return graph.invoke(
        {
            "question": question,
            "chat_history": [],
        }
    )


def print_case(
    name: str,
    question: str,
    result: dict,
):
    answer = result.get(
        "answer",
        "",
    )

    context = result.get(
        "context",
        "",
    )

    grounding = assess_answer_grounding(
        answer=answer,
        evidence=context,
    )

    print("\n" + "=" * 110)
    print(name)
    print("=" * 110)

    print("\nQUESTION:")
    print(question)

    print("\nEVIDENCE:")
    print(context[:6000])

    print("\nANSWER:")
    print(answer)

    print("\n5.5 RESULT:")
    print(
        "status:",
        grounding["status"],
    )
    print(
        "issue_count:",
        grounding["issue_count"],
    )

    for issue in grounding["issues"]:
        print(
            {
                "type": issue.issue_type,
                "claim": issue.answer_claim,
                "reason": issue.reason,
            }
        )

    return grounding


def test_real_graph_phd_case_is_checked_by_5_5():
    """
    Real production case from the Phase-5 testing.

    The earlier observed answer made an unsupported GATE exemption
    conclusion from a percentage threshold.

    This test is diagnostic: 5.5 should detect a high-risk relationship
    if that behavior is still present.
    """
    question = (
        "I have a four-year B.Tech with 74% and want regular Ph.D. admission."
    )

    result = invoke_real_graph(
        question
    )

    grounding = print_case(
        "REAL GRAPH — Ph.D. ELIGIBILITY",
        question,
        result,
    )

    answer = result.get(
        "answer",
        "",
    ).lower()

    # We only assert the detector's behavior if the problematic
    # relationship actually occurs in the live answer.
    if (
        "gate" in answer
        and (
            "exempt" in answer
            or "exemption" in answer
        )
    ):
        assert grounding["status"] == "review"

        assert any(
            issue.issue_type
            in {
                "exemption_conclusion",
                "unsupported_causal_claim",
                "eligibility_conclusion",
            }
            for issue in grounding["issues"]
        )


def test_real_graph_hostel_case_is_checked_by_5_5():
    """
    Real production case from the Phase-5 testing.

    Earlier output interpreted ₹2,175 as the exact cost of the user's
    20-day stay.

    5.5 should identify that as a risky personalized cost interpretation.
    """
    question = (
        "I'll stay for 20 days and I'm okay sharing a room to save money."
    )

    result = invoke_real_graph(
        question
    )

    grounding = print_case(
        "REAL GRAPH — HOSTEL COST",
        question,
        result,
    )

    answer = result.get(
        "answer",
        "",
    ).lower()

    if (
        "2175" in answer
        or "2,175" in answer
    ):
        if (
            "your" in answer
            and (
                "cost" in answer
                or "fee" in answer
                or "charge" in answer
            )
        ):
            assert grounding["status"] == "review"

            assert any(
                issue.issue_type
                in {
                    "personal_cost_interpretation",
                    "unsupported_numeric_relationship",
                }
                for issue in grounding["issues"]
            )


def test_real_graph_research_case_can_remain_grounded():
    """
    Real production case where the retrieved evidence and final answer
    were already substantially aligned.

    5.5 should not reject ordinary grounded research answers.
    """
    question = (
        "My interest is robotics and control systems, "
        "and I'm considering Electrical Engineering."
    )

    result = invoke_real_graph(
        question
    )

    grounding = print_case(
        "REAL GRAPH — RESEARCH",
        question,
        result,
    )

    assert grounding["status"] in {
        "grounded",
        "review",
    }

    # If a review occurs here, it must be inspectable rather than hidden.
    if grounding["status"] == "review":
        assert grounding["issue_count"] > 0
