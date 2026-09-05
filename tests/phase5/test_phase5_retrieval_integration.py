"""
Phase 5 -> Existing Retrieval Integration Tests

These tests verify that the four Phase-5 layers actually form one
pipeline before real retrieval/evidence processing begins.

They intentionally test difficult student language rather than simple
"What programs are available?" questions.
"""

from types import SimpleNamespace

from backend.situation_retrieval_integration import (
    prepare_situation_retrieval,
)


def test_personal_phd_question_reaches_retrieval_ready_state():
    result = prepare_situation_retrieval(
        "I have a four-year B.Tech with 74% and want regular Ph.D. admission."
    )

    situation = result[
        "student_situation"
    ]

    decision = result[
        "decision_context"
    ]

    plan = result[
        "retrieval_plan"
    ]

    control = result[
        "retrieval_control"
    ]

    assert situation.intent == "phd_eligibility"

    assert decision.target == "phd"
    assert decision.goal == "determine_eligibility"

    assert decision.facts[
        "degree"
    ] == "bachelors_degree"

    assert 74.0 in decision.facts[
        "percentages"
    ]

    query_blob = " ".join(
        result["generated_queries"]
    ).lower()

    assert "ph.d." in query_blob
    assert "bachelor" in query_blob
    assert "74 percent" in query_blob
    assert "eligibility" in query_blob

    assert control.mode == "situational"
    assert plan.primary_query


def test_hostel_problem_preserves_real_decision_constraints():
    result = prepare_situation_retrieval(
        "I'll stay 20 days and I'm okay sharing a room to save money."
    )

    decision = result[
        "decision_context"
    ]

    assert decision.target == "hostel"
    assert decision.goal == "minimize_cost"

    assert decision.constraints[
        "stay_duration"
    ] == {
        "value": 20.0,
        "unit": "days",
    }

    assert decision.constraints[
        "occupancy"
    ] == "double"

    assert decision.preferences[
        "cost_sensitive"
    ] is True

    query_blob = " ".join(
        result["generated_queries"]
    ).lower()

    assert "20 days" in query_blob
    assert "double occupancy" in query_blob


def test_research_problem_preserves_interest_and_target():
    result = prepare_situation_retrieval(
        "My interest is robotics and control systems, and I'm considering Electrical Engineering."
    )

    situation = result[
        "student_situation"
    ]

    decision = result[
        "decision_context"
    ]

    queries = result[
        "generated_queries"
    ]

    assert situation.intent == "research"

    assert decision.target == (
        "electrical engineering"
    )

    blob = " ".join(
        queries
    ).lower()

    assert "robotics" in blob
    assert "control systems" in blob
    assert "electrical engineering" in blob


def test_irrelevant_personal_story_does_not_pollute_retrieval():
    result = prepare_situation_retrieval(
        "I play football every weekend and love music, "
        "but my B.Tech is 72% and my main concern is Ph.D. eligibility."
    )

    blob = " ".join(
        result["generated_queries"]
    ).lower()

    assert "football" not in blob
    assert "music" not in blob

    assert "72 percent" in blob
    assert "bachelor" in blob
    assert "eligibility" in blob


def test_negative_hostel_constraint_survives_all_four_layers():
    result = prepare_situation_retrieval(
        "I need my own room for five days and I don't need bedding."
    )

    decision = result[
        "decision_context"
    ]

    assert decision.constraints[
        "occupancy"
    ] == "single"

    assert decision.constraints[
        "bedding"
    ] == "without"

    blob = " ".join(
        result["generated_queries"]
    ).lower()

    assert "single occupancy" in blob
    assert "without bedding" in blob
    assert "with bedding" not in blob


def test_low_confidence_path_is_conservative():
    """
    Directly exercise the Phase-5 control contract through a deliberately
    low-confidence context. This proves the integration does not blindly
    fan out when understanding is uncertain.
    """
    from backend.situation_retrieval_control import (
        build_retrieval_control,
    )

    context = SimpleNamespace(
        confidence=0.20,
        facts={},
        preferences={},
        constraints={},
    )

    result = build_retrieval_control(
        original_query=(
            "I'm not really sure what I need."
        ),
        primary_query=(
            "college eligibility requirements"
        ),
        alternate_queries=(
            "admission requirements",
            "program requirements",
        ),
        context=context,
    )

    assert result.mode == "conservative"

    assert result.queries == (
        "I'm not really sure what I need.",
    )


def test_retrieval_fanout_is_bounded():
    result = prepare_situation_retrieval(
        "I have a B.Tech with 74%, prefer research, "
        "need hostel accommodation for 20 days, "
        "can share a room, and want to minimize cost "
        "while applying for regular Ph.D."
    )

    assert len(
        result["generated_queries"]
    ) <= 3


def test_phase5_does_not_make_policy_decision():
    result = prepare_situation_retrieval(
        "I have 74% and want to apply for regular Ph.D."
    )

    blob = " ".join(
        result["generated_queries"]
    ).lower()

    # Retrieval planning must search for the rule.
    # It must not claim the conclusion.
    assert "eligible" not in blob
    assert "not eligible" not in blob


def test_phase5_output_is_deterministic():
    question = (
        "I have a four-year B.Tech with 74% "
        "and want regular Ph.D. admission."
    )

    first = prepare_situation_retrieval(
        question
    )

    second = prepare_situation_retrieval(
        question
    )

    assert first[
        "generated_queries"
    ] == second[
        "generated_queries"
    ]

    assert first[
        "retrieval_plan"
    ] == second[
        "retrieval_plan"
    ]

    assert first[
        "retrieval_control"
    ] == second[
        "retrieval_control"
    ]
