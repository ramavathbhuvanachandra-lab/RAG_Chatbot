"""
Phase 5.6 — Situational Answer Quality Tests

Tests deliberately target realistic student language and ensure that
situation information reaches the answer layer without changing the
retrieval/evidence architecture.
"""

from types import SimpleNamespace

from backend.situational_answer import (
    augment_answer_question,
    build_situational_answer_context,
)


def test_phd_student_facts_are_preserved():

    situation = SimpleNamespace(
        intent="phd_eligibility",
        goal="determine_eligibility",
        user_facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (74.0,),
        },
        constraints={},
        preferences={},
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert (
        "goal: determine_eligibility"
        in result
    )

    assert (
        "degree: bachelors_degree"
        in result
    )

    assert (
        "degree_duration_years: 4"
        in result
    )

    assert "74.0" in result


def test_hostel_constraints_are_preserved():

    situation = SimpleNamespace(
        intent="hostel",
        goal="minimize_cost",
        user_facts={},
        constraints={
            "stay_duration": {
                "value": 5.0,
                "unit": "days",
            },
            "occupancy": "single",
            "bedding": "without",
        },
        preferences={
            "cost_sensitive": True,
        },
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert "stay_duration" in result
    assert "5.0" in result
    assert "single" in result
    assert "without" in result
    assert "cost_sensitive" in result


def test_decision_context_is_preferred():

    situation = SimpleNamespace(
        intent="hostel",
        goal="find_information",
        user_facts={},
        constraints={},
        preferences={},
    )

    decision = SimpleNamespace(
        target="hostel",
        goal="minimize_cost",
        facts={
            "resident_type": "student",
        },
        constraints={
            "occupancy": "double",
        },
        preferences={
            "cost_sensitive": True,
        },
    )

    result = build_situational_answer_context(
        situation=situation,
        decision_context=decision,
    )

    assert "target: hostel" in result
    assert "goal: minimize_cost" in result
    assert "resident_type: student" in result
    assert "occupancy: double" in result
    assert "cost_sensitive" in result


def test_missing_information_is_not_invented():

    situation = SimpleNamespace(
        intent="hostel",
        goal="minimize_cost",
        user_facts={},
        constraints={},
        preferences={},
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert "occupancy:" not in result
    assert "stay_duration:" not in result
    assert "resident_type:" not in result


def test_question_is_preserved_exactly():

    question = (
        "I have 74% in B.Tech. Can I apply for Ph.D.?"
    )

    situation = SimpleNamespace(
        intent="phd_eligibility",
        goal="determine_eligibility",
        user_facts={
            "percentages": (74.0,),
        },
        constraints={},
        preferences={},
    )

    result = augment_answer_question(
        answer_question=question,
        situation=situation,
    )

    assert result.startswith(
        question
    )


def test_answer_rules_protect_evidence_boundary():

    situation = SimpleNamespace(
        intent="phd_eligibility",
        goal="determine_eligibility",
        user_facts={
            "percentages": (74.0,),
        },
        constraints={},
        preferences={},
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert (
        "Do not invent missing facts"
        in result
    )

    assert (
        "Do not turn a user fact into an institutional rule"
        in result
    )

    assert (
        "Do not infer eligibility"
        in result
    )


def test_multi_intent_can_use_same_situation_context():

    decision = SimpleNamespace(
        target="college",
        goal="answer_multiple_requests",
        facts={
            "student_status": "prospective_student",
        },
        constraints={},
        preferences={},
    )

    result = build_situational_answer_context(
        decision_context=decision,
    )

    assert (
        "student_status: prospective_student"
        in result
    )


def test_empty_inputs_are_safe():

    result = build_situational_answer_context()

    assert result == ""


def test_empty_context_does_not_change_question():

    question = (
        "Where is the library?"
    )

    result = augment_answer_question(
        answer_question=question,
    )

    assert result == question


def test_context_is_deterministic():

    situation = SimpleNamespace(
        intent="hostel",
        goal="minimize_cost",
        user_facts={
            "resident_type": "student",
        },
        constraints={
            "occupancy": "double",
        },
        preferences={
            "cost_sensitive": True,
        },
    )

    first = build_situational_answer_context(
        situation=situation,
    )

    second = build_situational_answer_context(
        situation=situation,
    )

    assert first == second


def test_user_fact_is_not_converted_into_policy():

    situation = SimpleNamespace(
        intent="phd_eligibility",
        goal="determine_eligibility",
        user_facts={
            "percentages": (74.0,),
        },
        constraints={},
        preferences={},
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert (
        "74.0 is eligible"
        not in result.lower()
    )


def test_personal_preference_is_preserved():

    situation = SimpleNamespace(
        intent="hostel",
        goal="minimize_cost",
        user_facts={},
        constraints={
            "occupancy": "double",
        },
        preferences={
            "cost_sensitive": True,
        },
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert (
        "cost_sensitive: True"
        in result
    )


def test_three_month_duration_is_preserved():

    situation = SimpleNamespace(
        intent="hostel",
        goal="find_information",
        user_facts={},
        constraints={
            "stay_duration": {
                "value": 3.0,
                "unit": "months",
            },
        },
        preferences={},
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert "3.0" in result
    assert "months" in result


def test_irrelevant_facts_remain_facts_without_becoming_policy():

    situation = SimpleNamespace(
        intent="phd_eligibility",
        goal="determine_eligibility",
        user_facts={
            "hobby": "football",
            "percentages": (72.0,),
        },
        constraints={},
        preferences={},
    )

    result = build_situational_answer_context(
        situation=situation,
    )

    assert "hobby: football" in result
    assert "72.0" in result


def test_augmented_prompt_contains_original_question_and_rules():

    question = (
        "I need a single room for five days."
    )

    situation = SimpleNamespace(
        intent="hostel",
        goal="find_information",
        user_facts={},
        constraints={
            "stay_duration": {
                "value": 5.0,
                "unit": "days",
            },
            "occupancy": "single",
        },
        preferences={},
    )

    result = augment_answer_question(
        answer_question=question,
        situation=situation,
    )

    assert result.startswith(
        question
    )

    assert (
        "stay_duration"
        in result
    )

    assert (
        "occupancy: single"
        in result
    )

    assert (
        "available information"
        in result
    )
