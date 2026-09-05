"""
Phase 5.2 — Decision Context Tests

These tests validate separation of:
- facts
- preferences
- constraints
- goal
- target
- missing information
"""

from backend.student_situation import (
    understand_student_situation,
)

from backend.situation_decision import (
    build_decision_context,
    decision_context_to_dict,
)


def test_phd_decision_context_separates_facts_and_target():
    situation = understand_student_situation(
        "I have a four-year B.Tech with 72% and want regular Ph.D. admission because I prefer research."
    )

    context = build_decision_context(
        situation
    )

    assert context.goal == "determine_eligibility"
    assert context.target == "phd"

    assert context.facts["degree"] == "bachelors_degree"
    assert context.facts["degree_duration_years"] == 4
    assert 72.0 in context.facts["percentages"]

    assert context.preferences[
        "research_oriented"
    ] is True

    assert "research_oriented" not in context.facts
    assert "degree" not in context.preferences
    assert "phd" not in context.facts


def test_hostel_decision_context_separates_constraint_and_preference():
    situation = understand_student_situation(
        "I'll stay for 20 days and I'm comfortable sharing a room to save money."
    )

    context = build_decision_context(
        situation
    )

    assert context.goal == "minimize_cost"
    assert context.target == "hostel"

    assert context.constraints[
        "stay_duration"
    ] == {
        "value": 20.0,
        "unit": "days",
    }

    assert context.constraints[
        "occupancy"
    ] == "double"

    assert context.preferences[
        "cost_sensitive"
    ] is True


def test_hostel_bedding_is_a_constraint():
    situation = understand_student_situation(
        "I need a single room for 5 days and don't need bedding."
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "hostel"

    assert context.constraints[
        "occupancy"
    ] == "single"

    assert context.constraints[
        "bedding"
    ] == "without"


def test_research_interest_becomes_preference_information():
    situation = understand_student_situation(
        "My interest is robotics and control systems and I want to find a suitable research direction."
    )

    context = build_decision_context(
        situation
    )

    assert context.goal == "find_relevant_research"
    assert context.target == "research"

    interests = context.preferences[
        "research_interests"
    ]

    assert "robotics" in interests
    assert "control systems" in interests


def test_research_with_department_uses_department_as_target():
    situation = understand_student_situation(
        "I'm interested in robotics and control systems in Electrical Engineering."
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "electrical engineering"


def test_comparison_has_decision_goal():
    situation = understand_student_situation(
        "I'm deciding between M.Tech and M.S. by Research because I prefer research over coursework."
    )

    context = build_decision_context(
        situation
    )

    assert context.goal == "compare_options"
    assert context.target == "m.tech"

    assert context.preferences[
        "research_oriented"
    ] is True

    assert context.constraints[
        "comparison_requested"
    ] is True


def test_eligibility_without_personal_facts_marks_missing_information():
    situation = understand_student_situation(
        "Can I apply for a regular Ph.D.?"
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "phd"
    assert (
        "personal_qualification_details"
        in context.missing_information
    )


def test_hostel_minimize_cost_without_details_marks_missing_information():
    situation = understand_student_situation(
        "I want the cheapest hostel option."
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "hostel"
    assert (
        "stay_duration_or_budget"
        in context.missing_information
    )


def test_all_missing_information_is_advisory():
    situation = understand_student_situation(
        "I'm comparing options."
    )

    context = build_decision_context(
        situation
    )

    assert (
        "decision_criteria"
        in context.missing_information
    )


def test_no_fabricated_facts():
    situation = understand_student_situation(
        "I'm thinking about applying next year."
    )

    context = build_decision_context(
        situation
    )

    assert context.facts == {}
    assert "2028" not in context.facts
    assert "age" not in context.facts
    assert "salary" not in context.facts


def test_navigation_target_is_structured():
    situation = understand_student_situation(
        "I'm new on campus and need directions to the library."
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "campus_service"
    assert context.goal == "locate_service"


def test_emergency_target_is_structured():
    situation = understand_student_situation(
        "My friend got hurt near the academic block. Where should I take him?"
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "medical_service"
    assert context.goal == "locate_service"


def test_custom_entity_can_become_target():
    situation = understand_student_situation(
        "I'm considering the School of Quantum Computing.",
        entity_terms=(
            "School of Quantum Computing",
        ),
    )

    context = build_decision_context(
        situation
    )

    assert (
        context.target
        == "School of Quantum Computing"
    )


def test_serialization_is_json_friendly():
    situation = understand_student_situation(
        "I have a four-year B.Tech with 72% and want regular Ph.D. admission."
    )

    context = build_decision_context(
        situation
    )

    payload = decision_context_to_dict(
        context
    )

    assert isinstance(
        payload,
        dict,
    )

    assert isinstance(
        payload["facts"],
        dict,
    )

    assert isinstance(
        payload["preferences"],
        dict,
    )

    assert isinstance(
        payload["constraints"],
        dict,
    )

    assert isinstance(
        payload["missing_information"],
        list,
    )


def test_confidence_is_preserved():
    situation = understand_student_situation(
        "I have a B.Tech and want to apply for Ph.D."
    )

    context = build_decision_context(
        situation
    )

    assert (
        context.confidence
        == situation.confidence
    )


def test_fact_preference_constraint_names_do_not_leak_into_each_other():
    situation = understand_student_situation(
        "I have a B.Tech, prefer research, and need a shared hostel room."
    )

    context = build_decision_context(
        situation
    )

    assert "degree" in context.facts
    assert "research_oriented" in context.preferences
    assert "occupancy" in context.constraints

    assert "degree" not in context.preferences
    assert "degree" not in context.constraints
    assert "research_oriented" not in context.facts
    assert "research_oriented" not in context.constraints


def test_decision_context_has_stable_target_for_plain_hostel_question():
    situation = understand_student_situation(
        "I need information about hostel accommodation."
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "hostel"


def test_empty_message_remains_safe():
    situation = understand_student_situation(
        ""
    )

    context = build_decision_context(
        situation
    )

    assert context.goal == "find_information"
    assert context.target == "general_information"
    assert context.facts == {}
    assert context.preferences == {}
    assert context.constraints == {}


def test_explicit_budget_is_kept_as_constraint():
    situation = understand_student_situation(
        "I need hostel accommodation under ₹3000."
    )

    context = build_decision_context(
        situation
    )

    assert context.target == "hostel"
    assert context.constraints[
        "budget"
    ] == 3000.0


def test_research_preference_does_not_become_a_college_fact():
    situation = understand_student_situation(
        "I prefer research over coursework."
    )

    context = build_decision_context(
        situation
    )

    assert context.preferences[
        "research_oriented"
    ] is True

    assert "research_oriented" not in context.facts
