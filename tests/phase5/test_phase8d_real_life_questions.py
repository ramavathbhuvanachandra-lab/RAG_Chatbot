"""
IIT Jodhpur V1 — Phase 8D
Final real-life conversational regression tests.

These tests validate the behavior of the Phase-5 planning path using
natural user wording. They do not require every noun in a question to be
an extracted entity.

Important semantic distinctions:
    - research is an intent, not necessarily an entity;
    - library may be a real user target without being a configured entity;
    - explicit academic program entities must remain specific.
"""

from __future__ import annotations

import pytest

from backend.student_situation import understand_student_situation
from backend.situation_decision import build_decision_context
from backend.retrieval_query_planner import build_retrieval_plan
from backend.situation_retrieval_control import build_retrieval_control


REAL_LIFE_CASES = [
    (
        "What are the admission routes for M.Sc.?",
        "m.sc",
        "m.sc",
    ),
    (
        "How can I get into the MSc program?",
        "m.sc",
        "m.sc",
    ),
    (
        "I did my B.Tech and now I want PhD at IIT Jodhpur. What are the requirements?",
        "phd",
        "phd",
    ),
    (
        "I have 72% in B.Tech. Can I apply for a Ph.D.?",
        "phd",
        "phd",
    ),
    (
        "Can I join M.S. by Research after my bachelor's?",
        "m.s. by research",
        "m.s. by research",
    ),
    (
        "Is MS by Research available at IIT Jodhpur?",
        "m.s. by research",
        "m.s. by research",
    ),
    (
        "What is needed for M.Tech admission?",
        "m.tech",
        "m.tech",
    ),
    (
        "I'm interested in robotics in Electrical Engineering.",
        "electrical engineering",
        "electrical engineering",
    ),
    (
        "Which Electrical Engineering research area mentions robotics?",
        "electrical engineering",
        "electrical engineering",
    ),
    (
        "How much does hostel accommodation cost for a short stay?",
        "hostel",
        "hostel",
    ),
    (
        "Where is the library on campus?",
        None,
        "facilities",
    ),
]


@pytest.mark.parametrize(
    "question,expected_entity,expected_target",
    REAL_LIFE_CASES,
)
def test_real_life_question_pipeline(
    question: str,
    expected_entity: str | None,
    expected_target: str,
):
    situation = understand_student_situation(
        question
    )
    context = build_decision_context(
        situation
    )
    plan = build_retrieval_plan(
        situation,
        context,
    )
    control = build_retrieval_control(
        original_query=question,
        primary_query=plan.primary_query,
        alternate_queries=plan.alternate_queries,
        context=context,
        preserved_signals=plan.preserved_signals,
        confidence=context.confidence,
    )

    if expected_entity is not None:
        assert expected_entity in situation.entities

    assert context.target == expected_target
    assert plan.primary_query.strip()
    assert control.queries


def test_research_interest_is_not_required_to_create_research_entity():
    question = (
        "I am interested in robotics and control systems. "
        "Which research area should I look at?"
    )

    situation = understand_student_situation(
        question
    )
    context = build_decision_context(
        situation
    )

    assert situation.intent == "research"
    assert context.target == "research"
    assert "robotics" in situation.entities
    assert "control systems" in situation.entities


def test_library_question_does_not_require_library_entity_extraction():
    situation = understand_student_situation(
        "Where is the library on campus?"
    )
    context = build_decision_context(
        situation
    )

    assert situation.goal == "find_information"
    assert context.target == "facilities"


def test_msc_variants_never_collapse_into_ms():
    questions = [
        "What are the admission routes for M.Sc.?",
        "How can I get into the MSc program?",
        "I want to apply for M.Sc. admission. What do I do?",
        "Tell me about admission to the MSc program.",
    ]

    for question in questions:
        situation = understand_student_situation(
            question
        )
        assert "m.sc" in situation.entities
        assert "m.s." not in situation.entities


def test_ms_by_research_variants_stay_compound():
    questions = [
        "Can I join M.S. by Research after my bachelor's?",
        "Is MS by Research available at IIT Jodhpur?",
        "What is the admission process for M.S by Research?",
    ]

    for question in questions:
        situation = understand_student_situation(
            question
        )
        assert situation.entities == (
            "m.s. by research",
        )


def test_background_degree_does_not_replace_phd_target():
    situation = understand_student_situation(
        "I completed B.Tech with 74% and now want regular Ph.D. admission."
    )
    context = build_decision_context(
        situation
    )

    assert "b.tech" in situation.entities
    assert "phd" in situation.entities
    assert context.target == "phd"


def test_research_interests_do_not_replace_research_target():
    situation = understand_student_situation(
        "My interests are robotics and control systems "
        "and I want to explore research opportunities."
    )
    context = build_decision_context(
        situation
    )

    assert context.target == "research"


def test_msc_query_remains_specific_through_planning():
    question = "What are the admission routes for M.Sc.?"

    situation = understand_student_situation(question)
    context = build_decision_context(situation)
    plan = build_retrieval_plan(
        situation,
        context,
    )

    assert context.target == "m.sc"
    assert "m.sc" in plan.primary_query.lower()
    assert "m.s." not in plan.primary_query.lower()
