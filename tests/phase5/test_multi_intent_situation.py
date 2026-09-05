"""
Phase 5.7 — Multi-Intent + Situation Tests

These tests verify that personal situation information is attached only
to the independent request for which it is relevant.
"""

from types import SimpleNamespace

from backend.multi_intent_situation import (
    build_intent_situation,
    build_multi_intent_situations,
    format_multi_intent_situation_context,
    situation_to_dict,
)


def phd_situation():
    return SimpleNamespace(
        intent="phd_eligibility",
        goal="determine_eligibility",
        entities=("phd", "b.tech"),
        user_facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (72.0,),
        },
        constraints={},
        preferences={},
    )


def phd_decision():
    return SimpleNamespace(
        goal="determine_eligibility",
        target="phd",
        facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (72.0,),
        },
        preferences={},
        constraints={},
        missing_information=(),
    )


def test_phd_intent_receives_phd_situation():
    result = build_intent_situation(
        intent={
            "question": (
                "Can I apply for Ph.D. admission?"
            ),
            "topics": [
                "eligibility",
            ],
            "entities": [
                "phd",
            ],
        },
        situation=phd_situation(),
        decision_context=phd_decision(),
    )

    assert result.applies is True
    assert result.target == "phd"
    assert result.facts["degree"] == "bachelors_degree"
    assert 72.0 in result.facts["percentages"]


def test_library_intent_does_not_receive_phd_situation():
    result = build_intent_situation(
        intent={
            "question": (
                "Where is the library?"
            ),
            "topics": [
                "navigation",
            ],
            "entities": [
                "library",
            ],
        },
        situation=phd_situation(),
        decision_context=phd_decision(),
    )

    assert result.applies is False
    assert result.facts == {}
    assert result.preferences == {}
    assert result.constraints == {}


def test_hostel_intent_receives_hostel_situation():
    situation = SimpleNamespace(
        intent="hostel",
        goal="minimize_cost",
        entities=("hostel",),
        user_facts={},
        constraints={
            "stay_duration": {
                "value": 20.0,
                "unit": "days",
            },
            "occupancy": "double",
        },
        preferences={
            "cost_sensitive": True,
        },
    )

    decision = SimpleNamespace(
        goal="minimize_cost",
        target="hostel",
        facts={},
        preferences={
            "cost_sensitive": True,
        },
        constraints={
            "stay_duration": {
                "value": 20.0,
                "unit": "days",
            },
            "occupancy": "double",
        },
        missing_information=(),
    )

    result = build_intent_situation(
        intent={
            "question": (
                "What is the cheapest hostel option?"
            ),
            "topics": [
                "fees",
            ],
            "entities": [
                "hostel",
            ],
        },
        situation=situation,
        decision_context=decision,
    )

    assert result.applies is True
    assert result.target == "hostel"
    assert result.constraints["occupancy"] == "double"
    assert result.preferences["cost_sensitive"] is True


def test_mess_intent_does_not_receive_hostel_constraints():
    situation = SimpleNamespace(
        intent="hostel",
        goal="minimize_cost",
        entities=("hostel",),
        user_facts={},
        constraints={
            "stay_duration": {
                "value": 20.0,
                "unit": "days",
            },
            "occupancy": "double",
        },
        preferences={
            "cost_sensitive": True,
        },
    )

    decision = SimpleNamespace(
        goal="minimize_cost",
        target="hostel",
        facts={},
        preferences={
            "cost_sensitive": True,
        },
        constraints={
            "stay_duration": {
                "value": 20.0,
                "unit": "days",
            },
        },
        missing_information=(),
    )

    result = build_intent_situation(
        intent={
            "question": (
                "When does the mess open?"
            ),
            "topics": [
                "facilities",
            ],
            "entities": [
                "mess",
            ],
        },
        situation=situation,
        decision_context=decision,
    )

    assert result.applies is False
    assert result.constraints == {}


def test_two_intents_get_independent_situation_scope():
    situation = phd_situation()
    decision = phd_decision()

    results = build_multi_intent_situations(
        intent_units=[
            {
                "question": (
                    "Can I apply for Ph.D. admission?"
                ),
                "topics": [
                    "eligibility",
                ],
                "entities": [
                    "phd",
                ],
            },
            {
                "question": (
                    "Where is the library?"
                ),
                "topics": [
                    "navigation",
                ],
                "entities": [
                    "library",
                ],
            },
        ],
        situation=situation,
        decision_context=decision,
    )

    assert len(results) == 2
    assert results[0].applies is True
    assert results[1].applies is False


def test_three_intents_only_matching_intent_gets_situation():
    situation = phd_situation()
    decision = phd_decision()

    results = build_multi_intent_situations(
        intent_units=[
            {
                "question": "Can I apply for Ph.D.?",
                "topics": ["eligibility"],
                "entities": ["phd"],
            },
            {
                "question": "Where is the library?",
                "topics": ["navigation"],
                "entities": ["library"],
            },
            {
                "question": "What is the mess timing?",
                "topics": ["facilities"],
                "entities": ["mess"],
            },
        ],
        situation=situation,
        decision_context=decision,
    )

    assert [
        result.applies
        for result in results
    ] == [
        True,
        False,
        False,
    ]


def test_situation_serialization_is_json_like():
    situation = phd_situation()
    decision = phd_decision()

    scoped = build_intent_situation(
        intent={
            "question": "Can I apply for Ph.D.?",
            "topics": ["eligibility"],
            "entities": ["phd"],
        },
        situation=situation,
        decision_context=decision,
    )

    payload = situation_to_dict(
        scoped
    )

    assert isinstance(
        payload,
        dict,
    )

    assert payload["applies"] is True
    assert payload["target"] == "phd"
    assert isinstance(
        payload["facts"],
        dict,
    )


def test_unrelated_intent_produces_no_situation_context():
    result = format_multi_intent_situation_context(
        [
            {
                "question": "Where is the library?",
                "situation_context": {
                    "applies": False,
                },
            },
        ]
    )

    assert result == ""


def test_only_relevant_intent_is_formatted():
    result = format_multi_intent_situation_context(
        [
            {
                "question": "Can I apply for Ph.D.?",
                "situation_context": {
                    "applies": True,
                    "target": "phd",
                    "goal": "determine_eligibility",
                    "facts": {
                        "degree": "bachelors_degree",
                        "percentages": (72.0,),
                    },
                    "preferences": {},
                    "constraints": {},
                },
            },
            {
                "question": "Where is the library?",
                "situation_context": {
                    "applies": False,
                },
            },
        ]
    )

    assert "Intent 1" in result
    assert "72.0" in result
    assert "Intent 2" not in result


def test_situation_cannot_leak_to_unrelated_intent():
    result = format_multi_intent_situation_context(
        [
            {
                "question": "Can I apply for Ph.D.?",
                "situation_context": {
                    "applies": True,
                    "target": "phd",
                    "goal": "determine_eligibility",
                    "facts": {
                        "percentages": (72.0,),
                    },
                    "preferences": {},
                    "constraints": {},
                },
            },
            {
                "question": "Where is the library?",
                "situation_context": {
                    "applies": False,
                    "facts": {},
                },
            },
        ]
    )

    # The student's percentage appears only in Intent 1 context.
    assert result.count(
        "72.0"
    ) == 1


def test_empty_situation_is_safe():
    result = build_intent_situation(
        intent={
            "question": "Where is the library?",
            "topics": ["navigation"],
            "entities": ["library"],
        },
        situation=None,
        decision_context=None,
    )

    assert result.applies is False


def test_explicit_target_is_stronger_than_topic():
    situation = SimpleNamespace(
        intent="research",
        goal="find_relevant_research",
        entities=("electrical engineering",),
        user_facts={},
        constraints={},
        preferences={},
    )

    decision = SimpleNamespace(
        target="electrical engineering",
        goal="find_relevant_research",
        facts={},
        preferences={},
        constraints={},
        missing_information=(),
    )

    result = build_intent_situation(
        intent={
            "question": (
                "What research areas are available "
                "in Electrical Engineering?"
            ),
            "topics": [
                "research",
            ],
            "entities": [
                "electrical engineering",
            ],
        },
        situation=situation,
        decision_context=decision,
    )

    assert result.applies is True


def test_navigation_does_not_inherit_academic_percentage():
    situation = phd_situation()
    decision = phd_decision()

    result = build_intent_situation(
        intent={
            "question": (
                "Where is the hostel office?"
            ),
            "topics": [
                "navigation",
            ],
            "entities": [
                "hostel office",
            ],
        },
        situation=situation,
        decision_context=decision,
    )

    assert result.applies is False
