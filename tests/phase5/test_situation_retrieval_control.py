from dataclasses import dataclass, field

from backend.situation_retrieval_control import (
    build_retrieval_control,
    retrieval_control_to_queries,
)


@dataclass(frozen=True)
class Context:
    confidence: float = 0.9
    facts: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)


# ---------------------------------------------------------
# High-value real-world cases
# ---------------------------------------------------------

def test_phd_eligibility_uses_structured_query():
    result = build_retrieval_control(
        original_query=(
            "I have a B.Tech with 74% and want regular Ph.D. admission."
        ),
        primary_query=(
            "regular Ph.D. eligibility requirements "
            "bachelor's degree 74 percent"
        ),
        alternate_queries=(
            "Ph.D. admission eligibility bachelor's degree",
            "regular Ph.D. requirements four year degree",
        ),
        context=Context(
            confidence=0.94,
            facts={
                "degree": "bachelors_degree",
                "percentages": (74.0,),
            },
        ),
    )

    assert result.mode == "situational"
    assert result.original_preserved is True
    assert result.queries[0].startswith("regular Ph.D.")
    assert result.original_query.startswith("I have a B.Tech")
    assert result.original_query not in result.queries


def test_irrelevant_student_story_is_not_sent_to_retriever():
    result = build_retrieval_control(
        original_query=(
            "I play football and love music, but my B.Tech is 72% "
            "and I want Ph.D. admission."
        ),
        primary_query=(
            "Ph.D. eligibility requirements bachelor's degree 72 percent"
        ),
        context=Context(
            confidence=0.93,
            facts={
                "degree": "bachelors_degree",
                "percentages": (72.0,),
            },
        ),
    )

    blob = " ".join(result.queries).lower()

    assert "football" not in blob
    assert "music" not in blob
    assert "72 percent" in blob
    assert "bachelor's degree" in blob


def test_low_confidence_reverts_to_original_only():
    original = "I'm not sure what I need."

    result = build_retrieval_control(
        original_query=original,
        primary_query="eligibility requirements",
        alternate_queries=(
            "degree admission requirements",
            "doctoral eligibility",
        ),
        context=Context(
            confidence=0.20,
            facts={
                "degree": "bachelors_degree",
            },
        ),
    )

    assert result.mode == "conservative"
    assert result.queries == (original,)
    assert result.max_queries == 1


def test_low_confidence_without_original_uses_primary_fallback():
    result = build_retrieval_control(
        original_query="",
        primary_query="bachelor's degree eligibility",
        context=Context(
            confidence=0.30,
        ),
    )

    assert result.queries == (
        "bachelor's degree eligibility",
    )


def test_plain_query_does_not_fan_out():
    result = build_retrieval_control(
        original_query="Which B.Tech programs are available?",
        primary_query="B.Tech programs",
        alternate_queries=(
            "undergraduate B.Tech programs",
            "academic degree programs",
        ),
        context=Context(
            confidence=0.90,
        ),
    )

    assert result.mode == "focused"
    assert result.queries == (
        "B.Tech programs",
    )


def test_hostel_decision_keeps_duration_occupancy_and_cost():
    result = build_retrieval_control(
        original_query=(
            "I'll stay 20 days and I'm okay sharing a room to save money."
        ),
        primary_query=(
            "hostel cost charges rates 20 days stay "
            "double occupancy cost-sensitive"
        ),
        alternate_queries=(
            "double occupancy hostel charges",
            "long-term hostel accommodation charges",
            "single occupancy hostel charges",
        ),
        context=Context(
            confidence=0.95,
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
        ),
    )

    assert result.mode == "situational"
    assert len(result.queries) <= 3

    blob = " ".join(result.queries).lower()

    assert "20 days stay" in blob
    assert "double occupancy" in blob


def test_research_decision_keeps_interest_and_department():
    result = build_retrieval_control(
        original_query=(
            "I'm interested in robotics and control systems in "
            "Electrical Engineering."
        ),
        primary_query=(
            "Electrical Engineering research areas topics "
            "robotics control systems"
        ),
        alternate_queries=(
            "Electrical Engineering robotics research",
            "Electrical Engineering control systems research",
        ),
        context=Context(
            confidence=0.95,
            preferences={
                "research_interests": (
                    "robotics",
                    "control systems",
                ),
            },
        ),
    )

    blob = " ".join(result.queries).lower()

    assert "electrical engineering" in blob
    assert "robotics" in blob
    assert "control systems" in blob


def test_negative_bedding_constraint_is_not_reversed():
    result = build_retrieval_control(
        original_query=(
            "I need my own room and don't need bedding."
        ),
        primary_query=(
            "hostel room single occupancy without bedding"
        ),
        context=Context(
            confidence=0.92,
            constraints={
                "occupancy": "single",
                "bedding": "without",
            },
        ),
    )

    blob = " ".join(result.queries).lower()

    assert "without bedding" in blob
    assert "with bedding" not in blob


def test_future_year_is_not_invented_as_policy():
    result = build_retrieval_control(
        original_query=(
            "I'm planning for 2028 admission."
        ),
        primary_query=(
            "admission application requirements"
        ),
        context=Context(
            confidence=0.90,
            facts={
                "degree": "bachelors_degree",
            },
        ),
    )

    blob = " ".join(result.queries).lower()

    assert "2028" not in blob


def test_no_new_query_is_invented_by_control_layer():
    original = "I need hostel information."
    primary = "hostel accommodation information"
    alternates = (
        "hostel room rent",
        "hostel accommodation charges",
    )

    result = build_retrieval_control(
        original_query=original,
        primary_query=primary,
        alternate_queries=alternates,
        context=Context(
            confidence=0.92,
            constraints={
                "occupancy": "double",
            },
        ),
    )

    allowed = {
        primary,
        *alternates,
    }

    assert set(result.queries).issubset(
        allowed
    )


def test_duplicate_queries_are_removed():
    result = build_retrieval_control(
        original_query="question",
        primary_query="structured query",
        alternate_queries=(
            "structured query",
            "STRUCTURED QUERY",
            "alternate",
        ),
        context=Context(
            confidence=0.92,
            facts={
                "degree": "bachelors_degree",
            },
        ),
    )

    normalized = [
        query.casefold()
        for query in result.queries
    ]

    assert len(normalized) == len(
        set(normalized)
    )


def test_query_fanout_is_hard_bounded():
    result = build_retrieval_control(
        original_query="complex question",
        primary_query="primary query",
        alternate_queries=tuple(
            f"alternate {i}"
            for i in range(100)
        ),
        context=Context(
            confidence=0.99,
            facts={
                "degree": "bachelors_degree",
            },
        ),
    )

    assert len(result.queries) <= 3
    assert result.max_queries == 3


def test_original_is_retained_for_audit_even_when_not_searched():
    original = (
        "Honestly I am confused and I have been preparing for months."
    )

    result = build_retrieval_control(
        original_query=original,
        primary_query="admission requirements",
        context=Context(
            confidence=0.91,
            facts={
                "degree": "bachelors_degree",
            },
        ),
    )

    assert result.original_query == original
    assert result.original_preserved is True
    assert original not in result.queries


def test_flattened_queries_match_plan():
    result = build_retrieval_control(
        original_query="student question",
        primary_query="structured query",
        alternate_queries=("alternate",),
        context=Context(
            confidence=0.90,
            facts={
                "degree": "bachelors_degree",
            },
        ),
    )

    assert retrieval_control_to_queries(result) == list(
        result.queries
    )


def test_empty_inputs_are_safe():
    result = build_retrieval_control(
        original_query="",
        primary_query="",
        context=Context(
            confidence=0.90,
        ),
    )

    assert result.queries == ()


def test_confidence_is_clamped():
    high = build_retrieval_control(
        original_query="q",
        primary_query="p",
        context=Context(
            confidence=9,
        ),
    )

    low = build_retrieval_control(
        original_query="q",
        primary_query="p",
        context=Context(
            confidence=-5,
        ),
    )

    assert high.confidence == 1.0
    assert low.confidence == 0.0
