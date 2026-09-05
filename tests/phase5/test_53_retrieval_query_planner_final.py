"""
Phase 5.3 — Hard behavioral tests for retrieval input planning.

The tests intentionally use situation-driven, natural-language scenarios
and verify semantic preservation instead of exact query wording.
"""

from dataclasses import dataclass, field

from backend.retrieval_query_planner import (
    build_retrieval_plan,
    retrieval_plan_to_queries,
)


@dataclass(frozen=True)
class Situation:
    raw_text: str
    intent: str
    entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class Context:
    goal: str
    target: str
    facts: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    missing_information: tuple[str, ...] = ()
    confidence: float = 0.9


def make_plan(
    text,
    intent,
    goal,
    target,
    *,
    facts=None,
    preferences=None,
    constraints=None,
):
    return build_retrieval_plan(
        Situation(
            raw_text=text,
            intent=intent,
            entities=(target,),
        ),
        Context(
            goal=goal,
            target=target,
            facts=facts or {},
            preferences=preferences or {},
            constraints=constraints or {},
        ),
    )


# =========================================================
# Personal academic decisions
# =========================================================

def test_phd_case_preserves_target_background_duration_and_score():
    result = make_plan(
        "I have a four-year B.Tech with 74% and want regular Ph.D. admission.",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
        facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (74.0,),
        },
    )

    query = result.primary_query.lower()

    assert "regular ph.d." in query
    assert "eligibility requirements" in query
    assert "bachelor's degree" in query
    assert "4 year degree" in query
    assert "74 percent" in query


def test_regular_phd_context_is_retrieval_specific_even_without_score():
    result = make_plan(
        "Can I apply for a regular Ph.D.?",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
    )

    query = result.primary_query.lower()

    assert "regular ph.d." in query
    assert "eligibility requirements" in query


def test_cgpa_is_preserved_without_inventing_percentage():
    result = make_plan(
        "I have a 7.2/10 CGPA and am considering Ph.D. admission.",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
        facts={
            "cgpa_values": (7.2,),
        },
    )

    query = result.primary_query.lower()

    assert "7.2 cgpa" in query
    assert "70 percent" not in query
    assert "60 percent" not in query


def test_mtech_query_keeps_bachelor_background_separate_from_target():
    result = make_plan(
        "My bachelor's is from another field. Can I apply for M.Tech?",
        "mtech_eligibility",
        "determine_eligibility",
        "m.tech",
        facts={
            "degree": "bachelors_degree",
        },
    )

    query = result.primary_query.lower()

    assert "m.tech" in query
    assert "bachelor's degree" in query


# =========================================================
# Hostel decision problems
# =========================================================

def test_hostel_cost_saving_situation_preserves_all_explicit_constraints():
    result = make_plan(
        "I'll stay 20 days and I'm comfortable sharing a room to save money.",
        "hostel",
        "minimize_cost",
        "hostel",
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
    )

    query = result.primary_query.lower()

    assert "hostel" in query
    assert "cost charges rates" in query
    assert "20 days stay" in query
    assert "double occupancy" in query
    assert "cost-sensitive" in query


def test_hostel_single_occupancy_and_negative_bedding_constraint_survive():
    result = make_plan(
        "I need my own room for 5 days and don't need bedding.",
        "hostel",
        "estimate_cost",
        "hostel",
        constraints={
            "stay_duration": {
                "value": 5.0,
                "unit": "days",
            },
            "occupancy": "single",
            "bedding": "without",
        },
    )

    query = result.primary_query.lower()

    assert "5 days stay" in query
    assert "single occupancy" in query
    assert "without bedding" in query
    assert "with bedding" not in query


def test_hostel_budget_does_not_create_fake_duration():
    result = make_plan(
        "I need hostel accommodation under ₹3000.",
        "hostel",
        "minimize_cost",
        "hostel",
        constraints={
            "budget": 3000.0,
        },
    )

    query = result.primary_query.lower()

    assert "budget 3000" in query
    assert "days stay" not in query


def test_hostel_three_month_duration_is_preserved():
    result = make_plan(
        "I'll probably stay for three months.",
        "hostel",
        "estimate_cost",
        "hostel",
        constraints={
            "stay_duration": {
                "value": 3.0,
                "unit": "months",
            },
        },
    )

    assert "3 months stay" in result.primary_query.lower()


# =========================================================
# Research situations
# =========================================================

def test_research_selection_keeps_department_and_interests():
    result = make_plan(
        "My interest is robotics and control systems in Electrical Engineering.",
        "research",
        "find_relevant_research",
        "electrical engineering",
        preferences={
            "research_interests": (
                "robotics",
                "control systems",
            ),
        },
    )

    query = result.primary_query.lower()

    assert "electrical engineering" in query
    assert "research areas topics" in query
    assert "robotics" in query
    assert "control systems" in query


def test_research_preference_is_not_treated_as_institutional_fact():
    result = make_plan(
        "I prefer research over coursework.",
        "research",
        "find_relevant_research",
        "research",
        preferences={
            "research_oriented": True,
        },
    )

    query = result.primary_query.lower()

    assert "research-oriented" in query
    assert "research-oriented" in result.preserved_signals


def test_visual_computing_interest_survives():
    result = make_plan(
        "I'm interested in visual computing in Electrical Engineering.",
        "research",
        "find_relevant_research",
        "electrical engineering",
        preferences={
            "research_interests": (
                "visual computing",
            ),
        },
    )

    query = result.primary_query.lower()

    assert "visual computing" in query
    assert "electrical engineering" in query
    assert "research areas topics" in query


# =========================================================
# Comparisons
# =========================================================

def test_program_comparison_keeps_comparison_goal_and_preference():
    result = make_plan(
        "I'm deciding between M.Tech and M.S. by Research because I prefer research over coursework.",
        "programs",
        "compare_options",
        "m.tech",
        preferences={
            "research_oriented": True,
        },
        constraints={
            "comparison_requested": True,
        },
    )

    query = result.primary_query.lower()

    assert "comparison requirements differences" in query
    assert "compare options" in query
    assert "research-oriented" in query


def test_hostel_comparison_does_not_become_generic_fee_search():
    result = make_plan(
        "I'm comparing single and double rooms before deciding.",
        "hostel",
        "compare_options",
        "hostel",
        constraints={
            "comparison_requested": True,
        },
    )

    query = result.primary_query.lower()

    assert "hostel" in query
    assert "comparison requirements differences" in query
    assert "compare options" in query


# =========================================================
# Safety / anti-fabrication
# =========================================================

def test_missing_personal_facts_do_not_appear_as_guessed_values():
    result = make_plan(
        "Can I apply for a regular Ph.D.?",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
    )

    query = result.primary_query.lower()

    assert "regular ph.d." in query
    assert "eligibility requirements" in query
    assert "70 percent" not in query
    assert "60 percent" not in query
    assert "7.0 cgpa" not in query
    assert "bachelor's degree" not in query


def test_future_year_is_not_rewritten_into_a_known_policy():
    result = make_plan(
        "I'm planning for 2028 admission.",
        "admission",
        "plan_application",
        "admission",
    )

    query = result.primary_query.lower()

    assert "2028" not in query
    assert "admission application requirements" in query


def test_irrelevant_personal_details_are_excluded():
    result = make_plan(
        "I play football and love music, but my B.Tech is 72% and I want Ph.D. admission.",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
        facts={
            "degree": "bachelors_degree",
            "percentages": (72.0,),
        },
    )

    query = result.primary_query.lower()

    assert "football" not in query
    assert "music" not in query
    assert "72 percent" in query
    assert "bachelor's degree" in query


# =========================================================
# Routing-oriented situations
# =========================================================

def test_emergency_plan_is_small_and_service_oriented():
    result = make_plan(
        "My friend got hurt near the academic block. Where should I take him?",
        "emergency",
        "locate_service",
        "medical_service",
    )

    query = result.primary_query.lower()

    assert "medical_service" in query
    assert "location directions" in query


def test_navigation_plan_is_small_and_service_oriented():
    result = make_plan(
        "I'm new on campus and need directions to the library.",
        "navigation",
        "locate_service",
        "campus_service",
    )

    query = result.primary_query.lower()

    assert "campus_service" in query
    assert "location directions" in query


# =========================================================
# Reusability
# =========================================================

def test_custom_target_requires_no_planner_code_change():
    result = make_plan(
        "I'm considering the School of Quantum Computing.",
        "programs",
        "find_information",
        "School of Quantum Computing",
    )

    assert (
        "school of quantum computing"
        in result.primary_query.lower()
    )


def test_scope_hints_are_semantic_categories():
    result = make_plan(
        "I'm interested in robotics.",
        "research",
        "find_relevant_research",
        "electrical engineering",
        preferences={
            "research_interests": ("robotics",),
        },
    )

    hints = {
        item.casefold()
        for item in result.scope_hints
    }

    assert "research" in hints
    assert "electrical engineering" in hints


# =========================================================
# Retrieval fan-out / stability
# =========================================================

def test_alternate_queries_are_unique():
    result = make_plan(
        "I have a B.Tech with 74% and want Ph.D. admission.",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
        facts={
            "degree": "bachelors_degree",
            "percentages": (74.0,),
        },
    )

    queries = retrieval_plan_to_queries(
        result
    )

    normalized = [
        item.casefold()
        for item in queries
    ]

    assert len(normalized) == len(
        set(normalized)
    )


def test_retrieval_fanout_is_hard_bounded():
    result = make_plan(
        "I have many details and constraints.",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
        facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (74.0, 70.0, 65.0),
            "cgpa_values": (7.4, 7.0),
        },
        preferences={
            "research_interests": (
                "robotics",
                "vlsi",
                "control systems",
            ),
            "research_oriented": True,
            "interdisciplinary": True,
        },
        constraints={
            "stay_duration": {
                "value": 20.0,
                "unit": "days",
            },
            "occupancy": "double",
            "bedding": "without",
            "budget": 3000.0,
            "comparison_requested": True,
        },
    )

    assert len(
        result.alternate_queries
    ) <= 4

    assert len(
        retrieval_plan_to_queries(result)
    ) <= 5


def test_primary_query_is_never_repeated_as_alternate():
    result = make_plan(
        "I need hostel information.",
        "hostel",
        "find_information",
        "hostel",
    )

    assert all(
        item.casefold()
        != result.primary_query.casefold()
        for item in result.alternate_queries
    )


def test_deterministic_for_identical_input():
    args = dict(
        text="I have a four-year B.Tech with 72% and want Ph.D. admission.",
        intent="phd_eligibility",
        goal="determine_eligibility",
        target="phd",
        facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (72.0,),
        },
    )

    first = make_plan(**args)
    second = make_plan(**args)

    assert first == second


def test_preserved_signal_order_is_stable():
    result = make_plan(
        "I have a B.Tech with 72% and prefer research.",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
        facts={
            "degree": "bachelors_degree",
            "percentages": (72.0,),
        },
        preferences={
            "research_oriented": True,
        },
    )

    assert result.preserved_signals[0] == "bachelor's degree"
    assert "72 percent" in result.preserved_signals
    assert "research-oriented" in result.preserved_signals


def test_flattened_queries_preserve_primary_first():
    result = make_plan(
        "I have a B.Tech and want Ph.D.",
        "phd_eligibility",
        "determine_eligibility",
        "phd",
        facts={
            "degree": "bachelors_degree",
        },
    )

    flattened = retrieval_plan_to_queries(
        result
    )

    assert flattened[0] == result.primary_query
    assert flattened[1:] == list(
        result.alternate_queries
    )
