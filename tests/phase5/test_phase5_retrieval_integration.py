"""
Canonical Query Planning Tests

These tests verify the current reusable query-processing contract:

    user question
        ↓
    SemanticQueryFrame
        ↓
    RetrievalQueryPlan
        ↓
    bounded retrieval queries

The legacy StudentSituation / DecisionContext / RetrievalControl
pipeline is intentionally no longer tested here because it has been
removed from the active single-intent graph path.
"""

from dataclasses import replace

from backend.core.query_interpreter import (
    fallback_query_frame,
)

from backend.core.query_frame import (
    QueryFacet,
)

from backend.core.retrieval_query import (
    MAX_RETRIEVAL_QUERIES,
    build_retrieval_query_plan,
)


# ============================================================
# Test helpers
# ============================================================


def make_frame(
    question: str,
    *,
    target: str | None = None,
    request_type: str | None = None,
    facets: tuple[QueryFacet, ...] = (),
    qualifiers: tuple[str, ...] = (),
    conditions: tuple[str, ...] = (),
    relations: tuple[str, ...] = (),
    temporal_context: tuple[str, ...] = (),
    comparison_targets: tuple[str, ...] = (),
    preserved_terms: tuple[str, ...] = (),
    confidence: float = 0.90,
):
    """
    Build a valid SemanticQueryFrame starting from the canonical
    safe fallback frame.

    This keeps the test focused on retrieval planning rather than
    duplicating the frame constructor contract.
    """

    frame = fallback_query_frame(
        question
    )

    return replace(
        frame,
        target=target,
        request_type=request_type,
        facets=facets,
        qualifiers=qualifiers,
        conditions=conditions,
        relations=relations,
        temporal_context=temporal_context,
        comparison_targets=comparison_targets,
        preserved_terms=preserved_terms,
        confidence=confidence,
    )


# ============================================================
# Original question authority
# ============================================================


def test_original_question_is_always_primary():
    question = (
        "What are the admission routes for M.Sc.?"
    )

    frame = make_frame(
        question,
        target="M.Sc.",
        request_type="admission routes",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="admission routes",
                required=True,
                importance=1.0,
            ),
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    assert plan.original_query == question

    assert plan.primary_query == question

    assert plan.queries[0] == question


# ============================================================
# Entity preservation
# ============================================================


def test_exact_program_identity_is_preserved():
    question = (
        "What are the admission routes for M.Sc.?"
    )

    frame = make_frame(
        question,
        target="M.Sc.",
        request_type="admission routes",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="admission routes",
                required=True,
                importance=1.0,
            ),
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    assert "M.Sc." in plan.structured_query

    # Similar program names must not be introduced by the planner.
    assert "M.S." not in plan.structured_query


def test_msc_and_ms_remain_distinct():
    msc_question = (
        "What are the admission routes for M.Sc.?"
    )

    ms_question = (
        "What are the admission routes for M.S.?"
    )

    msc_frame = make_frame(
        msc_question,
        target="M.Sc.",
        request_type="admission routes",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="admission routes",
                required=True,
                importance=1.0,
            ),
        ),
    )

    ms_frame = make_frame(
        ms_question,
        target="M.S.",
        request_type="admission routes",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="admission routes",
                required=True,
                importance=1.0,
            ),
        ),
    )

    msc_plan = build_retrieval_query_plan(
        msc_frame
    )

    ms_plan = build_retrieval_query_plan(
        ms_frame
    )

    assert "M.Sc." in msc_plan.structured_query
    assert "M.S." not in msc_plan.structured_query

    assert "M.S." in ms_plan.structured_query
    assert "M.Sc." not in ms_plan.structured_query


# ============================================================
# Structured retrieval signal
# ============================================================


def test_structured_query_contains_information_need():
    question = (
        "What is the admission process for M.Tech?"
    )

    frame = make_frame(
        question,
        target="M.Tech",
        request_type="admission process",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="admission process",
                required=True,
                importance=1.0,
            ),
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    structured = (
        plan.structured_query.lower()
    )

    assert "m.tech" in structured
    assert "admission process" in structured


def test_conditions_are_preserved_in_structured_query():
    question = (
        "What are the admission requirements for Ph.D. "
        "with a four-year bachelor's degree?"
    )

    frame = make_frame(
        question,
        target="Ph.D.",
        request_type="admission requirements",
        conditions=(
            "four-year bachelor's degree",
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    structured = (
        plan.structured_query.lower()
    )

    assert "ph.d." in structured
    assert "admission requirements" in structured
    assert "four-year bachelor's degree" in structured


# ============================================================
# Low-confidence safety
# ============================================================


def test_low_confidence_uses_original_question_only():
    question = (
        "I'm not really sure what I need."
    )

    frame = make_frame(
        question,
        target="college eligibility requirements",
        request_type="eligibility",
        confidence=0.20,
    )

    plan = build_retrieval_query_plan(
        frame
    )

    assert plan.queries == (
        question,
    )

    assert plan.alternate_queries == ()

    assert plan.mode == (
        "original_only"
    )


# ============================================================
# No policy conclusion during planning
# ============================================================


def test_retrieval_planning_does_not_answer_policy_question():
    question = (
        "I have 74% and want to apply for regular Ph.D. admission."
    )

    frame = make_frame(
        question,
        target="Ph.D.",
        request_type="admission eligibility",
        conditions=(
            "74%",
            "regular admission",
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    blob = " ".join(
        plan.queries
    ).lower()

    # Planning should describe what to retrieve, not decide the result.
    assert "eligible" not in blob
    assert "not eligible" not in blob

    assert "ph.d." in blob
    assert "74%" in blob


# ============================================================
# Irrelevant information protection
# ============================================================


def test_only_interpreted_information_reaches_structured_query():
    question = (
        "I play football every weekend and love music, "
        "but I want information about Ph.D. admission."
    )

    frame = make_frame(
        question,
        target="Ph.D.",
        request_type="admission",
        preserved_terms=(
            "Ph.D.",
            "admission",
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    structured = (
        plan.structured_query.lower()
    )

    assert "ph.d." in structured
    assert "admission" in structured

    assert "football" not in structured
    assert "music" not in structured


# ============================================================
# Bounded fan-out
# ============================================================


def test_retrieval_query_count_is_bounded():
    question = (
        "I have a B.Tech with 74%, prefer research, "
        "need hostel accommodation for 20 days, "
        "can share a room, and want to minimize cost "
        "while applying for regular Ph.D."
    )

    frame = make_frame(
        question,
        target="Ph.D.",
        request_type="admission",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="admission",
                required=True,
                importance=1.0,
            ),
            QueryFacet(
                name="degree",
                value="B.Tech",
                required=True,
                importance=0.8,
            ),
        ),
        conditions=(
            "74%",
            "regular admission",
        ),
        preserved_terms=(
            "research",
            "hostel",
            "20 days",
            "double occupancy",
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    assert len(
        plan.queries
    ) <= MAX_RETRIEVAL_QUERIES

    assert len(
        plan.alternate_queries
    ) <= MAX_RETRIEVAL_QUERIES - 1


# ============================================================
# Determinism
# ============================================================


def test_retrieval_plan_is_deterministic():
    question = (
        "What are the admission routes for M.Sc.?"
    )

    frame = make_frame(
        question,
        target="M.Sc.",
        request_type="admission routes",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="admission routes",
                required=True,
                importance=1.0,
            ),
        ),
    )

    first = build_retrieval_query_plan(
        frame
    )

    second = build_retrieval_query_plan(
        frame
    )

    assert first == second


# ============================================================
# Query ordering
# ============================================================


def test_alternate_queries_are_only_supplementary():
    question = (
        "How can I apply for M.Tech?"
    )

    frame = make_frame(
        question,
        target="M.Tech",
        request_type="application procedure",
        facets=(
            QueryFacet(
                name="requested_attribute",
                value="application procedure",
                required=True,
                importance=1.0,
            ),
        ),
    )

    plan = build_retrieval_query_plan(
        frame
    )

    assert plan.primary_query == (
        plan.queries[0]
    )

    for alternate in plan.alternate_queries:
        assert alternate != question