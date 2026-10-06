from ai_platform.core.query.planner import (
    QueryIntentPlan,
    QueryPlan,
    should_stop_pipeline,
    intent_units_from_plan,
    plan_to_frame,
    plan_to_query,
    retrieval_queries_from_plan,
)


def test_out_of_scope_stops_only_when_confident():
    plan = QueryPlan(
        original_query="Who won yesterday's cricket match?",
        resolved_query="Who won yesterday's cricket match?",
        domain_decision="out_of_scope",
        domain_confidence=0.95,
        domain_reason="unrelated_sports_request",
        confidence=0.95,
    )

    assert should_stop_pipeline(plan) is True


def test_low_confidence_out_of_scope_does_not_stop():
    plan = QueryPlan(
        original_query="What is this?",
        resolved_query="What is this?",
        domain_decision="out_of_scope",
        domain_confidence=0.60,
        confidence=0.60,
    )

    assert should_stop_pipeline(plan) is False


def test_in_scope_plan_produces_query_contract():
    plan = QueryPlan(
        original_query=(
            "What is the minimum percentage required for M.Tech admission?"
        ),
        resolved_query=(
            "What is the minimum percentage required for M.Tech admission?"
        ),
        language="English",
        domain_decision="in_scope",
        domain_confidence=0.97,
        confidence=0.94,
        intents=[
            QueryIntentPlan(
                question=(
                    "What is the minimum percentage required "
                    "for M.Tech admission?"
                ),
                request_type="eligibility",
                target="M.Tech",
                requested_attributes=["minimum percentage"],
                retrieval_query=(
                    "M.Tech eligibility minimum percentage admission"
                ),
                confidence=0.95,
            )
        ],
    )

    query = plan_to_query(plan)
    frame = plan_to_frame(plan)

    assert query.original_query == plan.original_query
    assert query.target is not None
    assert query.target.text == "M.Tech"

    assert frame.target == "M.Tech"
    assert frame.is_multi_part is False


def test_multi_intent_produces_multiple_independent_units():
    plan = QueryPlan(
        original_query=(
            "What is M.Tech eligibility, what documents are needed, "
            "and what is the application fee?"
        ),
        resolved_query=(
            "What is M.Tech eligibility, what documents are needed, "
            "and what is the application fee?"
        ),
        language="English",
        domain_decision="in_scope",
        domain_confidence=0.97,
        is_multi_intent=True,
        intents=[
            QueryIntentPlan(
                question="What is M.Tech eligibility?",
                request_type="eligibility",
                target="M.Tech",
                requested_attributes=["eligibility"],
                retrieval_query="M.Tech eligibility",
                confidence=0.95,
            ),
            QueryIntentPlan(
                question="What documents are needed?",
                request_type="procedure",
                target="M.Tech",
                requested_attributes=["documents"],
                retrieval_query="M.Tech required documents",
                confidence=0.92,
            ),
            QueryIntentPlan(
                question="What is the application fee?",
                request_type="cost",
                target="M.Tech",
                requested_attributes=["application fee"],
                retrieval_query="M.Tech application fee",
                confidence=0.94,
            ),
        ],
        confidence=0.94,
    )

    units = intent_units_from_plan(plan)

    assert len(units) == 3
    assert all(unit["query"] is not None for unit in units)
    assert all(unit["query_frame"] is not None for unit in units)


def test_original_query_always_stays_first_retrieval_input():
    plan = QueryPlan(
        original_query="How do I apply for M.Tech?",
        resolved_query="How do I apply for M.Tech?",
        domain_decision="in_scope",
        domain_confidence=0.95,
        intents=[
            QueryIntentPlan(
                question="How do I apply for M.Tech?",
                request_type="procedure",
                target="M.Tech",
                requested_attributes=["application process"],
                retrieval_query="M.Tech application process",
            )
        ],
    )

    queries = retrieval_queries_from_plan(plan)

    assert queries[0] == "How do I apply for M.Tech?"
    assert "M.Tech application process" in queries


def test_out_of_scope_has_no_synthetic_institutional_answer():
    plan = QueryPlan(
        original_query="Tell me a joke.",
        resolved_query="Tell me a joke.",
        domain_decision="out_of_scope",
        domain_confidence=0.98,
        domain_reason="unrelated_casual_request",
    )

    assert should_stop_pipeline(plan) is True
    assert plan.intents == []
