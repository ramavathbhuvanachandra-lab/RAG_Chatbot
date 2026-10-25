"""Unit tests for Phase 2 Lossless Semantic Query Contract (LSQC).

Validates that query understanding losslessly preserves:
- Original query text and intent structure
- Degree and program bindings
- Applicant categories (GEN, OBC, SC, ST, EWS)
- Numeric constraints (percentages, CGPA, durations)
- Qualification routes and alternative pathways (AND/OR)
- Negation and temporal/deadline constraints
- Retrieval query mappings
"""
from __future__ import annotations

import pytest
from ai_platform.core.query.lsqc import (
    build_lossless_query,
    extract_applicant_categories,
    extract_condition_relationships,
    extract_degree_and_program,
    extract_numeric_requirements,
    extract_qualifications,
    verify_lossless_preservation,
)
from ai_platform.core.query.models import (
    AtomicIntentBinding,
    LosslessSemanticQuery,
    NumericRequirement,
    SemanticCondition,
)
from ai_platform.core.query.planner import QueryPlan, QueryIntentPlan


class TestLosslessExtractors:
    """Test extractor subroutines for zero loss on critical tokens."""

    def test_extract_applicant_categories(self) -> None:
        text = "Minimum marks for GEN/OBC category is 60% and 55% for SC/ST, with EWS relaxation"
        cats = extract_applicant_categories(text)
        assert "GEN" in cats
        assert "OBC" in cats
        assert "SC" in cats
        assert "ST" in cats
        assert "EWS" in cats

    def test_extract_numeric_requirements(self) -> None:
        text = "Must have a minimum of 60% marks or a minimum CGPA of 6.0 on a scale of 10"
        reqs = extract_numeric_requirements(text)
        values = {r.value for r in reqs}
        assert 60.0 in values
        assert 6.0 in values

    def test_extract_four_year_degree_duration(self) -> None:
        text = "A bachelor's degree of minimum four-year duration in engineering or science"
        reqs = extract_numeric_requirements(text)
        durations = [r for r in reqs if r.name == "degree_duration"]
        assert len(durations) == 1
        assert durations[0].value == 4.0
        assert durations[0].unit == "years"

    def test_extract_qualifications(self) -> None:
        text = "Candidates with a first-class qualification or a qualifying degree with GATE score"
        quals = extract_qualifications(text)
        assert "first-class" in quals
        assert "qualifying degree" in quals
        assert "gate" in quals

    def test_extract_degree_and_program(self) -> None:
        deg, prog = extract_degree_and_program("Is a first-class accepted for M.Sc. admission?")
        assert deg is not None
        assert "M.Sc" in deg or "MSc" in deg

    def test_extract_condition_relationships_or_branching(self) -> None:
        text = "Require 60% marks in aggregate OR a first-class as specified by the university"
        conds = extract_condition_relationships(text)
        assert len(conds) > 0
        # Must reflect OR branching
        or_conds = [c for c in conds if c.logical_relation == "OR"]
        assert len(or_conds) > 0


class TestLosslessQueryConstruction:
    """Test full LosslessSemanticQuery construction from QueryPlan."""

    def test_single_intent_lossless_contract(self) -> None:
        question = "What is the minimum CPI required for OBC candidates in M.Tech?"
        plan = QueryPlan(
            original_query=question,
            resolved_query=question,
            domain_decision="in_scope",
            confidence=0.9,
            is_multi_intent=False,
            preserved_terms=["CPI", "OBC", "M.Tech"],
            intents=[
                QueryIntentPlan(
                    question=question,
                    target="M.Tech",
                    requested_attributes=["eligibility", "cpi"],
                    constraints=["OBC", "CPI"],
                    retrieval_query="M.Tech OBC CPI eligibility",
                )
            ],
        )

        lsq = build_lossless_query(plan, question=question)
        assert isinstance(lsq, LosslessSemanticQuery)
        assert lsq.original_query == question
        assert "OBC" in lsq.applicant_categories
        assert lsq.program_binding == "M.Tech"
        assert len(lsq.intents) == 1
        assert lsq.intents[0].retrieval_query == "M.Tech OBC CPI eligibility"

        verification = verify_lossless_preservation(lsq, question)
        assert verification["passed"] is True

    def test_multi_intent_lossless_contract(self) -> None:
        question = "What is the M.Tech eligibility requirement and what is the application fee?"
        plan = QueryPlan(
            original_query=question,
            resolved_query=question,
            domain_decision="in_scope",
            confidence=0.9,
            is_multi_intent=True,
            preserved_terms=["M.Tech", "eligibility", "application fee"],
            intents=[
                QueryIntentPlan(
                    question="What is the M.Tech eligibility requirement?",
                    target="M.Tech",
                    requested_attributes=["eligibility"],
                    retrieval_query="M.Tech eligibility requirement",
                ),
                QueryIntentPlan(
                    question="What is the application fee?",
                    target="application fee",
                    requested_attributes=["fee"],
                    retrieval_query="application fee",
                ),
            ],
        )

        lsq = build_lossless_query(plan, question=question)
        assert lsq.is_multi_intent is True
        assert len(lsq.intents) == 2
        assert lsq.intents[0].question == "What is the M.Tech eligibility requirement?"
        assert lsq.intents[1].question == "What is the application fee?"
        assert lsq.intents[0].retrieval_query == "M.Tech eligibility requirement"
        assert lsq.intents[1].retrieval_query == "application fee"

    def test_deadline_policy_contract(self) -> None:
        question = "What happens if an admitted student fails to submit the required documents by the deadline?"
        plan = QueryPlan(
            original_query=question,
            resolved_query=question,
            domain_decision="in_scope",
            confidence=0.85,
            is_multi_intent=False,
            preserved_terms=["admitted student", "documents", "deadline"],
            intents=[
                QueryIntentPlan(
                    question=question,
                    requested_attributes=["policy", "deadline"],
                    retrieval_query=question,
                )
            ],
        )

        lsq = build_lossless_query(plan, question=question)
        assert "deadline_policy" in lsq.time_limits


class TestUnderstandQueryNodeLSQCIntegration:
    """Verify that understand_query_node produces lossless_query in state."""

    def test_understand_query_node_includes_lossless_query(self) -> None:
        from ai_platform.core.graph.nodes import CoreNodes, default_dependencies

        nodes = CoreNodes(default_dependencies())
        state = {
            "question": "Can a candidate with 55% marks in SC category apply for M.Sc.?",
            "chat_history": [],
        }
        res = nodes.understand_query_node(state)
        assert "lossless_query" in res
        lsq = res["lossless_query"]
        assert isinstance(lsq, LosslessSemanticQuery)
        assert "SC" in lsq.applicant_categories
        assert 55.0 in {r.value for r in lsq.numeric_constraints}
