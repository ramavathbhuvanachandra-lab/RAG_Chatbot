"""Targeted regression tests and counter-tests for query understanding and planner redesign.

Covers the 7 residual failure scenarios from the October 15, 2026 planner benchmark:
1. scen_04_story_exec_mtech_working_engineer (compound single-program eligibility consolidation)
2. scen_05_story_provisional_admission_delay (Senate cancellation policy, no naked noun queries)
3. scen_06_story_phd_direct_from_btech_sc (relaxed category threshold + individual score assessment)
4. scen_10_multi_msc_jam_and_phd_duration (M.Sc. vs Ph.D. degree target binding isolation)
5. scen_11_multi_hostel_charges_and_anti_ragging (facility fee vs statutory committee splitting)
6. scen_12_multi_part_time_mtech_exp_and_phd_assistantship (M.Tech experience vs Ph.D. assistantship)
7. scen_15_probation_max_credits (probation credit limit, maximum comparator, preserved terms)

Also includes counter-tests verifying that fixes do not introduce inverse regressions.
"""
from __future__ import annotations

import re
import pytest

from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.lsqc import build_lossless_query
from ai_platform.core.query.planner import (
    build_query_plan,
    plan_to_frame,
    retrieval_queries_from_plan,
    should_stop_pipeline,
)


# ============================================================================
# PART 1: THE 7 FAILING SCENARIOS REGRESSION TESTS
# ============================================================================

def test_scen_04_story_exec_mtech_working_engineer() -> None:
    """scen_04: Compound single-program eligibility question must remain single-intent."""
    q = (
        "I completed my four-year B.Tech in Electrical Engineering three years ago and I have been "
        "working in an automotive R&D lab since graduation. I want to pursue the Executive M.Tech "
        "program at IIT Jodhpur without quitting my job. Do I meet the minimum work experience "
        "requirement to register, and is an entrance test or interview mandatory?"
    )
    plan = build_query_plan(q)
    lossless = build_lossless_query(plan, question=q)
    retrieval_queries = retrieval_queries_from_plan(plan, limit=4)

    assert plan.domain_decision == "in_scope"
    assert should_stop_pipeline(plan) is False
    # Single eligibility verification objective for Executive M.Tech
    assert plan.is_multi_intent is False
    assert len(plan.intents) == 1

    target = plan.intents[0].target or (lossless.program_binding or lossless.degree_binding)
    assert target is not None
    assert "M.Tech" in target or "MTech" in target

    # Numeric constraints preserved
    values = {nc.value for nc in lossless.numeric_constraints}
    assert 4.0 in values  # 4-year degree
    assert 3.0 in values  # 3 years experience

    # Preserved terms
    preserved = " ".join(plan.preserved_terms).lower() + " " + plan.resolved_query.lower()
    assert "m.tech" in preserved or "mtech" in preserved
    assert "work experience" in preserved or "experience" in preserved
    assert "interview" in preserved

    # Forbidden naked retrieval queries
    for rq in retrieval_queries:
        assert rq.strip().lower() not in {"without", "m.tech work experience without", "m.tech eligibility without"}


def test_scen_05_story_provisional_admission_delay() -> None:
    """scen_05: Provisional admission policy / Senate cancellation authority must stay single-intent."""
    q = (
        "A student who was admitted provisionally in the autumn semester missed the academic calendar deadline "
        "to submit their final degree certificates and marks sheets because their previous university delayed "
        "graduation results. Can the Senate cancel their admission solely for failing to submit these documents "
        "by the specified deadline, and does the Senate retain authority to cancel admission later if documents "
        "contain false information?"
    )
    plan = build_query_plan(q)
    lossless = build_lossless_query(plan, question=q)
    retrieval_queries = retrieval_queries_from_plan(plan, limit=4)

    assert plan.domain_decision == "in_scope"
    assert should_stop_pipeline(plan) is False
    # Unified administrative / governance policy inquiry
    assert plan.is_multi_intent is False
    assert len(plan.intents) == 1

    # Preserved terms
    preserved = " ".join(plan.preserved_terms).lower() + " " + plan.resolved_query.lower()
    assert "senate" in preserved
    assert "admission" in preserved
    assert "provisional" in preserved

    # Forbidden naked noun retrieval queries
    for rq in retrieval_queries:
        assert rq.strip().lower() not in {"documents", "deadline", "without"}


def test_scen_06_story_phd_direct_from_btech_sc() -> None:
    """scen_06: Ph.D. SC category relaxed cutoff + 66% score check is a unified objective."""
    q = (
        "I am an SC category student with a 4-year B.Tech degree holding a 66% aggregate score. I want "
        "to apply directly for regular Ph.D. admission at IIT Jodhpur without having a Master's degree. "
        "Given the 70% requirement for General candidates with a 4-year bachelor's degree, what is the "
        "relaxed percentage cutoff for SC/ST candidates, and does my 66% score satisfy that minimum threshold?"
    )
    plan = build_query_plan(q)
    lossless = build_lossless_query(plan, question=q)
    retrieval_queries = retrieval_queries_from_plan(plan, limit=4)

    assert plan.domain_decision == "in_scope"
    assert should_stop_pipeline(plan) is False
    assert plan.is_multi_intent is False
    assert len(plan.intents) == 1

    target = plan.intents[0].target or (lossless.program_binding or lossless.degree_binding)
    assert target is not None
    assert "Ph.D" in target or "PhD" in target

    # All three categories extracted
    cats = {c.upper() for c in lossless.applicant_categories}
    assert "SC" in cats
    assert "ST" in cats
    assert ("GEN" in cats or "GENERAL" in cats)

    # Numerical constraints: 4-year degree, 66%, 70%
    values = {nc.value for nc in lossless.numeric_constraints}
    assert 66.0 in values
    assert 70.0 in values
    assert 4.0 in values

    # Forbidden queries
    for rq in retrieval_queries:
        assert rq.strip().lower() not in {"without", "phd minimum percentage sc st without"}


def test_scen_10_multi_msc_jam_and_phd_duration() -> None:
    """scen_10: Multi-intent across distinct degrees must bind M.Sc. to intent 0, Ph.D. to intent 1."""
    q = (
        "What is the qualifying degree requirement for M.Sc. and what degree duration is required "
        "for Ph.D. applicants holding a bachelor's degree?"
    )
    plan = build_query_plan(q)
    lossless = build_lossless_query(plan, question=q)

    assert plan.domain_decision == "in_scope"
    assert should_stop_pipeline(plan) is False
    assert plan.is_multi_intent is True
    assert len(plan.intents) >= 2

    # Intent 0 must be bound to M.Sc.
    intent_0 = plan.intents[0]
    i0_target = intent_0.target or (lossless.intents[0].target or lossless.intents[0].degree)
    assert i0_target is not None
    assert "M.Sc" in i0_target or "MSc" in i0_target
    assert "Ph.D" not in i0_target and "PhD" not in i0_target

    # Intent 1 must be bound to Ph.D.
    intent_1 = plan.intents[1]
    i1_target = intent_1.target or (lossless.intents[1].target or lossless.intents[1].degree)
    assert i1_target is not None
    assert "Ph.D" in i1_target or "PhD" in i1_target


def test_scen_11_multi_hostel_charges_and_anti_ragging() -> None:
    """scen_11: Hostel charges and Anti-Ragging Committee chairperson must be split into 2 intents."""
    q = (
        "What are the short-term hostel room charges per day for students, and who serves as the "
        "chairperson of the Institute Anti-Ragging Committee?"
    )
    plan = build_query_plan(q)
    lossless = build_lossless_query(plan, question=q)

    assert plan.domain_decision == "in_scope"
    assert should_stop_pipeline(plan) is False
    assert plan.is_multi_intent is True
    assert len(plan.intents) >= 2

    # Preserved terms
    preserved = " ".join(plan.preserved_terms).lower() + " " + plan.resolved_query.lower()
    assert "hostel" in preserved
    assert "anti-ragging" in preserved or "anti ragging" in preserved

    # Target checks
    t0 = (plan.intents[0].target or "").lower()
    t1 = (plan.intents[1].target or "").lower()
    assert "hostel" in t0 or "fee" in t0 or "charge" in t0
    assert "anti-ragging" in t1 or "ragging" in t1 or "committee" in t1


def test_scen_12_multi_part_time_mtech_exp_and_phd_assistantship() -> None:
    """scen_12: Executive M.Tech experience and Ph.D. assistantship must split with correct degree bindings."""
    q = (
        "How many years of work experience are required for Executive M.Tech admission, and what "
        "assistantship is available for regular full-time Ph.D. scholars?"
    )
    plan = build_query_plan(q)
    lossless = build_lossless_query(plan, question=q)

    assert plan.domain_decision == "in_scope"
    assert should_stop_pipeline(plan) is False
    assert plan.is_multi_intent is True
    assert len(plan.intents) >= 2

    # Intent 0 bound to M.Tech
    t0 = plan.intents[0].target or (lossless.intents[0].target or lossless.intents[0].degree)
    assert t0 is not None
    assert "M.Tech" in t0 or "MTech" in t0
    assert "Ph.D" not in t0

    # Intent 1 bound to Ph.D.
    t1 = plan.intents[1].target or (lossless.intents[1].target or lossless.intents[1].degree)
    assert t1 is not None
    assert "Ph.D" in t1 or "PhD" in t1

    preserved = " ".join(plan.preserved_terms).lower() + " " + plan.resolved_query.lower()
    assert "work experience" in preserved or "experience" in preserved
    assert "assistantship" in preserved


def test_scen_15_probation_max_credits() -> None:
    """scen_15: Probation under-load maximum credit limit preserves probation, credits, and max comparator."""
    q = (
        "What is the maximum credit limit per semester under the under-load rule for a student placed on "
        "academic probation?"
    )
    plan = build_query_plan(q)
    lossless = build_lossless_query(plan, question=q)

    assert plan.domain_decision == "in_scope"
    assert should_stop_pipeline(plan) is False
    assert plan.is_multi_intent is False
    assert len(plan.intents) == 1

    preserved = " ".join(plan.preserved_terms).lower() + " " + plan.resolved_query.lower()
    assert "probation" in preserved
    # Both 'credit' and 'credits' / 'credit limit' must survive
    assert "credit" in preserved or "credits" in preserved
    assert any(term.lower() in ("credits", "credit", "credit limit") for term in plan.preserved_terms)

    # Meaning of maximum must survive in constraints, qualifiers, or preserved terms
    all_constraints = " ".join(str(c) for c in plan.intents[0].constraints).lower()
    all_terms = " ".join(plan.preserved_terms).lower()
    assert ("max" in all_constraints or "maximum" in all_constraints or "max" in all_terms or "maximum" in all_terms)


# ============================================================================
# PART 2: COUNTER-TESTS (PREVENT INVERSE REGRESSIONS)
# ============================================================================

class TestCounterTests:
    """Counter-tests asserting both sides of segmentation and binding rules."""

    def test_counter_separate_degrees_must_split(self) -> None:
        """Counter: Asking about two distinct degrees (M.Sc. fee + Ph.D. eligibility) MUST split."""
        q = "What is the application fee for M.Sc. and what is the eligibility criteria for Ph.D.?"
        plan = build_query_plan(q)
        assert plan.is_multi_intent is True
        assert len(plan.intents) >= 2
        t0 = plan.intents[0].target
        t1 = plan.intents[1].target
        assert t0 and ("M.Sc" in t0 or "MSc" in t0)
        assert t1 and ("Ph.D" in t1 or "PhD" in t1)

    def test_counter_facility_and_governance_must_split(self) -> None:
        """Counter: Facility inquiry + committee inquiry MUST split."""
        q = "Where is the student hostel located, and who is on the anti-ragging committee?"
        plan = build_query_plan(q)
        assert plan.is_multi_intent is True
        assert len(plan.intents) >= 2

    def test_counter_single_programme_checklist_must_not_split(self) -> None:
        """Counter: Multiple eligibility facets for a SINGLE degree must NOT split."""
        q = "To apply for regular M.Tech, do I need a valid GATE score, a 60% B.Tech aggregate, and is an interview required?"
        plan = build_query_plan(q)
        assert plan.is_multi_intent is False
        assert len(plan.intents) == 1
        assert "M.Tech" in (plan.intents[0].target or "")

    def test_counter_local_target_binding_isolation(self) -> None:
        """Counter: First clause degree must NOT be overwritten by second clause degree."""
        q = "Does M.Sc. require JAM, and does B.Tech require JEE?"
        plan = build_query_plan(q)
        assert len(plan.intents) >= 2
        assert "M.Sc" in (plan.intents[0].target or "") or "MSc" in (plan.intents[0].target or "")
        assert "B.Tech" in (plan.intents[1].target or "") or "BTech" in (plan.intents[1].target or "")

    def test_counter_contextual_without_vs_negation(self) -> None:
        """Counter: Contextual 'without quitting my job' must NOT generate 'without' retrieval query."""
        q = "Can I do an Executive M.Tech without quitting my job?"
        plan = build_query_plan(q)
        retrieval_queries = retrieval_queries_from_plan(plan)
        for rq in retrieval_queries:
            assert rq.strip().lower() != "without"
            assert "without" not in rq.split() or len(rq.split()) > 2

    def test_counter_disjunctive_or_qualification_route(self) -> None:
        """Counter: Qualifying degree OR first class must be captured as an OR relationship."""
        q = "Is admission granted based on a first class or a qualifying degree with 60%?"
        plan = build_query_plan(q)
        lossless = build_lossless_query(plan, question=q)
        or_conds = [c for c in lossless.condition_relationships if c.logical_relation == "OR"]
        assert len(or_conds) > 0

    def test_counter_conversational_history_override(self) -> None:
        """Counter: Explicit new degree mention in follow-up must override antecedent degree."""
        history = [
            {"role": "user", "content": "What is the fee for regular M.Tech?"},
            {"role": "assistant", "content": "The fee for regular M.Tech is Rs 30,000 per semester."},
        ]
        q = "What about Ph.D.?"
        resolved = resolve_conversation(question=q, chat_history=history)
        resolved_q = resolved.get("resolved_question") or q
        plan = build_query_plan(resolved_q)
        target = plan.intents[0].target
        assert target is not None
        assert "Ph.D" in target or "PhD" in target
        assert "M.Tech" not in target

    def test_counter_out_of_scope_boundary(self) -> None:
        """Counter: Completely out-of-scope question must be stopped."""
        q = "Who won the cricket world cup in 2023?"
        plan = build_query_plan(q)
        assert plan.domain_decision == "out_of_scope"
        assert should_stop_pipeline(plan) is True
