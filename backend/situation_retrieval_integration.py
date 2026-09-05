"""
IIT Jodhpur V1 — Phase 5 Integration
Student Situation -> Existing Retrieval

Purpose
-------
Connect Phase 5.1 -> 5.2 -> 5.3 -> 5.4 to the existing retrieval
pipeline without creating a second retrieval system.

This node only prepares retrieval inputs.

It does NOT:
- retrieve documents
- rerank documents
- call an LLM
- generate answers
- change evidence logic

The existing hybrid_retrieve() remains the retrieval engine.
"""

from __future__ import annotations

from typing import Any

from backend.student_situation import (
    understand_student_situation,
)
from backend.situation_decision import (
    build_decision_context,
)
from backend.retrieval_query_planner import (
    build_retrieval_plan,
)
from backend.situation_retrieval_control import (
    build_retrieval_control,
)


def prepare_situation_retrieval(
    question: str,
) -> dict[str, Any]:
    """
    Build the complete Phase-5 retrieval input chain.

    Flow:
        question
          -> StudentSituation
          -> DecisionContext
          -> RetrievalPlan
          -> RetrievalControlPlan
          -> bounded retrieval queries
    """

    # -----------------------------------------------------
    # Phase 5.1
    # -----------------------------------------------------
    situation = understand_student_situation(
        question
    )

    # -----------------------------------------------------
    # Phase 5.2
    # -----------------------------------------------------
    decision_context = build_decision_context(
        situation
    )

    # -----------------------------------------------------
    # Phase 5.3
    # -----------------------------------------------------
    retrieval_plan = build_retrieval_plan(
        situation,
        decision_context,
    )

    # -----------------------------------------------------
    # Phase 5.4
    # -----------------------------------------------------
    retrieval_control = build_retrieval_control(
        original_query=question,
        primary_query=retrieval_plan.primary_query,
        alternate_queries=retrieval_plan.alternate_queries,
        context=decision_context,
        preserved_signals=retrieval_plan.preserved_signals,
        confidence=decision_context.confidence,
    )

    return {
        "student_situation": situation,
        "decision_context": decision_context,
        "retrieval_plan": retrieval_plan,
        "retrieval_control": retrieval_control,

        # This preserves the existing GraphState contract while giving
        # hybrid_retrieve() a bounded query list.
        "generated_queries": list(
            retrieval_control.queries
        ),
    }


def prepare_situation_retrieval_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    LangGraph node wrapper.

    Uses resolved_question when conversation resolution produced one.
    """
    question = state.get(
        "resolved_question",
        state["question"],
    )

    return prepare_situation_retrieval(
        question
    )
