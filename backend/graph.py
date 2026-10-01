"""
LangGraph workflow.

Pipeline
--------
Conversation resolution
    ↓
Multi-intent planning
    ↓
Single-intent:
    Canonical query processing
    ↓
    Hybrid retrieval
    ↓
    RRF / fusion
    ↓
    Reranking
    ↓
    Local context expansion
    ↓
    Evidence verification
    ↓
    Evidence coverage
    ↓
    Compression
    ↓
    Answer generation

Multi-intent:
    Independent evidence processing
    ↓
    One final answer generation call

Important invariants
--------------------
- The user's resolved question remains the authoritative question.
- Canonical query planning preserves that question as the primary query.
- Single-intent retrieval no longer depends on the legacy Phase-5
  StudentSituation -> DecisionContext -> RetrievalPlan chain.
- Multi-intent behavior remains unchanged for now.
- Retrieval, evidence, and answer-generation implementations are not
  redesigned in this file.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from backend.state import GraphState

from backend.core.query_pipeline import (
    process_query,
)

from backend.nodes import (
    resolve_conversation_node,
    plan_multi_intent_node,
    process_multi_intent_node,
    hybrid_retrieve,
    fuse_retrieved_documents,
    initial_rerank_documents,
    expand_retrieved_context,
    final_rerank_documents,
    assess_evidence_node,
    assess_evidence_coverage_node,
    compress_context,
    generate_answer,
)


# =========================================================
# Canonical Query Preparation
# =========================================================


def prepare_query_node(
    state: GraphState,
) -> GraphState:
    """
    Run the canonical reusable query-processing pipeline.

    Flow:
        resolved question
            ↓
        QueryInterpreter
            ↓
        SemanticQueryFrame
            ↓
        RetrievalQueryPlan
            ↓
        bounded retrieval queries

    The original user question remains the first/primary query.
    """

    question = str(
        state.get(
            "resolved_question",
            state["question"],
        )
        or ""
    ).strip()

    if not question:
        raise ValueError(
            "Cannot prepare query from an empty question."
        )

    result = process_query(
        question
    )

    return {
        "generated_queries": list(
            result.plan.queries
        ),
    }


# =========================================================
# Routing
# =========================================================


def route_after_intent_planning(
    state: GraphState,
) -> str:
    """
    Route to the multi-intent branch or the normal single-intent branch.
    """

    if state.get(
        "is_multi_intent",
        False,
    ):
        return "multi_intent"

    return "single_intent"


# =========================================================
# Create Graph
# =========================================================


def create_graph():
    """
    Build the production LangGraph workflow.
    """

    workflow = StateGraph(
        GraphState
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    workflow.add_node(
        "resolve_conversation",
        resolve_conversation_node,
    )

    workflow.add_node(
        "plan_multi_intent",
        plan_multi_intent_node,
    )

    workflow.add_node(
        "process_multi_intent",
        process_multi_intent_node,
    )

    workflow.add_node(
        "prepare_query",
        prepare_query_node,
    )

    workflow.add_node(
        "hybrid_retrieve",
        hybrid_retrieve,
    )

    workflow.add_node(
        "fuse_retrieved_documents",
        fuse_retrieved_documents,
    )

    workflow.add_node(
        "initial_rerank_documents",
        initial_rerank_documents,
    )

    workflow.add_node(
        "expand_retrieved_context",
        expand_retrieved_context,
    )

    workflow.add_node(
        "final_rerank_documents",
        final_rerank_documents,
    )

    workflow.add_node(
        "assess_evidence",
        assess_evidence_node,
    )

    workflow.add_node(
        "assess_evidence_coverage",
        assess_evidence_coverage_node,
    )

    workflow.add_node(
        "compress_context",
        compress_context,
    )

    workflow.add_node(
        "generate_answer",
        generate_answer,
    )

    # -----------------------------------------------------
    # Common entry
    # -----------------------------------------------------

    workflow.add_edge(
        START,
        "resolve_conversation",
    )

    workflow.add_edge(
        "resolve_conversation",
        "plan_multi_intent",
    )

    # -----------------------------------------------------
    # Intent routing
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        "plan_multi_intent",
        route_after_intent_planning,
        {
            "single_intent": "prepare_query",
            "multi_intent": "process_multi_intent",
        },
    )

    # -----------------------------------------------------
    # Multi-intent path
    #
    # Kept unchanged for this migration step.
    # -----------------------------------------------------

    workflow.add_edge(
        "process_multi_intent",
        "generate_answer",
    )

    # -----------------------------------------------------
    # Single-intent path
    # -----------------------------------------------------

    workflow.add_edge(
        "prepare_query",
        "hybrid_retrieve",
    )

    workflow.add_edge(
        "hybrid_retrieve",
        "fuse_retrieved_documents",
    )

    workflow.add_edge(
        "fuse_retrieved_documents",
        "initial_rerank_documents",
    )

    workflow.add_edge(
        "initial_rerank_documents",
        "expand_retrieved_context",
    )

    workflow.add_edge(
        "expand_retrieved_context",
        "final_rerank_documents",
    )

    workflow.add_edge(
        "final_rerank_documents",
        "assess_evidence",
    )

    workflow.add_edge(
        "assess_evidence",
        "assess_evidence_coverage",
    )

    workflow.add_edge(
        "assess_evidence_coverage",
        "compress_context",
    )

    workflow.add_edge(
        "compress_context",
        "generate_answer",
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    workflow.add_edge(
        "generate_answer",
        END,
    )

    return workflow.compile()


__all__ = [
    "create_graph",
    "route_after_intent_planning",
    "prepare_query_node",
]