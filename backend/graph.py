"""
IIT Jodhpur V1 — LangGraph Workflow

Production pipeline
-------------------
Single-intent questions retain the established linear retrieval/evidence
pipeline.

Multi-intent questions are routed after conversation resolution into an
intent-specific evidence orchestration node. Each intent reuses the same
deterministic evidence pipeline, after which one final answer LLM call
composes the independent results.

Important invariants
--------------------
- Single-intent behavior remains unchanged after planning.
- Multi-intent processing does not add an answer LLM call per intent.
- Evidence for each intent is evaluated independently.
- A final answer is generated once.
"""

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from backend.state import GraphState

from backend.situation_retrieval_integration import (
    prepare_situation_retrieval_node,
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
# Routing
# =========================================================

def route_after_intent_planning(
    state: GraphState,
) -> str:
    """
    Route to the independent multi-intent path or the existing single-
    intent path.
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
    Build the production IIT Jodhpur V1 workflow.
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
        "prepare_situation_retrieval",
        prepare_situation_retrieval_node,
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
            "single_intent": "prepare_situation_retrieval",
            "multi_intent": "process_multi_intent",
        },
    )

    # -----------------------------------------------------
    # Multi-intent path
    # -----------------------------------------------------

    workflow.add_edge(
        "process_multi_intent",
        "generate_answer",
    )

    # -----------------------------------------------------
    # Phase-5 single-intent path
    # -----------------------------------------------------

    workflow.add_edge(
        "prepare_situation_retrieval",
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