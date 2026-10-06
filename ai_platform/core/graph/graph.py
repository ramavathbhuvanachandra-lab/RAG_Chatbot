"""Canonical LangGraph wiring for the reusable AI platform."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ai_platform.core.graph.nodes import CoreNodes
from ai_platform.core.graph.state import RAGState


def _route_after_query_planning(
    state: dict[str, Any],
) -> str:
    """
    High-confidence out-of-scope queries go directly to answer.

    answer_node has a terminal guard and therefore does NOT call the answer
    LLM in this branch.
    """
    if bool(state.get("pipeline_stop")):
        return "stop"

    return "continue"


def _route_after_intent_planning(
    state: dict[str, Any],
) -> str:
    return (
        "multi_intent"
        if bool(state.get("is_multi_intent"))
        else "single_intent"
    )


def build_graph(
    nodes: CoreNodes | None = None,
):
    runtime_nodes = nodes or CoreNodes()

    workflow = StateGraph(
        RAGState
    )

    workflow.add_node(
        "resolve_conversation",
        runtime_nodes.resolve_conversation_node,
    )

    workflow.add_node(
        "understand_query",
        runtime_nodes.understand_query_node,
    )

    workflow.add_node(
        "plan_multi_intent",
        runtime_nodes.plan_multi_intent_node,
    )

    workflow.add_node(
        "retrieve_and_qualify",
        runtime_nodes.retrieve_and_qualify_node,
    )

    workflow.add_node(
        "process_multi_intent",
        runtime_nodes.process_multi_intent_node,
    )

    workflow.add_node(
        "answer",
        runtime_nodes.answer_node,
    )

    workflow.add_edge(
        START,
        "resolve_conversation",
    )

    workflow.add_edge(
        "resolve_conversation",
        "understand_query",
    )

    workflow.add_conditional_edges(
        "understand_query",
        _route_after_query_planning,
        {
            "stop": "answer",
            "continue": "plan_multi_intent",
        },
    )

    workflow.add_conditional_edges(
        "plan_multi_intent",
        _route_after_intent_planning,
        {
            "single_intent": "retrieve_and_qualify",
            "multi_intent": "process_multi_intent",
        },
    )

    workflow.add_edge(
        "retrieve_and_qualify",
        "answer",
    )

    workflow.add_edge(
        "process_multi_intent",
        "answer",
    )

    workflow.add_edge(
        "answer",
        END,
    )

    return workflow.compile()


__all__ = [
    "build_graph",
]
