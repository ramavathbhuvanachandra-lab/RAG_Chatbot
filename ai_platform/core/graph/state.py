"""LangGraph state contract for the reusable AI platform."""

from __future__ import annotations

from typing import Any, Sequence, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    chat_history: Sequence[Any]

    resolved_question: str
    conversation_mode: str
    active_topic: str
    active_entity: str

    # Canonical query-planner outputs.
    query_plan: Any
    query_scope: str
    query_scope_confidence: float
    query_scope_reason: str
    pipeline_stop: bool
    stop_reason: str

    is_multi_intent: bool
    intent_count: int
    intent_units: Sequence[Any]
    intent_questions: Sequence[str]

    query: Any
    query_frame: Any
    retrieval_queries: Sequence[str]

    retrieval_results: Sequence[Any]
    retrieval_weights: Sequence[float]

    fused_candidates: Sequence[Any]

    verified_candidates: Sequence[Any]
    uncertain_candidates: Sequence[Any]
    rejected_candidates: Sequence[Any]

    ranked_candidates: Sequence[Any]

    evidence_groups: Sequence[Any]
    evidence_documents: Sequence[Any]
    evidence_candidates: Sequence[Any]
    evidence_boundary_trace: Any
    rejected_evidence_anchor_ids: Sequence[str]

    evidence_assessment: Any
    evidence_coverage: Any
    claim_audit: Any
    evidence_package: Any
    evidence_package_status: str
    final_evidence_context: str
    ready_for_generation: bool
    claims: Sequence[Any]
    evidence_units: Sequence[Any]

    answer: str
    grounding_assessment: Any
    guard_result: Any
    answer_generation: Any
    answer_grounding: Any
    answer_guard: Any
    answer_grounding_status: str
    answer_guard_status: str
    answer_guard_reason: str
    model_calls: int
    context: str
    error: str


__all__ = ["RAGState"]
