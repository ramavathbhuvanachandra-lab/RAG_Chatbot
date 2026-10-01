"""
Canonical question-processing pipeline for the reusable RAG core.

Flow
----
User question
    ↓
Question understanding
    ↓
SemanticQueryFrame
    ↓
RetrievalQueryPlan

Design principles
-----------------
- The original user wording is always preserved.
- Question understanding extracts the information need; it does not answer it.
- The resulting frame guides retrieval but is never treated as factual truth.
- The original question remains the primary retrieval query.
- Retrieval planning is bounded and conservative.
- Weak/failed understanding still leaves the original question available.
- No institution-specific vocabulary belongs here.
- Model configuration remains centralized in backend.llm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.core.query_frame import (
    SemanticQueryFrame,
)
from backend.core.query_interpreter import (
    fallback_query_frame,
    interpret_query,
)
from backend.core.retrieval_query import (
    RetrievalQueryPlan,
    build_retrieval_query_plan,
)
from backend.llm import query_llm


# ============================================================
# Result contract
# ============================================================


@dataclass(frozen=True, slots=True)
class QueryPipelineResult:
    """
    Canonical result of question understanding + retrieval planning.

    `frame`
        Represents what the user is asking.

    `plan`
        Represents the bounded retrieval inputs derived from that meaning.

    Understanding and retrieval remain separate responsibilities.
    """

    frame: SemanticQueryFrame

    plan: RetrievalQueryPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "retrieval_plan": self.plan.to_dict(),
        }


# ============================================================
# Main pipeline
# ============================================================


def process_query(
    query: str,
) -> QueryPipelineResult:
    """
    Process one user question through the canonical reusable pipeline.

    Steps:

        1. Preserve the original question.
        2. Understand the information need.
        3. Build a bounded retrieval plan.

    The original wording remains authoritative throughout.
    """

    original_query = str(
        query or ""
    ).strip()

    if not original_query:
        raise ValueError(
            "query cannot be empty."
        )

    frame = interpret_query(
        original_query,
        llm=query_llm,
    )

    plan = build_retrieval_query_plan(
        frame
    )

    return QueryPipelineResult(
        frame=frame,
        plan=plan,
    )


# ============================================================
# Safe pipeline
# ============================================================


def process_query_safe(
    query: str,
) -> QueryPipelineResult:
    """
    Safely process one user question.

    Unexpected failures in question interpretation fall back to a frame
    containing the original question. Retrieval planning then continues
    from that safe frame.

    Deterministic planning errors are not silently swallowed.
    """

    original_query = str(
        query or ""
    ).strip()

    if not original_query:
        raise ValueError(
            "query cannot be empty."
        )

    try:

        return process_query(
            original_query
        )

    except Exception as exc:

        # The interpreter normally handles model failures internally.
        # This fallback protects the application from an unexpected
        # interpretation-layer failure while preserving the user's wording.
        try:

            frame = fallback_query_frame(
                original_query
            )

            plan = build_retrieval_query_plan(
                frame
            )

            return QueryPipelineResult(
                frame=frame,
                plan=plan,
            )

        except Exception:
            # Do not hide a genuine retrieval-planning/programming error.
            raise exc


# ============================================================
# Retrieval convenience API
# ============================================================


def build_retrieval_queries(
    query: str,
) -> tuple[str, ...]:
    """
    Return the bounded retrieval queries for a user question.

    The first query is always the original user wording.
    """

    result = process_query_safe(
        query
    )

    return result.plan.queries


def primary_retrieval_query(
    query: str,
) -> str:
    """
    Return the authoritative primary retrieval query.

    The primary query is always the original user wording.
    """

    result = process_query_safe(
        query
    )

    return result.plan.primary_query


# ============================================================
# Diagnostics
# ============================================================


def explain_query_processing(
    query: str,
) -> dict[str, Any]:
    """
    Return a complete diagnostic representation of question processing.
    """

    result = process_query_safe(
        query
    )

    return result.to_dict()


# ============================================================
# Public API
# ============================================================


__all__ = [
    "QueryPipelineResult",
    "process_query",
    "process_query_safe",
    "build_retrieval_queries",
    "primary_retrieval_query",
    "explain_query_processing",
]