"""
Canonical retrieval-query planning for the reusable RAG core.

Purpose
-------
Convert a SemanticQueryFrame into a small, conservative retrieval plan.

Core rule
---------
The original user question is ALWAYS the primary retrieval query.

A deterministic structured query may be added as a supplementary
retrieval signal. It can improve recall, but it can never replace
the user's original wording.

This module is:
    - institution-agnostic
    - deterministic
    - bounded
    - free of LLM-generated retrieval rewrites
    - safe for reuse across institutions
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from backend.core.query_frame import SemanticQueryFrame


# ============================================================
# Constants
# ============================================================

MAX_RETRIEVAL_QUERIES = 3
MIN_SUBSTANTIVE_TOKEN_LENGTH = 2


# ============================================================
# Retrieval Query Plan
# ============================================================


@dataclass(frozen=True, slots=True)
class RetrievalQueryPlan:
    """
    Retrieval inputs derived from a SemanticQueryFrame.

    `original_query` is authoritative.

    `primary_query` MUST always equal the original user question.

    `structured_query` is an optional deterministic supplementary
    retrieval signal.

    `queries` always contains the original query first.

    `alternate_queries` contains only bounded supplementary queries.
    """

    original_query: str

    primary_query: str

    structured_query: str = ""

    # Kept as a compatibility field for older state/debug consumers.
    # The planner no longer generates or accepts an LLM semantic query.
    semantic_query: str = ""

    queries: tuple[str, ...] = field(
        default_factory=tuple
    )

    alternate_queries: tuple[str, ...] = field(
        default_factory=tuple
    )

    # Compatibility fields for older diagnostics/state consumers.
    # They are no longer part of retrieval decision-making.
    accepted_semantic_query: bool = False

    semantic_query_rejection_reason: str = (
        "semantic_query_generation_disabled"
    )

    mode: str = "original_only"

    confidence: float = 0.0

    def __post_init__(self) -> None:
        original = str(
            self.original_query or ""
        ).strip()

        primary = str(
            self.primary_query or ""
        ).strip()

        if not original:
            raise ValueError(
                "original_query cannot be empty."
            )

        if not primary:
            raise ValueError(
                "primary_query cannot be empty."
            )

        # The original question is always authoritative.
        if primary != original:
            raise ValueError(
                "primary_query must always equal original_query."
            )

        normalized_queries = _deduplicate_queries(
            self.queries
        )

        if not normalized_queries:
            normalized_queries = (
                primary,
            )

        if normalized_queries[0].casefold() != (
            primary.casefold()
        ):
            raise ValueError(
                "queries[0] must be the primary query."
            )

        # Hard safety bound.
        if len(normalized_queries) > MAX_RETRIEVAL_QUERIES:
            normalized_queries = (
                normalized_queries[
                    :MAX_RETRIEVAL_QUERIES
                ]
            )

        alternate_queries = tuple(
            normalized_queries[1:]
        )

        confidence = float(
            self.confidence
        )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        object.__setattr__(
            self,
            "original_query",
            original,
        )

        object.__setattr__(
            self,
            "primary_query",
            primary,
        )

        object.__setattr__(
            self,
            "structured_query",
            str(
                self.structured_query
                or ""
            ).strip(),
        )

        object.__setattr__(
            self,
            "semantic_query",
            str(
                self.semantic_query
                or ""
            ).strip(),
        )

        object.__setattr__(
            self,
            "queries",
            normalized_queries,
        )

        object.__setattr__(
            self,
            "alternate_queries",
            alternate_queries,
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "primary_query": self.primary_query,
            "structured_query": self.structured_query,
            "semantic_query": self.semantic_query,
            "queries": list(
                self.queries
            ),
            "alternate_queries": list(
                self.alternate_queries
            ),
            "accepted_semantic_query": (
                self.accepted_semantic_query
            ),
            "semantic_query_rejection_reason": (
                self.semantic_query_rejection_reason
            ),
            "mode": self.mode,
            "confidence": self.confidence,
        }


# ============================================================
# Text normalization
# ============================================================


def normalize_query_text(
    text: str,
) -> str:
    """
    Normalize text for comparison/deduplication only.

    This function NEVER replaces the original retrieval query.
    """

    value = str(
        text or ""
    ).casefold()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    return " ".join(
        value.split()
    )


def _tokens(
    text: str,
) -> set[str]:
    normalized = normalize_query_text(
        text
    )

    return {
        token
        for token in normalized.split()
        if len(token)
        >= MIN_SUBSTANTIVE_TOKEN_LENGTH
    }


def _deduplicate_queries(
    queries: Iterable[str],
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for query in queries:
        cleaned = " ".join(
            str(
                query or ""
            ).strip().split()
        )

        if not cleaned:
            continue

        key = normalize_query_text(
            cleaned
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            cleaned
        )

    return tuple(
        result
    )


# ============================================================
# Semantic-signal extraction
# ============================================================


def _facet_values(
    frame: SemanticQueryFrame,
) -> list[str]:
    values: list[str] = []

    for facet in frame.facets:
        value = str(
            facet.value or ""
        ).strip()

        if value:
            values.append(
                value
            )

    return values


def _structured_values(
    frame: SemanticQueryFrame,
) -> list[str]:
    """
    Collect meaning-bearing values from the interpreted frame.

    These values come from the user's understood information need.
    They are not institution-specific vocabulary.
    """

    values: list[str] = []

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if frame.target:
        values.append(
            frame.target
        )

    # --------------------------------------------------------
    # Request type
    # --------------------------------------------------------

    if frame.request_type:
        values.append(
            frame.request_type
        )

    # --------------------------------------------------------
    # Facets
    # --------------------------------------------------------

    values.extend(
        _facet_values(
            frame
        )
    )

    # --------------------------------------------------------
    # Qualifiers
    # --------------------------------------------------------

    values.extend(
        frame.qualifiers
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    values.extend(
        frame.conditions
    )

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    values.extend(
        frame.relations
    )

    # --------------------------------------------------------
    # Temporal context
    # --------------------------------------------------------

    values.extend(
        frame.temporal_context
    )

    # --------------------------------------------------------
    # Comparison targets
    # --------------------------------------------------------

    values.extend(
        frame.comparison_targets
    )

    # --------------------------------------------------------
    # User-preserved terminology
    # --------------------------------------------------------

    values.extend(
        frame.preserved_terms
    )

    return list(
        _deduplicate_queries(
            values
        )
    )


# ============================================================
# Structured query
# ============================================================


def build_structured_query(
    frame: SemanticQueryFrame,
) -> str:
    """
    Build a deterministic supplementary retrieval query.

    This is NOT an answer and it is NOT a replacement for the
    user's question.

    Example:

        target = "M.Sc."
        request_type = "admission routes"

    may become:

        "M.Sc. admission routes"

    The target is retained exactly as represented by the
    interpreted frame.
    """

    values = _structured_values(
        frame
    )

    return " ".join(
        values
    )


# ============================================================
# Retrieval plan construction
# ============================================================


def _should_add_structured_query(
    original_query: str,
    structured_query: str,
    confidence: float,
) -> bool:
    """
    Decide whether the structured query is useful enough to add.

    Conservative policy:

        - no structured query when empty
        - no structured query when interpretation confidence is low
        - no structured query when it is effectively identical
          to the original user question
    """

    if not structured_query:
        return False

    if confidence < 0.50:
        return False

    if (
        normalize_query_text(
            structured_query
        )
        == normalize_query_text(
            original_query
        )
    ):
        return False

    return True


def build_retrieval_query_plan(
    frame: SemanticQueryFrame,
) -> RetrievalQueryPlan:
    """
    Build a bounded retrieval plan.

    Ordering:

        1. Original user question
        2. Deterministic structured query

    The original question can never be displaced.

    No LLM-generated semantic query is created here.
    """

    if not isinstance(
        frame,
        SemanticQueryFrame,
    ):
        raise TypeError(
            "frame must be a SemanticQueryFrame."
        )

    original_query = str(
        frame.original_query or ""
    ).strip()

    if not original_query:
        raise ValueError(
            "frame.original_query cannot be empty."
        )

    confidence = float(
        frame.confidence
    )

    structured_query = (
        build_structured_query(
            frame
        )
    )

    queries: list[str] = [
        original_query
    ]

    mode = "original_only"

    # --------------------------------------------------------
    # Supplementary structured retrieval
    # --------------------------------------------------------

    if _should_add_structured_query(
        original_query=original_query,
        structured_query=structured_query,
        confidence=confidence,
    ):
        queries.append(
            structured_query
        )

        mode = "original_plus_structured"

    # --------------------------------------------------------
    # Final safety bound + deduplication
    # --------------------------------------------------------

    queries = list(
        _deduplicate_queries(
            queries
        )
    )[
        :MAX_RETRIEVAL_QUERIES
    ]

    alternate_queries = tuple(
        queries[1:]
    )

    return RetrievalQueryPlan(
        original_query=original_query,
        primary_query=original_query,
        structured_query=structured_query,
        semantic_query="",
        queries=tuple(
            queries
        ),
        alternate_queries=alternate_queries,
        accepted_semantic_query=False,
        semantic_query_rejection_reason=(
            "semantic_query_generation_disabled"
        ),
        mode=mode,
        confidence=confidence,
    )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "RetrievalQueryPlan",
    "normalize_query_text",
    "build_structured_query",
    "build_retrieval_query_plan",
    "MAX_RETRIEVAL_QUERIES",
]