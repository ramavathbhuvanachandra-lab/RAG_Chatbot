
"""
IIT Jodhpur V1 — Phase 5.4
Situation-Aware Retrieval Control

Thin controller around the existing retrieval engine.

The original student wording is always preserved in the plan for audit
and fallback, but high-confidence retrieval uses structured queries so
irrelevant conversational details do not pollute search.

No retrieval, reranking, LLM, policy evaluation, or answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class RetrievalControlPlan:
    """Bounded final query set for the existing retriever."""

    original_query: str
    queries: tuple[str, ...]
    mode: str
    original_preserved: bool
    max_queries: int
    preserved_signals: tuple[str, ...] = ()
    confidence: float = 0.0


def _normalize(text: str) -> str:
    """Normalize whitespace only."""
    return " ".join(
        str(text or "").strip().split()
    )


def _dedupe_preserve_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Deduplicate strings case-insensitively."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        cleaned = _normalize(value)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return tuple(result)


def _has_structured_information(
    context: Any,
) -> bool:
    """
    Determine whether Phase 5.2 produced decision information.

    Raw user wording is intentionally ignored here.
    """
    return bool(
        getattr(context, "facts", {})
        or getattr(context, "preferences", {})
        or getattr(context, "constraints", {})
    )


def _bounded_alternates(
    primary_query: str,
    alternate_queries: Sequence[str],
) -> tuple[str, ...]:
    """Choose at most two distinct alternates."""
    primary_key = _normalize(
        primary_query
    ).casefold()

    result: list[str] = []

    for query in _dedupe_preserve_order(
        alternate_queries
    ):
        if query.casefold() == primary_key:
            continue

        result.append(query)

        if len(result) == 2:
            break

    return tuple(result)


def build_retrieval_control(
    *,
    original_query: str,
    primary_query: str,
    alternate_queries: Sequence[str] = (),
    context: Any,
    preserved_signals: Sequence[str] = (),
    confidence: float | None = None,
) -> RetrievalControlPlan:
    """
    Select which queries actually enter the existing retriever.

    Policy:
        confidence < 0.50:
            original wording only when available.

        confidence >= 0.50 and no structured information:
            one focused query.

        confidence >= 0.50 with structured information:
            primary structured query + at most two alternates.

    The original wording is retained separately in every plan.
    """
    original = _normalize(
        original_query
    )
    primary = _normalize(
        primary_query
    )

    if confidence is None:
        confidence = float(
            getattr(context, "confidence", 0.0)
        )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    structured = _has_structured_information(
        context
    )

    preserved = _dedupe_preserve_order(
        preserved_signals
    )

    # -----------------------------------------------------
    # Low confidence: do not trust the transformation.
    # -----------------------------------------------------
    if confidence < 0.50:
        queries = (
            (original,)
            if original
            else ((primary,) if primary else ())
        )

        return RetrievalControlPlan(
            original_query=original,
            queries=queries,
            mode="conservative",
            original_preserved=bool(original),
            max_queries=1,
            preserved_signals=preserved,
            confidence=confidence,
        )

    # -----------------------------------------------------
    # Simple/high-confidence request: no need for fan-out.
    # -----------------------------------------------------
    if not structured:
        if primary:
            queries = (primary,)
        elif original:
            queries = (original,)
        else:
            queries = ()

        return RetrievalControlPlan(
            original_query=original,
            queries=queries,
            mode="focused",
            original_preserved=bool(original),
            max_queries=1,
            preserved_signals=preserved,
            confidence=confidence,
        )

    # -----------------------------------------------------
    # Strong situation: structured query + bounded alternates.
    # -----------------------------------------------------
    candidates: list[str] = []

    if primary:
        candidates.append(primary)

    candidates.extend(
        _bounded_alternates(
            primary,
            alternate_queries,
        )
    )

    queries = _dedupe_preserve_order(
        candidates
    )[:3]

    return RetrievalControlPlan(
        original_query=original,
        queries=queries,
        mode="situational",
        original_preserved=bool(original),
        max_queries=3,
        preserved_signals=preserved,
        confidence=confidence,
    )


def retrieval_control_to_queries(
    plan: RetrievalControlPlan,
) -> list[str]:
    """Return only the queries to send to the existing retriever."""
    return list(plan.queries)
