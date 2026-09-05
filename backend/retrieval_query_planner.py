"""
IIT Jodhpur V1 — Phase 5.3
Situation → Retrieval Input Planner

Thin planning layer around the EXISTING retrieval engine.

It does not retrieve, rerank, call an LLM, or decide policy.
It converts Phase 5.1/5.2 structured meaning into concise retrieval
queries while preserving important facts, preferences, constraints, and
target qualifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RetrievalPlan:
    """Retrieval inputs for the existing retrieval pipeline."""

    primary_query: str
    alternate_queries: tuple[str, ...] = ()
    scope_hints: tuple[str, ...] = ()
    preserved_signals: tuple[str, ...] = ()
    confidence: float = 0.0


def _normalize_space(text: str) -> str:
    """Collapse whitespace and remove query-edge punctuation."""
    value = str(text or "").strip()
    value = (
        value.replace("–", "-")
        .replace("—", "-")
    )
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,;:-")


def _clean_fragment(value: Any) -> str:
    """Turn structured values into concise search fragments."""
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            key_text = _clean_fragment(key)
            item_text = _clean_fragment(item)
            if key_text and item_text:
                parts.append(f"{key_text} {item_text}")
            elif item_text:
                parts.append(item_text)
        return " ".join(parts)

    if isinstance(value, (tuple, list, set)):
        return " ".join(
            _clean_fragment(item)
            for item in value
            if _clean_fragment(item)
        )

    return str(value).strip()


def _canonical_term(value: Any) -> str:
    """Normalize internal semantic labels into searchable terminology."""
    text = _normalize_space(
        _clean_fragment(value)
    ).lower()

    aliases = {
        "bachelors_degree": "bachelor's degree",
        "masters_degree": "master's degree",
        "full_time": "full-time",
        "part_time": "part-time",
        "not_employed": "not employed",
        "phd": "Ph.D.",
        "m.tech": "M.Tech",
        "m.s.": "M.S.",
        "m.s": "M.S.",
        "m.s. by research": "M.S. by Research",
        "b.tech": "B.Tech",
        "b.sc": "B.Sc.",
        "b.a": "B.A.",
    }

    return aliases.get(text, text)


def _dedupe_preserve_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Deduplicate case-insensitively while preserving order."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        cleaned = _normalize_space(value)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return tuple(result)


def _goal_phrase(goal: str) -> str:
    """Map a structured goal to retrieval-oriented terminology."""
    mapping = {
        "determine_eligibility": "eligibility requirements",
        "understand_requirements": "requirements criteria",
        "minimize_cost": "cost charges rates",
        "estimate_cost": "cost charges rates",
        "compare_options": "comparison requirements differences",
        "choose_option": "options criteria",
        "find_relevant_research": "research areas topics",
        "plan_application": "admission application requirements",
        "get_contact": "contact details",
        "locate_service": "location directions",
        "find_information": "information",
    }

    return mapping.get(
        goal,
        str(goal).replace("_", " "),
    )


def _fact_signals(
    facts: Mapping[str, Any],
) -> tuple[str, ...]:
    """Convert explicit user facts into retrieval fragments."""
    signals: list[str] = []

    for key, value in facts.items():
        if key == "percentages":
            for item in value:
                signals.append(f"{float(item):g} percent")
            continue

        if key == "cgpa_values":
            for item in value:
                signals.append(f"{float(item):g} CGPA")
            continue

        if key == "degree_duration_years":
            signals.append(f"{value} year degree")
            continue

        text = _canonical_term(value)
        if text:
            signals.append(text)

    return _dedupe_preserve_order(signals)


def _preference_signals(
    preferences: Mapping[str, Any],
) -> tuple[str, ...]:
    """Convert explicit preferences into retrieval fragments."""
    signals: list[str] = []

    for key, value in preferences.items():
        if key == "research_oriented" and value is True:
            signals.append("research-oriented")
            continue

        if key == "interdisciplinary" and value is True:
            signals.append("interdisciplinary")
            continue

        if key == "cost_sensitive" and value is True:
            signals.append("cost-sensitive")
            continue

        if key == "research_interests":
            for item in value:
                text = _canonical_term(item)
                if text:
                    signals.append(text)
            continue

        text = _canonical_term(value)
        if text:
            signals.append(text)

    return _dedupe_preserve_order(signals)


def _constraint_signals(
    constraints: Mapping[str, Any],
) -> tuple[str, ...]:
    """Convert explicit decision constraints into search fragments."""
    signals: list[str] = []

    for key, value in constraints.items():
        if key == "stay_duration":
            if isinstance(value, Mapping):
                amount = value.get("value")
                unit = value.get("unit")
                if amount is not None and unit:
                    signals.append(
                        f"{float(amount):g} {unit} stay"
                        if isinstance(amount, (int, float))
                        else f"{amount} {unit} stay"
                    )
            continue

        if key == "occupancy":
            signals.append(f"{value} occupancy")
            continue

        if key == "bedding":
            signals.append(f"{value} bedding")
            continue

        if key == "budget":
            if isinstance(value, (int, float)):
                signals.append(f"budget {float(value):g}")
            else:
                signals.append(f"budget {value}")
            continue

        if key == "comparison_requested" and value is True:
            signals.append("compare options")
            continue

        text = _canonical_term(value)
        if text:
            signals.append(text)

    return _dedupe_preserve_order(signals)


def _all_preserved_signals(
    context: Any,
) -> tuple[str, ...]:
    """Collect high-value information in stable priority order."""
    return _dedupe_preserve_order(
        (
            *_fact_signals(
                getattr(context, "facts", {})
            ),
            *_preference_signals(
                getattr(context, "preferences", {})
            ),
            *_constraint_signals(
                getattr(context, "constraints", {})
            ),
        )
    )


def _target_phrase(
    situation: Any,
    context: Any,
) -> str:
    """
    Resolve the primary target.

    Explicit "regular Ph.D." wording is preserved because admission mode
    can materially affect which evidence is relevant.
    """
    target = _canonical_term(
        getattr(context, "target", "")
    )

    raw = _normalize_space(
        getattr(situation, "raw_text", "")
    ).lower()

    if (
        target.casefold() == "ph.d."
        and (
            "regular ph.d" in raw
            or "regular phd" in raw
        )
    ):
        return "regular Ph.D."

    return target


def _scope_hints(
    situation: Any,
    context: Any,
) -> tuple[str, ...]:
    """Build coarse semantic scope metadata."""
    values: list[str] = []

    intent = getattr(
        situation,
        "intent",
        "",
    )
    target = getattr(
        context,
        "target",
        "",
    )

    if intent:
        values.append(
            _canonical_term(intent)
        )

    if target:
        values.append(
            _canonical_term(target)
        )

    return _dedupe_preserve_order(values)


def _join_query(
    parts: Sequence[str],
) -> str:
    """Create a concise query from semantic fragments."""
    return _normalize_space(
        " ".join(
            _dedupe_preserve_order(parts)
        )
    )


def build_retrieval_plan(
    situation: Any,
    context: Any,
) -> RetrievalPlan:
    """
    Build retrieval inputs for the existing retrieval engine.

    No external calls are made.
    """
    target = _target_phrase(
        situation,
        context,
    )

    goal = _goal_phrase(
        getattr(
            context,
            "goal",
            "",
        )
    )

    facts = getattr(
        context,
        "facts",
        {},
    )
    preferences = getattr(
        context,
        "preferences",
        {},
    )
    constraints = getattr(
        context,
        "constraints",
        {},
    )

    fact_terms = _fact_signals(facts)
    preference_terms = _preference_signals(preferences)
    constraint_terms = _constraint_signals(constraints)

    preserved = _dedupe_preserve_order(
        (
            *fact_terms,
            *preference_terms,
            *constraint_terms,
        )
    )

    primary_query = _join_query(
        (
            target,
            goal,
            *preserved[:8],
        )
    )

    alternatives: list[str] = []

    if target and goal:
        alternatives.append(
            _join_query(
                (target, goal)
            )
        )

    if target and preserved:
        alternatives.append(
            _join_query(
                (target, *preserved)
            )
        )

    if fact_terms:
        alternatives.append(
            _join_query(
                (target, goal, *fact_terms)
            )
        )

    if preference_terms:
        alternatives.append(
            _join_query(
                (target, goal, *preference_terms)
            )
        )

    if constraint_terms:
        alternatives.append(
            _join_query(
                (target, goal, *constraint_terms)
            )
        )

    alternate_queries = tuple(
        query
        for query in _dedupe_preserve_order(
            alternatives
        )
        if query.casefold()
        != primary_query.casefold()
    )[:4]

    confidence = float(
        getattr(
            context,
            "confidence",
            0.0,
        )
    )

    return RetrievalPlan(
        primary_query=primary_query,
        alternate_queries=alternate_queries,
        scope_hints=_scope_hints(
            situation,
            context,
        ),
        preserved_signals=preserved,
        confidence=max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        ),
    )


def retrieval_plan_to_queries(
    plan: RetrievalPlan,
) -> list[str]:
    """Flatten the plan for the existing retriever."""
    return [
        plan.primary_query,
        *plan.alternate_queries,
    ]
