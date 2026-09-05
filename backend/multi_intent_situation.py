"""
IIT Jodhpur V1 — Phase 5.7
Multi-Intent + Situation

Purpose
-------
Attach relevant student situation information to the correct intent in a
multi-intent request.

The existing multi-intent system already:
    - decomposes requests
    - retrieves evidence independently
    - preserves evidence ownership by intent
    - builds one final answer

This module adds only one thing:

    global student situation
                ↓
        intent relevance
                ↓
       per-intent situation

Design principles
-----------------
- Deterministic.
- Conservative.
- No LLM call.
- No retrieval.
- No policy evaluation.
- No mutation of the original StudentSituation.
- Irrelevant intents receive no unrelated personal context.

The global situation is applied only when there is sufficient evidence
that the intent is related to the student's situation.

This prevents:

    "I have 72% B.Tech, can I apply for Ph.D., and where is the library?"

from becoming:

    library intent + 72% + B.Tech + Ph.D.

inside the answer model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


# =========================================================
# Data contract
# =========================================================

@dataclass(frozen=True)
class IntentSituation:
    """
    Situation information that belongs to one intent.
    """

    applies: bool
    goal: str = ""
    target: str = ""
    facts: Mapping[str, Any] = None
    preferences: Mapping[str, Any] = None
    constraints: Mapping[str, Any] = None
    missing_information: tuple[str, ...] = ()


# =========================================================
# Normalization helpers
# =========================================================

def _normalize(
    value: Any,
) -> str:
    """Normalize text for comparison."""
    return " ".join(
        str(value or "").strip().casefold().split()
    )


def _non_empty_mapping(
    value: Any,
) -> dict[str, Any]:
    """
    Convert mapping-like values to a clean dictionary.
    """
    if not isinstance(
        value,
        Mapping,
    ):
        return {}

    return {
        str(key): item
        for key, item in value.items()
        if item not in (
            None,
            "",
            [],
            {},
            (),
        )
    }


# =========================================================
# Intent / situation matching
# =========================================================

def _intent_target(
    intent: Mapping[str, Any],
) -> str:
    """
    Obtain the strongest target signal available for an intent.
    """
    entities = intent.get(
        "entities",
        [],
    )

    if entities:
        return _normalize(
            entities[0]
        )

    topics = intent.get(
        "topics",
        [],
    )

    if topics:
        return _normalize(
            topics[0]
        )

    return ""


def _situation_target(
    situation: Any,
    decision_context: Any,
) -> str:
    """
    Prefer the normalized DecisionContext target.
    """
    target = _normalize(
        getattr(
            decision_context,
            "target",
            "",
        )
    )

    if target:
        return target

    return _normalize(
        getattr(
            situation,
            "intent",
            "",
        )
    )


def _situation_entities(
    situation: Any,
) -> set[str]:
    """
    Extract explicit entities from the situation.
    """
    return {
        _normalize(
            entity
        )
        for entity in getattr(
            situation,
            "entities",
            (),
        )
        if _normalize(entity)
    }


def _intent_terms(
    intent: Mapping[str, Any],
) -> set[str]:
    """
    Collect normalized semantic terms from an intent.
    """
    terms: set[str] = set()

    terms.update(
        _normalize(
            value
        )
        for value in intent.get(
            "topics",
            [],
        )
        if _normalize(value)
    )

    terms.update(
        _normalize(
            value
        )
        for value in intent.get(
            "entities",
            [],
        )
        if _normalize(value)
    )

    terms.update(
        token
        for token in _normalize(
            intent.get(
                "question",
                "",
            )
        ).split()
        if len(token) >= 4
    )

    return terms


# =========================================================
# Relevance rules
# =========================================================

def _intent_matches_situation(
    intent: Mapping[str, Any],
    situation: Any,
    decision_context: Any,
) -> bool:
    """
    Determine whether the global student situation belongs to this intent.

    Conservative strategy:
        1. Exact target match.
        2. Situation entity appears in intent terms.
        3. Situation intent is explicitly represented in the request.
        4. For situation-heavy intents such as hostel/research/eligibility,
           explicit semantic overlap is required.

    We intentionally do NOT attach situation context to every intent.
    """

    if situation is None:
        return False

    intent_name = _normalize(
        getattr(
            situation,
            "intent",
            "",
        )
    )

    target = _situation_target(
        situation,
        decision_context,
    )

    terms = _intent_terms(
        intent
    )

    question = _normalize(
        intent.get(
            "question",
            "",
        )
    )

    # -----------------------------------------------------
    # Strong explicit target match.
    # -----------------------------------------------------

    if target:
        if target in question:
            return True

        if target in terms:
            return True

    # -----------------------------------------------------
    # Explicit situation entity match.
    # -----------------------------------------------------

    for entity in _situation_entities(
        situation
    ):
        if entity in terms:
            return True

    # -----------------------------------------------------
    # Intent-name match.
    #
    # This is useful when entity detectors do not expose a useful
    # entity but the independent intent question clearly mentions
    # the situation category.
    # -----------------------------------------------------

    if intent_name:
        intent_key = intent_name.replace(
            "_",
            " ",
        )

        if (
            intent_key in question
            or intent_name in question
        ):
            return True

    # -----------------------------------------------------
    # Conservative synonym families.
    # -----------------------------------------------------

    families = {
        "hostel": {
            "hostel",
            "accommodation",
            "room",
            "rooms",
            "lodging",
        },
        "research": {
            "research",
            "research areas",
            "research themes",
            "laboratory",
            "laboratories",
        },
        "phd_eligibility": {
            "phd",
            "ph.d",
            "eligibility",
            "admission",
            "requirements",
        },
        "mtech_eligibility": {
            "mtech",
            "m.tech",
            "eligibility",
            "admission",
            "requirements",
        },
        "navigation": {
            "library",
            "location",
            "directions",
            "where",
            "campus",
        },
    }

    family = families.get(
        intent_name
    )

    if not family:
        return False

    return bool(
        terms & family
    )


# =========================================================
# Public API
# =========================================================

def build_intent_situation(
    *,
    intent: Mapping[str, Any],
    situation: Any = None,
    decision_context: Any = None,
) -> IntentSituation:
    """
    Build situation information for one independent intent.

    If the situation does not belong to the intent, returns an empty
    IntentSituation with applies=False.
    """

    applies = _intent_matches_situation(
        intent,
        situation,
        decision_context,
    )

    if not applies:
        return IntentSituation(
            applies=False,
            facts={},
            preferences={},
            constraints={},
        )

    goal = str(
        getattr(
            decision_context,
            "goal",
            "",
        )
        or getattr(
            situation,
            "goal",
            "",
        )
        or ""
    )

    target = str(
        getattr(
            decision_context,
            "target",
            "",
        )
        or ""
    )

    facts = _non_empty_mapping(
        getattr(
            decision_context,
            "facts",
            {},
        )
    )

    if not facts:
        facts = _non_empty_mapping(
            getattr(
                situation,
                "user_facts",
                {},
            )
        )

    preferences = _non_empty_mapping(
        getattr(
            decision_context,
            "preferences",
            {},
        )
    )

    if not preferences:
        preferences = _non_empty_mapping(
            getattr(
                situation,
                "preferences",
                {},
            )
        )

    constraints = _non_empty_mapping(
        getattr(
            decision_context,
            "constraints",
            {},
        )
    )

    if not constraints:
        constraints = _non_empty_mapping(
            getattr(
                situation,
                "constraints",
                {},
            )
        )

    missing_information = tuple(
        getattr(
            decision_context,
            "missing_information",
            (),
        )
        or ()
    )

    return IntentSituation(
        applies=True,
        goal=goal,
        target=target,
        facts=facts,
        preferences=preferences,
        constraints=constraints,
        missing_information=missing_information,
    )


def build_multi_intent_situations(
    *,
    intent_units: list[Mapping[str, Any]],
    situation: Any = None,
    decision_context: Any = None,
) -> list[IntentSituation]:
    """
    Build situation context independently for every intent.
    """
    return [
        build_intent_situation(
            intent=intent,
            situation=situation,
            decision_context=decision_context,
        )
        for intent in intent_units
    ]


def situation_to_dict(
    situation: IntentSituation,
) -> dict[str, Any]:
    """
    Serialize an intent-scoped situation into a simple dictionary.
    """
    return {
        "applies": situation.applies,
        "goal": situation.goal,
        "target": situation.target,
        "facts": dict(
            situation.facts or {}
        ),
        "preferences": dict(
            situation.preferences or {}
        ),
        "constraints": dict(
            situation.constraints or {}
        ),
        "missing_information": list(
            situation.missing_information
        ),
    }


def format_multi_intent_situation_context(
    intent_results: list[Mapping[str, Any]],
) -> str:
    """
    Format already-scoped situations for the final answer model.

    Only intents with applies=True receive personal situation context.
    """
    sections: list[str] = []

    for index, intent in enumerate(
        intent_results,
        start=1,
    ):
        situation = intent.get(
            "situation_context",
            {},
        )

        if not situation:
            continue

        if not situation.get(
            "applies",
            False,
        ):
            continue

        lines = [
            f"Situation for Intent {index}:",
        ]

        goal = situation.get(
            "goal"
        )

        target = situation.get(
            "target"
        )

        if goal:
            lines.append(
                f"- goal: {goal}"
            )

        if target:
            lines.append(
                f"- target: {target}"
            )

        facts = situation.get(
            "facts",
            {},
        )

        if facts:
            lines.append(
                "User facts:"
            )

            for key, value in facts.items():
                lines.append(
                    f"- {key}: {value}"
                )

        preferences = situation.get(
            "preferences",
            {},
        )

        if preferences:
            lines.append(
                "Preferences:"
            )

            for key, value in preferences.items():
                lines.append(
                    f"- {key}: {value}"
                )

        constraints = situation.get(
            "constraints",
            {},
        )

        if constraints:
            lines.append(
                "Constraints:"
            )

            for key, value in constraints.items():
                lines.append(
                    f"- {key}: {value}"
                )

        sections.append(
            "\n".join(lines)
        )

    if not sections:
        return ""

    return (
        "MULTI-INTENT SITUATION CONTEXT:\n"
        "Use personal situation information only for the numbered "
        "intent where it is provided. Never transfer a user's facts "
        "or constraints to another intent.\n\n"
        + "\n\n".join(
            sections
        )
    )
