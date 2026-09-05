"""
IIT Jodhpur V1 — Phase 5.2
Decision-Ready Situation Context

Purpose
-------
Normalize a StudentSituation from Phase 5.1 into a representation that
later query planning and routing can consume without having to reinterpret
the original user message.

This module does NOT:
- retrieve documents
- evaluate college policy
- generate answers
- call an LLM

Core distinction
----------------
facts
    Things explicitly true about the student/situation.

preferences
    What the student would like.

constraints
    Conditions that restrict the decision.

goal
    What the student is trying to accomplish.

target
    The college/program/service the decision concerns.

missing_information
    Information that may be useful for a decision, but is not present.

No field is invented from the college corpus.

Phase 8D target-preservation rule
---------------------------------
The target is not simply "the first entity".

Decision routing keeps the original Phase-5 semantics:
- eligibility intents retain their intent-specific target;
- research-domain entities do not replace the research target;
- explicit academic-program entities can become the research target when
  they are more specific;
- explicit department/domain entities retain the historical research target
  behavior;
- other intents can use an explicit entity as the fallback target.

This prevents both:
    B.Tech + Ph.D. admission -> target "b.tech"
and:
    robotics/control systems research -> target "control systems"

while preserving:
    M.S. by Research + research -> target "m.s. by research"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from backend.student_situation import StudentSituation


# =========================================================
# Public data contract
# =========================================================

@dataclass(frozen=True)
class DecisionContext:
    """Decision-ready representation of a student situation."""

    goal: str
    target: str

    facts: Mapping[str, Any] = field(
        default_factory=dict
    )

    preferences: Mapping[str, Any] = field(
        default_factory=dict
    )

    constraints: Mapping[str, Any] = field(
        default_factory=dict
    )

    missing_information: tuple[str, ...] = ()

    confidence: float = 0.0


# =========================================================
# Target resolution helpers
# =========================================================

# These are semantic program forms, not institution IDs. They cover the
# academic-program vocabulary already present in StudentSituation. The
# matching algorithm remains generic, and other institutions can provide
# additional entity_terms without changing the retrieval architecture.
_ACADEMIC_PROGRAM_ENTITY_KEYS = frozenset(
    {
        "phd",
        "m.tech",
        "m.s.",
        "m.s. by research",
        "m.sc",
        "b.tech",
        "b.sc",
        "b.a",
    }
)

_RESEARCH_DEPARTMENT_KEYS = frozenset(
    {
        "electrical engineering",
        "computer science",
        "computer science and engineering",
        "mechanical engineering",
        "physics",
        "chemistry",
    }
)


def _explicit_program_entities(
    entities: tuple[str, ...],
) -> tuple[str, ...]:
    """Return explicit academic-program entities in source order."""
    return tuple(
        entity
        for entity in entities
        if entity.casefold()
        in _ACADEMIC_PROGRAM_ENTITY_KEYS
    )


def _research_department_entities(
    entities: tuple[str, ...],
) -> tuple[str, ...]:
    """Return explicit research-department entities in source order."""
    return tuple(
        entity
        for entity in entities
        if entity.casefold()
        in _RESEARCH_DEPARTMENT_KEYS
    )


def _target_from_situation(
    situation: StudentSituation,
) -> str:
    """
    Resolve the primary target while preserving Phase-5 semantics.

    Precedence:
        1. Intent-specific eligibility/service target
        2. Research: explicit academic program
        3. Research: explicit department/domain
        4. Research fallback
        5. Other explicit entity
        6. Intent fallback

    The distinction between academic programs and research domains is
    important. A user's current degree (for example B.Tech) is a fact and
    must not become the target of a Ph.D. eligibility question. Likewise,
    "control systems" is a research interest, not necessarily the target
    program or institutional object.
    """
    entities = tuple(
        str(entity).strip()
        for entity in situation.entities
        if str(entity).strip()
    )

    # -----------------------------------------------------
    # Eligibility and service intents preserve their
    # established semantic target regardless of background
    # degree entities also mentioned in the message.
    # -----------------------------------------------------
    if situation.intent == "hostel":
        return "hostel"

    if situation.intent == "phd_eligibility":
        return "phd"

    if situation.intent == "mtech_eligibility":
        return "m.tech"

    if situation.intent == "navigation":
        return "campus_service"

    if situation.intent == "emergency":
        return "medical_service"

    # -----------------------------------------------------
    # Research is special:
    # an explicit academic program is more specific than the
    # broad research intent, while research departments remain
    # useful targets exactly as they were before 8D.
    # -----------------------------------------------------
    if situation.intent == "research":
        program_entities = _explicit_program_entities(
            entities
        )

        if program_entities:
            return program_entities[0]

        department_entities = _research_department_entities(
            entities
        )

        if department_entities:
            return department_entities[0]

        return "research"

    # -----------------------------------------------------
    # For other intents, use the first explicit entity as the
    # concrete target. This preserves the previous fallback
    # contract without changing intent-specific routing above.
    # -----------------------------------------------------
    if entities:
        return entities[0]

    return situation.intent


# =========================================================
# Fact normalization
# =========================================================

def _normalize_facts(
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy explicit user facts into a stable dictionary."""
    return dict(facts)


# =========================================================
# Preference extraction
# =========================================================

def _normalize_preferences(
    situation: StudentSituation,
) -> dict[str, Any]:
    """Separate preference-like signals from constraints."""
    preferences: dict[str, Any] = {}

    constraints = situation.constraints

    if constraints.get(
        "research_preference"
    ) is True:
        preferences[
            "research_oriented"
        ] = True

    if constraints.get(
        "interdisciplinary_preference"
    ) is True:
        preferences[
            "interdisciplinary"
        ] = True

    if constraints.get(
        "cost_sensitive"
    ) is True:
        preferences[
            "cost_sensitive"
        ] = True

    if "research_interests" in constraints:
        preferences[
            "research_interests"
        ] = tuple(
            constraints[
                "research_interests"
            ]
        )

    return preferences


# =========================================================
# Constraint extraction
# =========================================================

def _normalize_constraints(
    situation: StudentSituation,
) -> dict[str, Any]:
    """Keep decision-limiting conditions separate from preferences."""
    source = situation.constraints
    constraints: dict[str, Any] = {}

    for key in (
        "stay_duration",
        "occupancy",
        "bedding",
        "budget",
        "comparison_requested",
    ):
        if key in source:
            constraints[key] = source[key]

    return constraints


# =========================================================
# Missing information
# =========================================================

def _missing_information(
    situation: StudentSituation,
    target: str,
    facts: Mapping[str, Any],
    preferences: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> tuple[str, ...]:
    """Identify useful missing information without inventing facts."""
    missing: list[str] = []

    if (
        situation.goal == "determine_eligibility"
        and not facts
    ):
        missing.append(
            "personal_qualification_details"
        )

    if (
        situation.goal == "compare_options"
        and not preferences
        and not constraints
    ):
        missing.append(
            "decision_criteria"
        )

    if (
        situation.intent == "hostel"
        and
        situation.goal in {
            "estimate_cost",
            "minimize_cost",
        }
    ):
        if (
            "stay_duration" not in constraints
            and "budget" not in constraints
        ):
            missing.append(
                "stay_duration_or_budget"
            )

    if (
        situation.goal == "find_relevant_research"
        and
        "research_interests" not in preferences
    ):
        missing.append(
            "research_interest"
        )

    return tuple(
        missing
    )


# =========================================================
# Validation
# =========================================================

def _validate(
    context: DecisionContext,
) -> DecisionContext:
    """Enforce basic invariants before later stages."""
    if not context.goal:
        raise ValueError(
            "DecisionContext.goal cannot be empty."
        )

    if not context.target:
        raise ValueError(
            "DecisionContext.target cannot be empty."
        )

    if (
        context.confidence < 0.0
        or context.confidence > 1.0
    ):
        raise ValueError(
            "DecisionContext.confidence must be between 0 and 1."
        )

    overlap = (
        set(context.facts)
        & set(context.preferences)
        & set(context.constraints)
    )

    if overlap:
        raise ValueError(
            "Decision fields overlap: "
            + ", ".join(
                sorted(overlap)
            )
        )

    return context


# =========================================================
# Public API
# =========================================================

def build_decision_context(
    situation: StudentSituation,
) -> DecisionContext:
    """Convert StudentSituation into a Phase-5.2 DecisionContext."""
    target = _target_from_situation(
        situation
    )

    facts = _normalize_facts(
        situation.user_facts
    )

    preferences = _normalize_preferences(
        situation
    )

    constraints = _normalize_constraints(
        situation
    )

    missing = _missing_information(
        situation,
        target,
        facts,
        preferences,
        constraints,
    )

    context = DecisionContext(
        goal=situation.goal,
        target=target,
        facts=facts,
        preferences=preferences,
        constraints=constraints,
        missing_information=missing,
        confidence=situation.confidence,
    )

    return _validate(
        context
    )


def decision_context_to_dict(
    context: DecisionContext,
) -> dict[str, Any]:
    """Serialize DecisionContext into a JSON-safe dictionary."""
    return {
        "goal": context.goal,
        "target": context.target,
        "facts": dict(
            context.facts
        ),
        "preferences": dict(
            context.preferences
        ),
        "constraints": dict(
            context.constraints
        ),
        "missing_information": list(
            context.missing_information
        ),
        "confidence": context.confidence,
    }