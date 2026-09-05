"""
IIT Jodhpur V1 — Phase 5.6
Situational Answer Quality

Purpose
-------
Preserve the student's relevant situation into the final answer stage.

This module does NOT:
    - retrieve documents
    - call an LLM
    - change evidence
    - decide institutional policy
    - create another retrieval pipeline

It only builds a compact, deterministic instruction block containing:
    - target
    - goal
    - user facts
    - constraints
    - preferences

The existing answer LLM remains the only answer-generation call.
"""

from __future__ import annotations

from typing import Any


# =========================================================
# Normalization
# =========================================================

def _normalize_text(
    value: Any,
) -> str:
    """
    Normalize scalar values for deterministic prompt formatting.
    """
    return " ".join(
        str(value or "").strip().split()
    )


def _format_mapping(
    title: str,
    mapping: Any,
) -> list[str]:
    """
    Format a dictionary into stable prompt lines.

    Empty values are intentionally omitted.
    """
    if not isinstance(
        mapping,
        dict,
    ):
        return []

    lines: list[str] = []

    for key, value in mapping.items():

        if value in (
            None,
            "",
            [],
            {},
            (),
        ):
            continue

        lines.append(
            f"- {key}: {value}"
        )

    if not lines:
        return []

    return [
        title,
        *lines,
    ]


# =========================================================
# Public API
# =========================================================

def build_situational_answer_context(
    *,
    situation: Any = None,
    decision_context: Any = None,
) -> str:
    """
    Build the situational context used by the answer model.

    DecisionContext is preferred because it represents the normalized
    decision-facing interpretation.

    StudentSituation remains a fallback for fields that may not have
    been promoted into DecisionContext.
    """

    if (
        situation is None
        and decision_context is None
    ):
        return ""

    lines: list[str] = [
        "SITUATION CONTEXT:",
        (
            "Use the following user-specific information when it is "
            "relevant to the question. Do not invent missing facts. "
            "Do not treat user facts as institutional rules."
        ),
    ]

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    target = _normalize_text(
        getattr(
            decision_context,
            "target",
            "",
        )
    )

    if target:
        lines.append(
            f"- target: {target}"
        )

    # -----------------------------------------------------
    # Goal
    # -----------------------------------------------------

    goal = _normalize_text(
        getattr(
            decision_context,
            "goal",
            "",
        )
    )

    if not goal:
        goal = _normalize_text(
            getattr(
                situation,
                "goal",
                "",
            )
        )

    if goal:
        lines.append(
            f"- goal: {goal}"
        )

    # -----------------------------------------------------
    # User facts
    # -----------------------------------------------------

    facts = getattr(
        decision_context,
        "facts",
        {},
    )

    if not facts:
        facts = getattr(
            situation,
            "user_facts",
            {},
        )

    lines.extend(
        _format_mapping(
            "User facts:",
            facts,
        )
    )

    # -----------------------------------------------------
    # Constraints
    # -----------------------------------------------------

    constraints = getattr(
        decision_context,
        "constraints",
        {},
    )

    if not constraints:
        constraints = getattr(
            situation,
            "constraints",
            {},
        )

    lines.extend(
        _format_mapping(
            "Constraints:",
            constraints,
        )
    )

    # -----------------------------------------------------
    # Preferences
    # -----------------------------------------------------

    preferences = getattr(
        decision_context,
        "preferences",
        {},
    )

    if not preferences:
        preferences = getattr(
            situation,
            "preferences",
            {},
        )

    lines.extend(
        _format_mapping(
            "Preferences:",
            preferences,
        )
    )

    # -----------------------------------------------------
    # Answer behavior
    # -----------------------------------------------------

    lines.extend(
        [
            "",
            "SITUATIONAL ANSWER RULES:",
            (
                "1. Answer the user's actual situation, not merely "
                "the generic topic."
            ),
            (
                "2. Preserve relevant user facts, constraints, and "
                "preferences."
            ),
            (
                "3. Do not invent a missing user fact."
            ),
            (
                "4. Do not turn a user fact into an institutional rule."
            ),
            (
                "5. Do not infer eligibility, exemption, fees, "
                "requirements, or other policy conclusions beyond "
                "the supplied evidence."
            ),
            (
                "6. When the available information does not resolve the "
                "user's specific situation, state the uncertainty instead "
                "of guessing."
            ),
        ]
    )

    return "\n".join(
        lines
    )


def augment_answer_question(
    *,
    answer_question: str,
    situation: Any = None,
    decision_context: Any = None,
) -> str:
    """
    Add situational context to the existing answer instruction.

    The original question is preserved exactly at the beginning.
    """
    original = str(
        answer_question or ""
    )

    context = build_situational_answer_context(
        situation=situation,
        decision_context=decision_context,
    )

    if not context:
        return original

    return (
        original
        + "\n\n"
        + context
    )
