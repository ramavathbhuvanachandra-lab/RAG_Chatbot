"""
IIT Jodhpur V1 — Conversation Resolver

Phase 7 — Final conversational hardening.

Purpose
-------
Resolve context-dependent user questions before retrieval while preserving
the correct conversational hierarchy.

The resolver distinguishes:

    main subject
        ↓
    active subtopic
        ↓
    active subtopic detail

Example:

    Ph.D. admission
        ↓
    bachelor's route
        ↓
    percentage
        ↓
    GATE

This allows a later question such as:

    "What percentage is needed for that route?"

to retain the correct parent context without hardcoding the institution.

Design principles
-----------------
- Institution agnostic.
- Facts never come from this module.
- Conversation history is context, not evidence.
- Standalone questions stay standalone.
- Explicit topic switches do not inherit stale context.
- Natural follow-ups can inherit hierarchical context.
- Conditional/alternative questions are supported.
- Ambiguous references are never guessed.
- Resolver failure never breaks the chatbot.
- Recent context is bounded.
- Internal diagnostics never become user-facing content.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from backend.llm import query_llm


# =========================================================
# Configuration
# =========================================================

MAX_HISTORY_MESSAGES = 8

VALID_MODES = {
    "standalone",
    "topic_switch",
    "follow_up",
    "ambiguous",
}

VALID_DIAGNOSTIC_REASONS = {
    "no_history",
    "obviously_ambiguous",
    "clearly_self_contained",
    "context_candidate",
    "resolver_success",
    "resolver_failure",
    "invalid_model_output",
}


# =========================================================
# Explicit conversational follow-ups
# =========================================================

EXPLICIT_FOLLOW_UP_PATTERNS = [
    r"^\s*what about\b",
    r"^\s*how about\b",
    r"^\s*and what about\b",
    r"^\s*how much\b",
    r"^\s*how many\b",
    r"^\s*when is it\b",
    r"^\s*where is it\b",
    r"^\s*where are they\b",
    r"^\s*is it\b",
    r"^\s*is that\b",
    r"^\s*does it\b",
    r"^\s*do they\b",
    r"^\s*can i\b",
    r"^\s*can they\b",
    r"^\s*could i\b",
    r"^\s*could they\b",
    r"^\s*would i\b",
    r"^\s*would they\b",
    r"^\s*what are its\b",
    r"^\s*what is its\b",
    r"^\s*tell me more\b",
    r"^\s*more about\b",
]


# =========================================================
# Elliptical continuation patterns
# =========================================================

ELLIPTICAL_FOLLOW_UP_PATTERNS = [
    r"^\s*and\b",
    r"^\s*also\b",
    r"^\s*then\b",
    r"\bthe other\b",
    r"\banother\b",
    r"\bthe same\b",
    r"\bthat\b",
    r"\bthose\b",
    r"\bthis\b",
    r"\bthese\b",
    r"\bthat route\b",
    r"\bthis route\b",
    r"\bthe other route\b",
    r"\banother route\b",
    r"\bthat option\b",
    r"\bthis option\b",
    r"\bthe other option\b",
    r"\banother option\b",
    r"\bthat program\b",
    r"\bthis program\b",
    r"\bthe other program\b",
    r"\banother program\b",
    r"\bthat degree\b",
    r"\bthis degree\b",
    r"\bthe other degree\b",
    r"\banother degree\b",
    r"\bthat requirement\b",
    r"\bthis requirement\b",
    r"\bthose requirements\b",
]


# =========================================================
# Conditional / alternative continuation patterns
# =========================================================

CONDITIONAL_CONTINUATION_PATTERNS = [
    r"^\s*what\s+if\b",
    r"^\s*what\s+happens\s+if\b",
    r"^\s*what\s+would\s+happen\s+if\b",
    r"^\s*would\s+the\s+alternative\b",
    r"^\s*would\s+an\s+alternative\b",
    r"^\s*would\s+another\b",
    r"^\s*can\s+i\s+instead\b",
    r"^\s*could\s+i\s+instead\b",
    r"^\s*would\s+i\s+instead\b",
    r"^\s*instead\b",
    r"^\s*alternatively\b",
    r"^\s*as\s+an\s+alternative\b",
    r"^\s*another\s+(?:way|option|route|path)\b",
    r"^\s*the\s+alternative\b",
    r"\binstead\s+of\s+(?:that|this|it)\b",
    r"\brather\s+than\s+(?:that|this|it)\b",
    r"\bif\s+i\s+(?:choose|take|switch|select)\b",
    r"\bif\s+we\s+(?:choose|take|switch|select)\b",
]


# =========================================================
# Context-reference patterns
# =========================================================

CONTEXT_REFERENCE_PATTERNS = [
    r"\bfor that\b",
    r"\bfor this\b",
    r"\bwith that\b",
    r"\bwith this\b",
    r"\bin that case\b",
    r"\bin this case\b",
    r"\bon that\b",
    r"\bon this\b",
    r"\bfrom that\b",
    r"\bfrom this\b",
    r"\bunder that\b",
    r"\bunder this\b",
    r"\bfor the other\b",
    r"\bfor another\b",
    r"\bwith the other\b",
    r"\bwith another\b",
]


# =========================================================
# Obviously ambiguous reference patterns
# =========================================================

AMBIGUOUS_REFERENCE_PATTERNS = [
    r"^\s*what about (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*how about (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*what is (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*how is (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*what are (?:those|these|they)\s*[?!.]*\s*$",
    r"^\s*and\s+(?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*(?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*(?:those|these|they)\s*[?!.]*\s*$",
    r"^\s*can\s+i\s+take\s+(?:that|this|it)\s+instead\s*[?!.]*\s*$",
    r"^\s*could\s+i\s+take\s+(?:that|this|it)\s+instead\s*[?!.]*\s*$",
    r"^\s*would\s+(?:that|this|it)\s+work\s*[?!.]*\s*$",
]


# =========================================================
# Generic contextual question patterns
# =========================================================

CONTEXTUAL_QUESTION_PATTERNS = [
    r"^\s*what\s+(?:percentage|percent|marks|score|cgpa|cpi)\b",
    r"^\s*what\s+(?:(?:entrance|admission)\s+)?(?:exam|examination|test)\b",
    r"^\s*what\s+(?:criteria|criterion|requirement|requirements)\b",
    r"^\s*what\s+(?:eligibility|qualification)\b",
    r"^\s*which\s+one\b",
    r"^\s*which\s+(?:option|route|path|program|programme|degree)\b",
    r"^\s*who\s+(?:is|are)\s+(?:eligible|related)\b",
    r"^\s*how\s+(?:long|much|many)\b",
    r"^\s*when\s+(?:is|are|does|do)\b",
    r"^\s*where\s+(?:is|are|can|do)\b",
    r"^\s*is\s+(?:it|that)\b",
    r"^\s*are\s+(?:they|those|these)\b",
    r"^\s*does\s+(?:it|that)\b",
    r"^\s*do\s+(?:they|those|these)\b",
    r"^\s*can\s+(?:i|they|we)\b",
    r"^\s*could\s+(?:i|they|we)\b",
    r"^\s*would\s+(?:i|they|we)\b",
]


# =========================================================
# Generic reference words
# =========================================================

CONTEXT_REFERENCE_WORDS = {
    "it",
    "that",
    "this",
    "they",
    "them",
    "those",
    "these",
    "there",
    "here",
    "one",
}


# =========================================================
# Generic continuation prefixes
# =========================================================

CONTINUATION_PREFIX_PATTERNS = [
    r"^\s*and\s+",
    r"^\s*also\s+",
    r"^\s*then\s+",
    r"^\s*so\s+",
    r"^\s*okay\s+",
    r"^\s*ok\s+",
    r"^\s*what\s+about\s+",
    r"^\s*how\s+about\s+",
]


# =========================================================
# LLM invocation
# =========================================================

def invoke_query_llm(
    prompt: str,
):
    """
    Invoke the conversation/query-understanding model.

    Wrapped so tests can monkeypatch resolver calls.
    """

    return query_llm.invoke(
        prompt
    )


# =========================================================
# Text helpers
# =========================================================

def _normalize_question(
    question: str,
) -> str:
    """
    Normalize whitespace without changing semantic content.
    """

    return re.sub(
        r"\s+",
        " ",
        str(
            question or ""
        ),
    ).strip()


def _matches_any(
    question: str,
    patterns: List[str],
) -> bool:
    """
    Return True when any regex matches.
    """

    return any(
        re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def _question_tokens(
    question: str,
) -> List[str]:
    """
    Return simple lexical tokens.
    """

    return re.findall(
        r"[a-zA-Z0-9']+",
        question.lower(),
    )


def _contains_reference_word(
    question: str,
) -> bool:
    """
    Detect generic conversational references.
    """

    return bool(
        set(
            _question_tokens(
                question
            )
        )
        & CONTEXT_REFERENCE_WORDS
    )


# =========================================================
# Question classification
# =========================================================

def _is_obviously_ambiguous(
    question: str,
) -> bool:
    """
    Detect references that are too vague to resolve safely.
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    if not normalized:
        return False

    return _matches_any(
        normalized,
        AMBIGUOUS_REFERENCE_PATTERNS,
    )


def _is_explicit_follow_up(
    question: str,
) -> bool:
    """
    Detect explicit conversational follow-up wording.
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    return _matches_any(
        normalized,
        EXPLICIT_FOLLOW_UP_PATTERNS,
    )


def _is_elliptical_follow_up(
    question: str,
) -> bool:
    """
    Detect continuation structures.
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    if _matches_any(
        normalized,
        ELLIPTICAL_FOLLOW_UP_PATTERNS,
    ):
        return True

    if _matches_any(
        normalized,
        CONTEXT_REFERENCE_PATTERNS,
    ):
        return True

    return False


def _is_conditional_continuation(
    question: str,
) -> bool:
    """
    Detect conditional/alternative structures.
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    return _matches_any(
        normalized,
        CONDITIONAL_CONTINUATION_PATTERNS,
    )


def _is_structurally_contextual(
    question: str,
) -> bool:
    """
    Detect question structures that commonly omit a parent subject.
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    if not normalized:
        return False

    if _matches_any(
        normalized,
        CONTEXTUAL_QUESTION_PATTERNS,
    ):
        return True

    if _contains_reference_word(
        normalized
    ):
        return True

    if _matches_any(
        normalized,
        CONTINUATION_PREFIX_PATTERNS,
    ):
        return True

    return False


# =========================================================
# Strong explicit subject
# =========================================================

def _has_strong_explicit_subject(
    question: str,
) -> bool:
    """
    Detect a strongly named subject.

    This is deliberately lightweight and generic.
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    if not normalized:
        return False

    strong_patterns = [
        # Academic programs / degrees.
        r"\bm\.?\s*sc\.?\b",
        r"\bm\.?\s*tech\.?\b",
        r"\bb\.?\s*tech\.?\b",
        r"\bph\.?\s*d\.?\b",
        r"\bphd\b",

        # Organizational structures.
        r"\bschool of\b",
        r"\bdepartment of\b",
        r"\bcentre of\b",
        r"\bcenter of\b",

        # Broad institutional domains.
        r"\bhostel\b",
        r"\blibrary\b",
        r"\badmission\b",
        r"\badmissions\b",
        r"\bresearch\b",
        r"\bplacement\b",
        r"\bscholarship\b",
        r"\bsyllabus\b",
        r"\bcurriculum\b",
    ]

    return _matches_any(
        normalized,
        strong_patterns,
    )


# =========================================================
# Context candidate
# =========================================================

def _is_context_candidate(
    question: str,
) -> bool:
    """
    Decide whether the latest question should be evaluated using
    conversational context.

    Priority:

        1. explicit ambiguity
        2. explicit conversational follow-up
        3. conditional / alternative continuation
        4. elliptical continuation
        5. strong explicit subject
        6. generic contextual shape
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    if not normalized:
        return False

    # Explicit ambiguity.
    if _is_obviously_ambiguous(
        normalized
    ):
        return True

    # Direct conversational construction.
    if _is_explicit_follow_up(
        normalized
    ):
        return True

    # Conditional/alternative continuation.
    if _is_conditional_continuation(
        normalized
    ):
        if _has_strong_explicit_subject(
            normalized
        ):
            return False

        return True

    # Elliptical continuation.
    if _is_elliptical_follow_up(
        normalized
    ):
        return True

    # Strong explicit subject means it can stand independently.
    if _has_strong_explicit_subject(
        normalized
    ):
        return False

    # Generic contextual structure.
    if _is_structurally_contextual(
        normalized
    ):
        return True

    return False


def _is_follow_up_like(
    question: str,
) -> bool:
    """
    Backward-compatible helper.
    """

    return _is_context_candidate(
        question
    )


# =========================================================
# History normalization
# =========================================================

def _normalize_history(
    messages: List[Dict[str, Any]] | None,
) -> List[Dict[str, str]]:
    """
    Normalize history into valid user/assistant messages.
    """

    if not messages:
        return []

    normalized_history: List[
        Dict[str, str]
    ] = []

    for message in messages:

        if not isinstance(
            message,
            dict,
        ):
            continue

        role = str(
            message.get(
                "role",
                "",
            )
        ).strip().lower()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        normalized_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized_history


def _recent_history(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Keep a bounded recent context window.
    """

    return messages[
        -MAX_HISTORY_MESSAGES:
    ]


def _build_history(
    messages: List[Dict[str, Any]],
) -> str:
    """
    Convert recent history into resolver context.
    """

    recent = _recent_history(
        _normalize_history(
            messages
        )
    )

    lines: List[str] = []

    for message in recent:

        lines.append(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )

    return "\n".join(
        lines
    )


# =========================================================
# JSON extraction
# =========================================================

def _extract_json(
    value: Any,
) -> Dict[str, Any]:
    """
    Extract JSON from model output.
    """

    if value is None:
        raise ValueError(
            "Resolver returned no response."
        )

    if hasattr(
        value,
        "content",
    ):
        text = value.content
    else:
        text = value

    text = str(
        text or ""
    ).strip()

    if not text:
        raise ValueError(
            "Resolver returned an empty response."
        )

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        pass

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if start == -1 or end == -1:
        raise ValueError(
            "Resolver returned no JSON object."
        )

    parsed = json.loads(
        text[
            start:
            end + 1
        ]
    )

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "Resolver JSON was not an object."
        )

    return parsed


# =========================================================
# Result builder
# =========================================================

def _result(
    *,
    mode: str,
    question: str,
    active_topic: str = "",
    active_entity: str = "",
    active_subtopic: str = "",
    conversation_path: str = "",
    resolver_called: bool = False,
    diagnostic_reason: str = "",
) -> Dict[str, Any]:
    """
    Build normalized resolver output.

    active_subtopic and conversation_path provide hierarchical context
    for downstream components while remaining optional/backward compatible.
    """

    if mode not in VALID_MODES:
        mode = "ambiguous"

    if (
        diagnostic_reason
        not in VALID_DIAGNOSTIC_REASONS
    ):
        diagnostic_reason = ""

    return {
        "mode": mode,
        "resolved_question": _normalize_question(
            question
        ),
        "active_topic": _normalize_question(
            active_topic
        ),
        "active_entity": _normalize_question(
            active_entity
        ),
        "active_subtopic": _normalize_question(
            active_subtopic
        ),
        "conversation_path": _normalize_question(
            conversation_path
        ),
        "resolver_called": bool(
            resolver_called
        ),
        "diagnostic_reason": diagnostic_reason,
    }


# =========================================================
# Parent-context safety
# =========================================================

def _requires_parent_context(
    question: str,
) -> bool:
    """
    Detect wording that explicitly depends on a previous concept.
    """

    normalized = (
        question
        .lower()
        .strip()
    )

    patterns = [
        r"\bthat route\b",
        r"\bthis route\b",
        r"\bthe other route\b",
        r"\banother route\b",
        r"\bthat option\b",
        r"\bthis option\b",
        r"\bthe other option\b",
        r"\banother option\b",
        r"\bthat program\b",
        r"\bthis program\b",
        r"\bthe other program\b",
        r"\banother program\b",
        r"\bthat degree\b",
        r"\bthis degree\b",
        r"\bthe other degree\b",
        r"\banother degree\b",
        r"\bthat requirement\b",
        r"\bthis requirement\b",
        r"\bthose requirements\b",
        r"\bfor that\b",
        r"\bfor this\b",
        r"\bwith that\b",
        r"\bwith this\b",
        r"\binstead\b",
        r"\balternative\b",
    ]

    return _matches_any(
        normalized,
        patterns,
    )


def _history_contains_parent_signal(
    history_text: str,
) -> bool:
    """
    Lightweight parent-context safety check.
    """

    normalized = (
        history_text
        .lower()
    )

    parent_signals = (
        "ph.d.",
        "ph.d",
        "phd",
        "admission",
        "eligibility",
        "application",
        "program",
        "programme",
        "degree",
        "hostel",
        "research",
        "department",
        "school",
        "course",
        "fee",
        "fees",
        "percentage",
        "marks",
        "gate",
    )

    return any(
        signal in normalized
        for signal in parent_signals
    )


def _resolved_question_preserves_context(
    resolved_question: str,
    history_text: str,
    active_topic: str,
    active_entity: str,
    active_subtopic: str,
) -> bool:
    """
    Verify that a contextual rewrite contains enough semantic information.
    """

    resolved = (
        resolved_question
        .lower()
        .strip()
    )

    if not resolved:
        return False

    if (
        active_topic
        or active_entity
        or active_subtopic
    ):
        return True

    if not _history_contains_parent_signal(
        history_text
    ):
        return True

    return len(
        _question_tokens(
            resolved
        )
    ) >= 5


# =========================================================
# Final resolver prompt
# =========================================================

RESOLVER_PROMPT = """
You are the conversation resolver for a college AI assistant.

Your ONLY task is to determine whether the latest user question depends
on recent conversation context and, when necessary, rewrite it into a
self-contained retrieval question.

You DO NOT answer the question.

You DO NOT invent facts.

You DO NOT add institutional facts that are not present in the
conversation.

Conversation is context only. It is NOT evidence.

------------------------------------------------------------
CORE CONVERSATIONAL MODEL
------------------------------------------------------------

Treat the conversation as a hierarchical state:

    main subject
        ↓
    active subtopic
        ↓
    active detail

For example:

    Ph.D. admission
        ↓
    bachelor's route
        ↓
    percentage
        ↓
    GATE

Another example:

    hostel
        ↓
    short-term
        ↓
    single occupancy
        ↓
    bedding

Another:

    Electrical Engineering research
        ↓
    control
        ↓
    robotics

When a user asks a follow-up, preserve the deepest relevant context
from the recent conversation.

------------------------------------------------------------
ALLOWED MODES
------------------------------------------------------------

- standalone
- topic_switch
- follow_up
- ambiguous

------------------------------------------------------------
RULES
------------------------------------------------------------

1. A genuinely self-contained question is "standalone".

2. A clearly new subject is "topic_switch".

3. A question that depends on recent conversation is "follow_up".

4. A grammatically complete question can still be a follow-up.

Examples:

    "What percentage is needed?"
    "What entrance exam is required?"
    "Which one is related to control?"
    "How long is it?"

5. Explicit follow-up constructions must use recent context.

Examples:

    "What about research?"
    "What about the bachelor's route?"
    "What about single occupancy?"

6. Conditional and alternative questions can be follow-ups.

Examples:

    "What if I choose the other route?"
    "What happens if I switch?"
    "Would the alternative option work?"
    "What if I focus on control instead?"
    "What if I stay for several weeks instead?"

7. Nested follow-ups MUST preserve the immediate subtopic and its parent.

Example:

Conversation:
USER:
What are the regular Ph.D. eligibility requirements?

USER:
What about the bachelor's route?

Latest:
What percentage is needed for that route?

Correct:
{
  "mode": "follow_up",
  "resolved_question":
    "What percentage is required for the bachelor's route for regular Ph.D. admission?",
  "active_topic": "admission",
  "active_entity": "Ph.D.",
  "active_subtopic": "bachelor's route",
  "conversation_path":
    "Ph.D. admission > bachelor's route > percentage"
}

8. Do not collapse a nested conversation back to only the broad topic.

Wrong:
    "What percentage is required for Ph.D. admission?"

when the actual active subject is the bachelor's route.

9. If the latest question explicitly names a different independent
subject, treat it as a topic switch.

Example:

Previous:
    "Tell me about regular Ph.D. admission."

Latest:
    "What if I apply for M.Sc. admission instead?"

Correct:
{
  "mode": "topic_switch",
  "resolved_question":
    "What if I apply for M.Sc. admission instead?",
  "active_topic": "admission",
  "active_entity": "M.Sc.",
  "active_subtopic": "",
  "conversation_path":
    "M.Sc. admission"
}

10. Preserve academic-program distinctions:

    B.Tech
    M.Tech
    M.Sc.
    Ph.D.

11. Preserve organizational distinctions:

    institute
    school
    department
    centre/center

12. Preserve admission-mode distinctions:

    regular
    part-time
    sponsored
    external

13. Preserve user-selected alternatives.

Example:

    "What about the bachelor's route?"
    then:
    "What percentage is needed for that route?"

The second question refers to the bachelor's route, not generic admission.

14. If the conversation moves from:

    main subject → subtopic

then later:

    "What about that?"

the resolver should use the most recent valid referent, not an older
broad topic.

15. If a vague reference cannot be resolved confidently, use "ambiguous".

16. Never invent fees, marks, eligibility requirements, dates, programs,
rules, or other factual information.

17. Keep the rewritten question concise and retrieval-friendly.

18. Return JSON only.

------------------------------------------------------------
EXAMPLE 1 — NESTED ADMISSION CONTEXT
------------------------------------------------------------

USER:
What are the regular Ph.D. eligibility requirements?

ASSISTANT:
There are multiple qualifying routes.

USER:
What about the bachelor's route?

ASSISTANT:
The four-year bachelor's route is one route.

LATEST:
What percentage is needed for that route?

Correct:
{
  "mode": "follow_up",
  "resolved_question":
    "What percentage is required for the bachelor's route for regular Ph.D. admission?",
  "active_topic": "admission",
  "active_entity": "Ph.D.",
  "active_subtopic": "bachelor's route",
  "conversation_path":
    "Ph.D. admission > bachelor's route > percentage"
}

------------------------------------------------------------
EXAMPLE 2 — NESTED GATE CONTEXT
------------------------------------------------------------

USER:
What are the regular Ph.D. eligibility requirements?

ASSISTANT:
...

USER:
What about the bachelor's route?

ASSISTANT:
...

USER:
What percentage is needed for that route?

ASSISTANT:
...

LATEST:
Does GATE apply to that route?

Correct:
{
  "mode": "follow_up",
  "resolved_question":
    "Does GATE apply to the bachelor's route for regular Ph.D. admission?",
  "active_topic": "admission",
  "active_entity": "Ph.D.",
  "active_subtopic": "bachelor's route",
  "conversation_path":
    "Ph.D. admission > bachelor's route > GATE"
}

------------------------------------------------------------
EXAMPLE 3 — TOPIC SWITCH
------------------------------------------------------------

USER:
What are the regular Ph.D. eligibility requirements?

LATEST:
What hostel accommodation is available?

Correct:
{
  "mode": "topic_switch",
  "resolved_question":
    "What hostel accommodation is available?",
  "active_topic": "hostel",
  "active_entity": "hostel",
  "active_subtopic": "",
  "conversation_path": "hostel"
}

------------------------------------------------------------
EXAMPLE 4 — RETURN TO OLD TOPIC
------------------------------------------------------------

Conversation contains:
Ph.D. admission
then:
hostel
then:
hostel single occupancy

LATEST:
Going back to the Ph.D. question, what about GATE?

Correct:
{
  "mode": "follow_up",
  "resolved_question":
    "What are the GATE requirements for regular Ph.D. admission?",
  "active_topic": "admission",
  "active_entity": "Ph.D.",
  "active_subtopic": "GATE",
  "conversation_path":
    "Ph.D. admission > GATE"
}

------------------------------------------------------------
EXAMPLE 5 — HOSTEL HIERARCHY
------------------------------------------------------------

USER:
What hostel accommodation is available?

ASSISTANT:
...

USER:
What are the short-term rates?

LATEST:
What about single occupancy?

Correct:
{
  "mode": "follow_up",
  "resolved_question":
    "What are the short-term hostel rates for single occupancy?",
  "active_topic": "hostel",
  "active_entity": "hostel",
  "active_subtopic": "short-term > single occupancy",
  "conversation_path":
    "hostel > short-term > single occupancy"
}

------------------------------------------------------------
EXAMPLE 6 — SCHOOL BOUNDARY
------------------------------------------------------------

USER:
What programs are available in the School of Artificial Intelligence
and Data Science?

ASSISTANT:
Several programs are available.

LATEST:
What about research?

Correct:
{
  "mode": "follow_up",
  "resolved_question":
    "What research is being done in the School of Artificial Intelligence and Data Science?",
  "active_topic": "research",
  "active_entity":
    "School of Artificial Intelligence and Data Science",
  "active_subtopic": "",
  "conversation_path":
    "School of Artificial Intelligence and Data Science > research"
}

------------------------------------------------------------
EXAMPLE 7 — VAGUE REFERENCE
------------------------------------------------------------

USER:
We discussed several different routes.

LATEST:
Can I take that instead?

Correct:
{
  "mode": "ambiguous",
  "resolved_question":
    "Can I take that instead?",
  "active_topic": "",
  "active_entity": "",
  "active_subtopic": "",
  "conversation_path": ""
}

Recent conversation:
<HISTORY>

Latest user question:
<QUESTION>
"""


# =========================================================
# Main resolver
# =========================================================

def resolve_conversation(
    question: str,
    chat_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Resolve the latest user question against recent session context.
    """

    normalized_question = _normalize_question(
        question
    )

    history = _normalize_history(
        chat_history
    )

    # -----------------------------------------------------
    # Empty question.
    # -----------------------------------------------------

    if not normalized_question:

        return _result(
            mode="standalone",
            question="",
            diagnostic_reason="no_history",
        )

    # -----------------------------------------------------
    # No session history.
    # -----------------------------------------------------

    if not history:

        return _result(
            mode="standalone",
            question=normalized_question,
            diagnostic_reason="no_history",
        )

    # -----------------------------------------------------
    # Obviously ambiguous.
    # -----------------------------------------------------

    if _is_obviously_ambiguous(
        normalized_question
    ):

        return _result(
            mode="ambiguous",
            question=normalized_question,
            diagnostic_reason="obviously_ambiguous",
        )

    # -----------------------------------------------------
    # Determine whether conversational resolution is needed.
    # -----------------------------------------------------

    if not _is_context_candidate(
        normalized_question
    ):

        return _result(
            mode="topic_switch",
            question=normalized_question,
            diagnostic_reason="clearly_self_contained",
        )

    # -----------------------------------------------------
    # Build recent conversation.
    # -----------------------------------------------------

    history_text = _build_history(
        history
    )

    if not history_text:

        return _result(
            mode="ambiguous",
            question=normalized_question,
            diagnostic_reason="resolver_failure",
        )

    # -----------------------------------------------------
    # Build resolver prompt.
    # -----------------------------------------------------

    prompt = (
        RESOLVER_PROMPT
        .replace(
            "<HISTORY>",
            history_text,
        )
        .replace(
            "<QUESTION>",
            normalized_question,
        )
    )

    # -----------------------------------------------------
    # Call resolver model.
    # -----------------------------------------------------

    try:

        response = invoke_query_llm(
            prompt
        )

    except Exception:

        return _result(
            mode="ambiguous",
            question=normalized_question,
            resolver_called=True,
            diagnostic_reason="resolver_failure",
        )

    # -----------------------------------------------------
    # Parse model result.
    # -----------------------------------------------------

    try:

        if hasattr(
            response,
            "content",
        ):

            response_content = (
                response.content
            )

        else:

            response_content = response

        data = _extract_json(
            response_content
        )

    except Exception:

        return _result(
            mode="ambiguous",
            question=normalized_question,
            resolver_called=True,
            diagnostic_reason="invalid_model_output",
        )

    # -----------------------------------------------------
    # Normalize resolver fields.
    # -----------------------------------------------------

    mode = str(
        data.get(
            "mode",
            "ambiguous",
        )
    ).strip().lower()

    if mode not in VALID_MODES:

        mode = "ambiguous"

    resolved_question = _normalize_question(
        data.get(
            "resolved_question",
            normalized_question,
        )
    )

    active_topic = _normalize_question(
        data.get(
            "active_topic",
            "",
        )
    )

    active_entity = _normalize_question(
        data.get(
            "active_entity",
            "",
        )
    )

    active_subtopic = _normalize_question(
        data.get(
            "active_subtopic",
            "",
        )
    )

    conversation_path = _normalize_question(
        data.get(
            "conversation_path",
            "",
        )
    )

    if not resolved_question:

        resolved_question = (
            normalized_question
        )

    # -----------------------------------------------------
    # Deterministic ambiguity guard.
    # -----------------------------------------------------

    if _is_obviously_ambiguous(
        normalized_question
    ):

        return _result(
            mode="ambiguous",
            question=normalized_question,
            resolver_called=True,
            diagnostic_reason="obviously_ambiguous",
        )

    # -----------------------------------------------------
    # Explicitly self-contained conditional question.
    # -----------------------------------------------------

    if (
        _is_conditional_continuation(
            normalized_question
        )
        and
        _has_strong_explicit_subject(
            normalized_question
        )
    ):

        return {
            "mode": "topic_switch",
            "resolved_question": normalized_question,
            "active_topic": active_topic,
            "active_entity": active_entity,
            "active_subtopic": "",
            "conversation_path": (
                conversation_path
                or active_entity
                or active_topic
            ),
            "resolver_called": False,
            "diagnostic_reason": "clearly_self_contained",
        }

    # -----------------------------------------------------
    # Validate contextual rewrites.
    # -----------------------------------------------------

    if (
        mode == "follow_up"
        and
        _requires_parent_context(
            normalized_question
        )
    ):

        if not _resolved_question_preserves_context(
            resolved_question=resolved_question,
            history_text=history_text,
            active_topic=active_topic,
            active_entity=active_entity,
            active_subtopic=active_subtopic,
        ):

            return _result(
                mode="ambiguous",
                question=normalized_question,
                resolver_called=True,
                diagnostic_reason="invalid_model_output",
            )

    # -----------------------------------------------------
    # Follow-up.
    # -----------------------------------------------------

    if mode == "follow_up":

        return {
            "mode": "follow_up",
            "resolved_question": resolved_question,
            "active_topic": active_topic,
            "active_entity": active_entity,
            "active_subtopic": active_subtopic,
            "conversation_path": conversation_path,
            "resolver_called": True,
            "diagnostic_reason": "resolver_success",
        }

    # -----------------------------------------------------
    # Standalone / topic switch.
    # -----------------------------------------------------

    if mode in {
        "standalone",
        "topic_switch",
    }:

        return {
            "mode": mode,
            "resolved_question": normalized_question,
            "active_topic": active_topic,
            "active_entity": active_entity,
            "active_subtopic": active_subtopic,
            "conversation_path": conversation_path,
            "resolver_called": True,
            "diagnostic_reason": "resolver_success",
        }

    # -----------------------------------------------------
    # Safe ambiguous fallback.
    # -----------------------------------------------------

    return {
        "mode": "ambiguous",
        "resolved_question": normalized_question,
        "active_topic": "",
        "active_entity": "",
        "active_subtopic": "",
        "conversation_path": "",
        "resolver_called": True,
        "diagnostic_reason": "resolver_success",
    }