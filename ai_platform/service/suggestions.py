"""Lightweight, Grounded Follow-up Suggestion Engine.

Generates concise, context-aware follow-up question suggestions for students
based strictly on the current question, verified answer, and available source metadata.

Constraints:
- Zero additional LLM invocations (preserves latency and compute budgets).
- At most 3 distinct, concise suggestions.
- Suppressed for safe refusals, errors, empty evidence, and off-topic queries.
- Portable and institution-independent.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# Canonical educational degree / program markers
_PROGRAM_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bm\.?\s*tech\b", re.I), "M.Tech."),
    (re.compile(r"\bph\.?\s*d\.?\b|\bphd\b", re.I), "Ph.D."),
    (re.compile(r"\bm\.?\s*b\.?\s*a\.?\b|\bmba\b", re.I), "MBA"),
    (re.compile(r"\bm\.?\s*sc\.?\b", re.I), "M.Sc."),
    (re.compile(r"\bb\.?\s*tech\b", re.I), "B.Tech."),
    (re.compile(r"\bb\.?\s*sc\b", re.I), "B.Sc."),
)

_FACILITY_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bhostel\b", re.I), "Hostel"),
    (re.compile(r"\bmess\b", re.I), "Mess"),
    (re.compile(r"\blibrary\b", re.I), "Library"),
    (re.compile(r"\bgymkhana\b", re.I), "Gymkhana"),
    (re.compile(r"\bplacement\b|\binternship\b", re.I), "Placements"),
)

# Topic-oriented suggestion templates
_TOPIC_TEMPLATES: dict[str, list[str]] = {
    "fee": [
        "What are the hostel and mess charges?",
        "Are fee waivers or scholarships available for {entity}?",
        "What is the fee structure for sponsored or part-time {entity}?",
    ],
    "eligibility": [
        "What is the tuition fee for {entity}?",
        "What financial assistance or fellowship is offered for {entity}?",
        "What is the admission and selection procedure for {entity}?",
    ],
    "leave": [
        "What are the fellowship and stipend rules for {entity}?",
        "What are the academic progression rules for {entity}?",
        "What is the policy for attending conferences during {entity}?",
    ],
    "facility": [
        "What are the hostel room rent and mess fees?",
        "Where is this facility located on campus?",
        "What are the operating hours and rules?",
    ],
    "general": [
        "What are the eligibility criteria for {entity}?",
        "What is the fee structure for {entity}?",
        "What financial assistance or stipend is available for {entity}?",
    ],
}


def _detect_entity(text: str) -> str:
    """Detect dominant degree, program, or campus entity from text."""
    for pattern, name in _PROGRAM_PATTERNS:
        if pattern.search(text):
            return name
    for pattern, name in _FACILITY_PATTERNS:
        if pattern.search(text):
            return name
    return ""


def _detect_topic(text: str) -> str:
    """Identify query or answer topic focus."""
    low = text.casefold()
    if any(k in low for k in ("leave", "casual leave", "medical leave", "vacation")):
        return "leave"
    if any(k in low for k in ("fee", "fees", "tuition", "charges", "rent", "payment")):
        return "fee"
    if any(k in low for k in ("eligibility", "criteria", "cutoff", "percentage", "cgpa", "qualification")):
        return "eligibility"
    if any(k in low for k in ("hostel", "mess", "library", "gymkhana", "dining")):
        return "facility"
    return "general"


def _is_redundant(candidate: str, question: str, existing: Sequence[str]) -> bool:
    """Check if candidate question is too close to the current question or existing suggestions."""
    c_clean = " ".join(re.findall(r"\w+", candidate.casefold()))
    q_clean = " ".join(re.findall(r"\w+", question.casefold()))
    if c_clean == q_clean:
        return True
    # Word overlap check
    c_words = set(c_clean.split()) - {"what", "is", "are", "the", "for", "in", "and", "or", "to", "how", "do"}
    q_words = set(q_clean.split()) - {"what", "is", "are", "the", "for", "in", "and", "or", "to", "how", "do"}
    if c_words and q_words and (c_words == q_words or (len(c_words) >= 3 and len(c_words & q_words) >= len(c_words))):
        return True
    for item in existing:
        i_clean = " ".join(re.findall(r"\w+", item.casefold()))
        if c_clean == i_clean:
            return True
    return False


def generate_follow_up_suggestions(
    question: str,
    *,
    answer: str = "",
    sources: Sequence[dict[str, Any]] | None = None,
    is_safe_refusal: bool = False,
    evidence_status: str = "supported",
    max_suggestions: int = 3,
) -> list[str]:
    """Generate up to `max_suggestions` lightweight, grounded follow-up suggestions.

    Returns an empty list when the answer is an intentional abstention/safe refusal,
    when evidence is empty, or when an error occurred.
    """
    # Safe refusal or error check: never emit suggestions after a refusal or failure
    if is_safe_refusal or evidence_status in {"empty", "unsupported", "error", "empty_query", "service_error"}:
        return []

    clean_q = " ".join(str(question or "").strip().split())
    clean_a = " ".join(str(answer or "").strip().split())
    if not clean_q or not clean_a or len(clean_a) < 15:
        return []

    # Detect entity from question, answer, and sources
    combined_context = f"{clean_q} {clean_a}"
    if sources:
        for s in sources[:3]:
            combined_context += f" {s.get('heading', '')} {s.get('source', '')}"

    entity = _detect_entity(combined_context) or "this programme"
    topic = _detect_topic(clean_q)

    # Candidate suggestions from templates
    templates = _TOPIC_TEMPLATES.get(topic, _TOPIC_TEMPLATES["general"])
    results: list[str] = []

    for template in templates:
        candidate = template.format(entity=entity)
        if not _is_redundant(candidate, clean_q, results):
            results.append(candidate)
        if len(results) >= max_suggestions:
            break

    # If we need more, check other general templates
    if len(results) < max_suggestions:
        for template in _TOPIC_TEMPLATES["general"]:
            candidate = template.format(entity=entity)
            if not _is_redundant(candidate, clean_q, results):
                results.append(candidate)
            if len(results) >= max_suggestions:
                break

    return results[:max_suggestions]


__all__ = ["generate_follow_up_suggestions"]
