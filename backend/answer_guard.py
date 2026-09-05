"""
IIT Jodhpur V1 — Answer Guard

Purpose
-------
Apply lightweight deterministic post-processing to model-generated
answers before they reach the user.

Design principles
-----------------
- Preserve useful answers whenever possible.
- Remove internal retrieval/debug leakage.
- Remove unsupported generic escalation language.
- Never invent replacement facts.
- Use the exact unknown response only when the generated answer
  is empty or becomes unusable after sanitization.

This module intentionally does NOT attempt to determine full
semantic evidence sufficiency. That belongs to the later
evidence-validation phase.
"""

import re


# =========================================================
# Public constants
# =========================================================

UNKNOWN_RESPONSE = (
    "I'm sorry, I don't know based on the available information."
)


# =========================================================
# Internal retrieval leakage
# =========================================================

INTERNAL_REFERENCE_PATTERNS = [
    r"\bDocument\s+\d+\b",
    r"\bDoc(?:ument)?\s*#?\s*\d+\b",
    r"\bRRF\s+score\b",
    r"\bchunk\s+id\b",
    r"\bchunk_id\b",
    r"\bretrieval\s+rank\b",
    r"\bretrieval\s+score\b",
    r"\bsource\s+path\b",
]


# =========================================================
# Internal-reference sentence patterns
# =========================================================

INTERNAL_REFERENCE_SENTENCE_PATTERNS = [
    r"[^.!?]*\baccording to document\s+\d+\b[^.!?]*[.!?]?",
    r"[^.!?]*\bbased on (?:the )?(?:details|information) provided in document\s+\d+\b[^.!?]*[.!?]?",
    r"[^.!?]*\bthis information is (?:directly )?supported by (?:the )?retrieved context(?: in document\s+\d+)?[^.!?]*[.!?]?",
    r"[^.!?]*\bprovided in document\s+\d+\b[^.!?]*[.!?]?",
    r"[^.!?]*\bmentioned in document\s+\d+\b[^.!?]*[.!?]?",
    r"[^.!?]*\bthis information is based on document\s+\d+\b[^.!?]*[.!?]?",
]


# =========================================================
# Unsupported escalation patterns
# =========================================================

CONTACT_SENTENCE_PATTERNS = [
    r"[^.!?]*\bcontact the relevant office\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact the appropriate office\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact the admissions office\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact the office of admissions\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact the academic office\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact your department\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact the department\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact student guide\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact swc\b[^.!?]*[.!?]?",
    r"[^.!?]*\bcontact hwc\b[^.!?]*[.!?]?",
]


# =========================================================
# Detection helpers
# =========================================================

def contains_internal_leak(answer: str) -> bool:
    """
    Return True when an answer contains obvious internal
    retrieval/debug terminology.
    """

    text = str(answer or "")

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in INTERNAL_REFERENCE_PATTERNS
    )


def contains_contact_fallback(answer: str) -> bool:
    """
    Return True when the answer contains a generic unsupported
    escalation/contact recommendation.
    """

    text = str(answer or "")

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in CONTACT_SENTENCE_PATTERNS
    )


# =========================================================
# Sanitization
# =========================================================

def _remove_internal_reference_sentences(
    answer: str,
) -> str:
    """
    Remove sentences that explicitly expose internal retrieval
    references.
    """

    cleaned = answer

    for pattern in INTERNAL_REFERENCE_SENTENCE_PATTERNS:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


def _remove_contact_fallback_sentences(
    answer: str,
) -> str:
    """
    Remove unsupported generic escalation sentences.

    This does not remove legitimate institutional contact facts
    such as an official phone number or named office when the
    answer itself contains factual information supported by the
    retrieved context.
    """

    cleaned = answer

    for pattern in CONTACT_SENTENCE_PATTERNS:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


def _remove_remaining_internal_labels(
    answer: str,
) -> str:
    """
    Remove isolated internal labels that survive sentence-level
    sanitization.
    """

    cleaned = answer

    for pattern in INTERNAL_REFERENCE_PATTERNS:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


def _normalize_whitespace(
    answer: str,
) -> str:
    """
    Collapse whitespace introduced by sanitization.
    """

    return re.sub(
        r"\s+",
        " ",
        answer,
    ).strip()


# =========================================================
# Main guard
# =========================================================

def guard_answer(answer: str) -> dict:
    """
    Apply lightweight deterministic output protection.

    Returns:
        {
            "answer": <safe answer>,
            "status": <guard status>,
            "reason": <guard reason>,
        }

    Status values:
        clean
        sanitized_internal_reference
        sanitized_contact_fallback
        sanitized_multiple
        empty_answer
    """

    cleaned = str(answer or "").strip()

    # -----------------------------------------------------
    # Empty answer
    # -----------------------------------------------------

    if not cleaned:
        return {
            "answer": UNKNOWN_RESPONSE,
            "status": "empty_answer",
            "reason": "Generated answer was empty.",
        }

    original = cleaned

    # -----------------------------------------------------
    # Remove internal retrieval references
    # -----------------------------------------------------

    cleaned = _remove_internal_reference_sentences(
        cleaned
    )

    internal_changed = (
        cleaned.strip() != original.strip()
    )

    # -----------------------------------------------------
    # Remove unsupported generic escalation
    # -----------------------------------------------------

    before_contact = cleaned

    cleaned = _remove_contact_fallback_sentences(
        cleaned
    )

    contact_changed = (
        cleaned.strip() != before_contact.strip()
    )

    # -----------------------------------------------------
    # Remove remaining internal labels
    # -----------------------------------------------------

    before_labels = cleaned

    cleaned = _remove_remaining_internal_labels(
        cleaned
    )

    labels_changed = (
        cleaned.strip() != before_labels.strip()
    )

    # -----------------------------------------------------
    # Normalize whitespace
    # -----------------------------------------------------

    cleaned = _normalize_whitespace(
        cleaned
    )

    # -----------------------------------------------------
    # If sanitization removed everything
    # -----------------------------------------------------

    if not cleaned:
        return {
            "answer": UNKNOWN_RESPONSE,
            "status": "empty_answer",
            "reason": "Answer became empty after sanitization.",
        }

    # -----------------------------------------------------
    # Determine guard status
    # -----------------------------------------------------

    if (internal_changed and contact_changed) or (
        internal_changed and labels_changed
    ) or (
        contact_changed and labels_changed
    ):
        status = "sanitized_multiple"
        reason = (
            "Internal retrieval information and/or unsupported "
            "escalation language was removed."
        )

    elif internal_changed or labels_changed:
        status = "sanitized_internal_reference"
        reason = (
            "Internal retrieval information was removed."
        )

    elif contact_changed:
        status = "sanitized_contact_fallback"
        reason = (
            "Unsupported generic escalation was removed."
        )

    else:
        status = "clean"
        reason = (
            "No output-contract violation detected."
        )

    return {
        "answer": cleaned,
        "status": status,
        "reason": reason,
    }