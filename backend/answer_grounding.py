"""
IIT Jodhpur V1 — Phase 5.5
Evidence -> Answer Grounding

Purpose
-------
Detect high-risk semantic relationships in generated answers that are
not safely supported by supplied evidence.

This complements:
    - answer_claim_audit.py
    - supported_claim_preservation.py
    - answer_guard.py

It does NOT:
    - retrieve documents
    - call an LLM
    - decide institutional policy
    - modify the answer
    - perform general-purpose semantic reasoning

Current high-risk relationships:
    1. eligibility conclusions
    2. personal exemption conclusions
    3. unsupported causal conclusions
    4. unsupported personal/exact cost interpretations

Important V1 behavior
---------------------
A valid arithmetic derivation is allowed when the evidence supplies
the rate and the answer explicitly provides the duration and resulting
total.

Example:

    Evidence:
        "Single occupancy hostel room rent is ₹500 per day."

    Answer:
        "At ₹500 per day, five days would total ₹2,500."

    -> grounded

Conversational number words such as:

    five days
    ten days
    three months

are recognized where needed by the grounding logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# =========================================================
# Result model
# =========================================================

@dataclass(frozen=True)
class GroundingIssue:
    """One high-risk unsupported relationship."""

    issue_type: str
    answer_claim: str
    evidence_basis: str
    reason: str


# =========================================================
# Text helpers
# =========================================================

def _normalize(
    text: str,
) -> str:
    """
    Normalize whitespace without changing semantic content.
    """
    return " ".join(
        str(text or "").strip().split()
    )


def _lower(
    text: str,
) -> str:
    """
    Lowercase normalized text.
    """
    return _normalize(
        text
    ).casefold()


def _sentences(
    text: str,
) -> tuple[str, ...]:
    """
    Split generated text into sentence-like units.
    """
    value = _normalize(
        text
    )

    if not value:
        return ()

    return tuple(
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            value,
        )
        if part.strip()
    )


# =========================================================
# Conversational number words
# =========================================================

NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
}


# =========================================================
# High-risk answer patterns
# =========================================================

ELIGIBILITY_PATTERNS = (
    re.compile(
        r"\b(?:you|your)\b"
        r".{0,120}"
        r"\b(?:are|would be|will be|should be|may be|might be)\b"
        r".{0,80}"
        r"\b(?:eligible|qualified)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:therefore|thus|hence|so)\b"
        r".{0,120}"
        r"\b(?:you|your)\b"
        r".{0,80}"
        r"\b(?:eligible|qualified)\b",
        re.IGNORECASE,
    ),
)


EXEMPTION_PATTERNS = (
    re.compile(
        r"\b(?:you|your)\b"
        r".{0,140}"
        r"\b(?:are|would be|will be|should be|may be|might be)\b"
        r".{0,80}"
        r"\b(?:exempt|exempted)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you|your)\b"
        r".{0,140}"
        r"\b(?:qualify|qualifies)\b"
        r".{0,80}"
        r"\bexemption\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:therefore|thus|hence|so)\b"
        r".{0,120}"
        r"\b(?:you|your)\b"
        r".{0,80}"
        r"\b(?:exempt|exempted|exemption)\b",
        re.IGNORECASE,
    ),
)


PERSONAL_COST_PATTERNS = (
    re.compile(
        r"\b(?:your|you will|you would)\b"
        r".{0,100}"
        r"\b(?:exact|total)\b"
        r".{0,100}"
        r"(?:₹|rs\.?|inr)?\s*[\d,]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:your|you will|you would)\b"
        r".{0,120}"
        r"\b(?:cost|costs|fee|fees|pay|paying|charge|charges)\b"
        r".{0,100}"
        r"(?:₹|rs\.?|inr)?\s*[\d,]+",
        re.IGNORECASE,
    ),
)


PERSONAL_TOTAL_PATTERNS = (
    re.compile(
        r"\b(?:exact|total)\b"
        r".{0,100}"
        r"(?:₹|rs\.?|inr)?\s*[\d,]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:₹|rs\.?|inr)?\s*[\d,]+"
        r".{0,100}"
        r"\bfor the entire stay\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:flat|fixed)\s+(?:fee|charge|rate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:regardless|irrespective)\s+of\b",
        re.IGNORECASE,
    ),
)


CAUSAL_PATTERNS = (
    re.compile(
        r"\b(?:because|since)\b"
        r".{0,220}"
        r"\b(?:therefore|thus|hence|so)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:because|since)\b"
        r".{0,220}"
        r",\s*(?:you|your)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:therefore|thus|hence|so)\b"
        r".{0,140}"
        r"\b(?:you|your)\b",
        re.IGNORECASE,
    ),
)


# =========================================================
# Numeric patterns
# =========================================================

MONEY_PATTERN = re.compile(
    r"""
    (?:
        ₹\s*(?P<rupee>[0-9][0-9,]*(?:\.[0-9]+)?)
        |
        \b(?:rs\.?|inr)\s*(?P<rs>[0-9][0-9,]*(?:\.[0-9]+)?)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


DAILY_RATE_PATTERN = re.compile(
    r"""
    (?:
        ₹\s*(?P<rupee>[0-9][0-9,]*(?:\.[0-9]+)?)
        |
        \b(?:rs\.?|inr)\s*(?P<rs>[0-9][0-9,]*(?:\.[0-9]+)?)
    )
    \s*(?:per\s+day|/\s*day)
    """,
    re.IGNORECASE | re.VERBOSE,
)


DURATION_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)


# =========================================================
# Numeric extraction
# =========================================================

def _extract_money_values(
    text: str,
) -> tuple[float, ...]:
    """
    Extract monetary values from text.
    """
    values: list[float] = []

    for match in MONEY_PATTERN.finditer(
        text
    ):
        raw = (
            match.group("rupee")
            or match.group("rs")
        )

        if raw is None:
            continue

        values.append(
            float(
                raw.replace(
                    ",",
                    "",
                )
            )
        )

    return tuple(
        values
    )


def _extract_daily_rates(
    text: str,
) -> tuple[float, ...]:
    """
    Extract explicit per-day monetary rates.
    """
    rates: list[float] = []

    for match in DAILY_RATE_PATTERN.finditer(
        text
    ):
        raw = (
            match.group("rupee")
            or match.group("rs")
        )

        if raw is None:
            continue

        rates.append(
            float(
                raw.replace(
                    ",",
                    "",
                )
            )
        )

    return tuple(
        rates
    )


def _extract_day_durations(
    text: str,
) -> tuple[float, ...]:
    """
    Extract explicit day durations.

    Supports both numeric and conversational forms:

        5 days
        five days
        10 days
        ten days
        twenty days
    """
    normalized = _lower(
        text
    )

    results: list[float] = []

    # -----------------------------------------------------
    # Numeric form.
    # -----------------------------------------------------

    for number, unit in DURATION_PATTERN.findall(
        normalized
    ):
        if unit.casefold() in {
            "day",
            "days",
        }:
            results.append(
                float(number)
            )

    # -----------------------------------------------------
    # Word-number form.
    # -----------------------------------------------------

    word_pattern = re.compile(
        r"\b("
        + "|".join(
            re.escape(word)
            for word in NUMBER_WORDS
        )
        + r")\s+days?\b",
        re.IGNORECASE,
    )

    for word in word_pattern.findall(
        normalized
    ):
        value = NUMBER_WORDS.get(
            word.casefold()
        )

        if value is not None:
            results.append(
                value
            )

    return tuple(
        results
    )


# =========================================================
# Evidence checks
# =========================================================

def _evidence_explicitly_states_eligibility(
    evidence: str,
) -> bool:
    """
    Require evidence to explicitly establish eligibility.
    """
    lower = _lower(
        evidence
    )

    return bool(
        re.search(
            r"\b(?:applicant|applicants|candidate|candidates|student|students)\b"
            r".{0,160}"
            r"\b(?:are|is|will be|would be)\b"
            r".{0,80}"
            r"\b(?:eligible|qualified)\b",
            lower,
        )
        or re.search(
            r"\b(?:eligible|qualified)\s+for\b",
            lower,
        )
    )


def _evidence_explicitly_applies_exemption(
    evidence: str,
) -> bool:
    """
    Require evidence to establish personal applicability of an exemption.

    Category-wide statements are intentionally insufficient.

    Example:

        "SC/ST/PwD students are exempt"

    does not establish:

        "You are exempt"
    """
    lower = _lower(
        evidence
    )

    patterns = (
        r"\byou\s+(?:are|would be|will be|should be|may be|might be)\s+"
        r"(?:exempt|exempted)\b",
        r"\byou\s+(?:qualify|qualifies)\s+"
        r".{0,60}\bexemption\b",
        r"\bthe\s+applicant\s+(?:is|shall be|will be|may be)\s+"
        r"(?:exempt|exempted)\b",
        r"\bthe\s+candidate\s+(?:is|shall be|will be|may be)\s+"
        r"(?:exempt|exempted)\b",
    )

    return any(
        re.search(
            pattern,
            lower,
        )
        for pattern in patterns
    )


# =========================================================
# Arithmetic grounding
# =========================================================

def _supported_daily_arithmetic(
    answer_sentence: str,
    evidence: str,
) -> bool:
    """
    Validate simple daily-rate arithmetic.

    Supported example:

        Evidence:
            "Single occupancy hostel room rent is ₹500 per day."

        Answer:
            "At ₹500 per day, five days would total ₹2,500."

    Requirements:
        - answer contains a per-day rate
        - same rate exists in evidence
        - answer contains an explicit day duration
        - answer contains the calculated total
        - calculation is mathematically correct
    """
    answer_rates = _extract_daily_rates(
        answer_sentence
    )

    if not answer_rates:
        return False

    evidence_rates = _extract_daily_rates(
        evidence
    )

    if not evidence_rates:
        return False

    supported_rates = tuple(
        rate
        for rate in answer_rates
        if any(
            abs(
                rate - evidence_rate
            ) < 1e-9
            for evidence_rate in evidence_rates
        )
    )

    if not supported_rates:
        return False

    day_counts = _extract_day_durations(
        answer_sentence
    )

    if not day_counts:
        return False

    answer_money = _extract_money_values(
        answer_sentence
    )

    if len(answer_money) < 2:
        return False

    rate = supported_rates[0]

    for day_count in day_counts:

        expected_total = (
            rate * day_count
        )

        if any(
            abs(
                amount - expected_total
            ) < 1e-9
            for amount in answer_money
        ):
            return True

    return False


# =========================================================
# Relationship checks
# =========================================================

def _check_eligibility_conclusion(
    answer_sentence: str,
    evidence: str,
) -> GroundingIssue | None:

    if not any(
        pattern.search(
            answer_sentence
        )
        for pattern in ELIGIBILITY_PATTERNS
    ):
        return None

    if _evidence_explicitly_states_eligibility(
        evidence
    ):
        return None

    return GroundingIssue(
        issue_type="eligibility_conclusion",
        answer_claim=answer_sentence,
        evidence_basis=evidence,
        reason=(
            "The answer makes an eligibility conclusion, but the "
            "evidence does not explicitly establish that conclusion."
        ),
    )


def _check_exemption_conclusion(
    answer_sentence: str,
    evidence: str,
) -> GroundingIssue | None:

    if not any(
        pattern.search(
            answer_sentence
        )
        for pattern in EXEMPTION_PATTERNS
    ):
        return None

    if _evidence_explicitly_applies_exemption(
        evidence
    ):
        return None

    return GroundingIssue(
        issue_type="exemption_conclusion",
        answer_claim=answer_sentence,
        evidence_basis=evidence,
        reason=(
            "The answer applies an exemption to the user, but the "
            "evidence does not establish personal applicability."
        ),
    )


def _check_personal_cost_interpretation(
    answer_sentence: str,
    evidence: str,
) -> GroundingIssue | None:
    """
    Protect personal/exact cost interpretations while allowing valid
    arithmetic derivations.
    """

    # -----------------------------------------------------
    # A supported arithmetic calculation is valid.
    # -----------------------------------------------------

    if _supported_daily_arithmetic(
        answer_sentence,
        evidence,
    ):
        return None

    has_personal_language = any(
        pattern.search(
            answer_sentence
        )
        for pattern in PERSONAL_COST_PATTERNS
    )

    has_total_language = any(
        pattern.search(
            answer_sentence
        )
        for pattern in PERSONAL_TOTAL_PATTERNS
    )

    if not (
        has_personal_language
        or has_total_language
    ):
        return None

    if not _numeric_markers_supported(
        answer_sentence,
        evidence,
    ):
        return GroundingIssue(
            issue_type="unsupported_numeric_relationship",
            answer_claim=answer_sentence,
            evidence_basis=evidence,
            reason=(
                "The answer assigns a numeric value to the user's "
                "case, but the value is not present in the evidence."
            ),
        )

    return GroundingIssue(
        issue_type="personal_cost_interpretation",
        answer_claim=answer_sentence,
        evidence_basis=evidence,
        reason=(
            "The numeric value is present in the evidence, but the "
            "answer gives it a personal or exact interpretation that "
            "is not explicitly established."
        ),
    )


def _numeric_markers_supported(
    answer: str,
    evidence: str,
) -> bool:
    """
    Check whether answer numeric markers are supported by evidence.
    """
    answer_markers = _numeric_markers(
        answer
    )

    if not answer_markers:
        return True

    evidence_markers = _numeric_markers(
        evidence
    )

    normalized_evidence = {
        marker.casefold()
        for marker in evidence_markers
    }

    evidence_digits = {
        _digits_only(
            marker
        )
        for marker in evidence_markers
    }

    for marker in answer_markers:

        if marker in normalized_evidence:
            continue

        digits = _digits_only(
            marker
        )

        if (
            digits
            and digits in evidence_digits
        ):
            continue

        # Derived totals are intentionally allowed only through
        # _supported_daily_arithmetic().
        return False

    return True


def _numeric_markers(
    text: str,
) -> tuple[str, ...]:
    """
    Extract monetary, percentage, and score-like markers.
    """
    return tuple(
        _normalize(
            marker
        ).casefold()
        for marker in re.findall(
            r"""
            (?:
                ₹\s*\d[\d,]*(?:\.\d+)?
                |
                \b(?:rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?
                |
                \b\d+(?:\.\d+)?\s*%
                |
                \b\d+(?:\.\d+)?\s*/\s*\d+
            )
            """,
            text,
            re.IGNORECASE | re.VERBOSE,
        )
    )


def _digits_only(
    text: str,
) -> str:
    """Extract numeric characters for normalization."""
    return re.sub(
        r"[^\d.]",
        "",
        text.casefold(),
    )


def _check_causal_claim(
    answer_sentence: str,
    evidence: str,
) -> GroundingIssue | None:

    if not any(
        pattern.search(
            answer_sentence
        )
        for pattern in CAUSAL_PATTERNS
    ):
        return None

    evidence_lower = _lower(
        evidence
    )

    if any(
        connector in evidence_lower
        for connector in (
            "because",
            "since",
            "therefore",
            "thus",
            "hence",
        )
    ):
        return None

    return GroundingIssue(
        issue_type="unsupported_causal_claim",
        answer_claim=answer_sentence,
        evidence_basis=evidence,
        reason=(
            "The answer introduces a causal relationship that is not "
            "explicitly represented in the evidence."
        ),
    )


# =========================================================
# Public API
# =========================================================

def assess_answer_grounding(
    *,
    answer: str,
    evidence: str,
) -> dict:
    """
    Assess high-risk semantic relationships.

    Status:
        grounded
            No high-risk relationship detected.

        review
            One or more relationships require review.
    """
    answer = _normalize(
        answer
    )
    evidence = _normalize(
        evidence
    )

    if not answer:
        return {
            "status": "grounded",
            "issues": [],
            "issue_count": 0,
        }

    issues: list[GroundingIssue] = []

    for sentence in _sentences(
        answer
    ):

        checks = (
            _check_eligibility_conclusion(
                sentence,
                evidence,
            ),
            _check_exemption_conclusion(
                sentence,
                evidence,
            ),
            _check_personal_cost_interpretation(
                sentence,
                evidence,
            ),
            _check_causal_claim(
                sentence,
                evidence,
            ),
        )

        for issue in checks:
            if issue is not None:
                issues.append(
                    issue
                )

    return {
        "status": (
            "review"
            if issues
            else "grounded"
        ),
        "issues": issues,
        "issue_count": len(
            issues
        ),
    }


def has_grounding_issue(
    *,
    answer: str,
    evidence: str,
) -> bool:
    """
    Return True when a high-risk grounding issue exists.
    """
    return (
        assess_answer_grounding(
            answer=answer,
            evidence=evidence,
        )["status"]
        == "review"
    )
