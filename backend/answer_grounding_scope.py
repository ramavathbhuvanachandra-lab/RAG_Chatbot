"""
IIT Jodhpur V1 — Phase 5.5.1
Numeric Scope Attachment

Purpose
-------
Detect cases where a numeric value in the answer is attached to a
specific duration, occupancy, resident category, or "entire stay"
interpretation that the evidence does not actually establish.

This is an extension of Phase 5.5.

It does NOT:
    - retrieve
    - call an LLM
    - modify the answer
    - replace answer_claim_audit
    - replace answer_grounding
    - perform generic semantic reasoning

Examples
--------
Evidence:
    "₹2,175 for stays exceeding 11 days but less than a month."

Answer:
    "Your 20-day stay costs ₹2,175."

Result:
    review

Evidence:
    "Single occupancy room rent is ₹500 per day."

Answer:
    "At ₹500 per day, five days would total ₹2,500."

Result:
    grounded

The second case is an explicit arithmetic derivation from a supported
per-day rate, so the new layer does not incorrectly reject it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# =========================================================
# Result model
# =========================================================

@dataclass(frozen=True)
class ScopeIssue:
    """One unsupported numeric-scope relationship."""

    issue_type: str
    answer_claim: str
    evidence_basis: str
    reason: str


# =========================================================
# Normalization
# =========================================================

def _normalize(
    text: str,
) -> str:
    """Normalize whitespace while retaining semantic wording."""
    return " ".join(
        str(text or "").strip().split()
    )


def _lower(
    text: str,
) -> str:
    return _normalize(text).casefold()


def _sentences(
    text: str,
) -> tuple[str, ...]:
    """Split answer into sentence-like units."""
    normalized = _normalize(text)

    if not normalized:
        return ()

    return tuple(
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            normalized,
        )
        if part.strip()
    )


# =========================================================
# Numeric parsing
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

PER_DAY_RATE_PATTERN = re.compile(
    r"""
    (?:
        ₹\s*(?P<rupee>[0-9][0-9,]*(?:\.[0-9]+)?)
        |
        \b(?:rs\.?|inr)\s*(?P<rs>[0-9][0-9,]*(?:\.[0-9]+)?)
    )
    \s*(?:per|/)\s*day
    """,
    re.IGNORECASE | re.VERBOSE,
)

DURATION_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

OCCUPANCY_PATTERN = re.compile(
    r"\b(single|double)\s+occupancy\b",
    re.IGNORECASE,
)


def _money_values(
    text: str,
) -> tuple[float, ...]:
    """Extract monetary values."""
    values: list[float] = []

    for match in MONEY_PATTERN.finditer(text):

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

    return tuple(values)


def _duration_values(
    text: str,
) -> tuple[tuple[float, str], ...]:
    """Extract explicit durations."""
    return tuple(
        (
            float(number),
            unit.casefold(),
        )
        for number, unit
        in DURATION_PATTERN.findall(
            text
        )
    )


def _normalize_duration_unit(
    unit: str,
) -> str:
    """Normalize duration units."""
    unit = unit.casefold()

    if unit in {
        "day",
        "days",
    }:
        return "day"

    if unit in {
        "week",
        "weeks",
    }:
        return "week"

    if unit in {
        "month",
        "months",
    }:
        return "month"

    if unit in {
        "year",
        "years",
    }:
        return "year"

    return unit


# =========================================================
# Scope language
# =========================================================

ENTIRE_STAY_PATTERNS = (
    re.compile(
        r"\b(?:for|of)\s+the\s+entire\s+stay\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:entire|full)\s+stay\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bflat\s+(?:fee|charge|rate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfixed\s+(?:fee|charge|rate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:regardless|irrespective)\s+of\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexact\s+(?:total|cost|fee|charge)\b",
        re.IGNORECASE,
    ),
)

PERSONAL_COST_PATTERNS = (
    re.compile(
        r"\b(?:your|you will|you would)\b"
        r".{0,100}"
        r"\b(?:cost|costs|fee|fees|charge|charges|pay|paying|total)\b"
        r".{0,100}"
        r"(?:₹|rs\.?|inr)?\s*[\d,]+",
        re.IGNORECASE,
    ),
)


# =========================================================
# Evidence scope helpers
# =========================================================

def _money_has_nearby_term(
    *,
    evidence: str,
    money_value: float,
    term_pattern: re.Pattern[str],
    window: int = 180,
) -> bool:
    """
    Check whether a specific money value is explicitly associated with
    a scope term in the evidence.

    This prevents unrelated nearby table values from being treated as
    supporting the same numeric claim.
    """
    normalized = _normalize(
        evidence
    )

    for match in MONEY_PATTERN.finditer(
        normalized
    ):
        raw = (
            match.group("rupee")
            or match.group("rs")
        )

        if raw is None:
            continue

        value = float(
            raw.replace(
                ",",
                "",
            )
        )

        if value != money_value:
            continue

        start = max(
            0,
            match.start() - window,
        )

        end = min(
            len(normalized),
            match.end() + window,
        )

        neighborhood = normalized[
            start:end
        ]

        if term_pattern.search(
            neighborhood
        ):
            return True

    return False


def _money_explicitly_has_duration_scope(
    evidence: str,
    money_value: float,
) -> bool:
    """Check whether the amount is explicitly linked to a duration."""
    return _money_has_nearby_term(
        evidence=evidence,
        money_value=money_value,
        term_pattern=DURATION_PATTERN,
    )


def _money_explicitly_has_occupancy_scope(
    evidence: str,
    money_value: float,
) -> bool:
    """Check whether the amount is explicitly linked to occupancy."""
    return _money_has_nearby_term(
        evidence=evidence,
        money_value=money_value,
        term_pattern=OCCUPANCY_PATTERN,
    )


def _money_is_explicitly_personal(
    evidence: str,
    money_value: float,
) -> bool:
    """
    Check whether evidence explicitly assigns an amount to a person,
    applicant, or exact total.
    """
    normalized = _normalize(
        evidence
    )

    for match in MONEY_PATTERN.finditer(
        normalized
    ):
        raw = (
            match.group("rupee")
            or match.group("rs")
        )

        if raw is None:
            continue

        value = float(
            raw.replace(
                ",",
                "",
            )
        )

        if value != money_value:
            continue

        start = max(
            0,
            match.start() - 180,
        )

        end = min(
            len(normalized),
            match.end() + 180,
        )

        neighborhood = _lower(
            normalized[start:end]
        )

        if (
            re.search(
                r"\b(?:your|you|applicant|candidate)\b",
                neighborhood,
            )
            and re.search(
                r"\b(?:total|cost|fee|charge|pay)\b",
                neighborhood,
            )
        ):
            return True

        if any(
            pattern.search(
                neighborhood
            )
            for pattern in ENTIRE_STAY_PATTERNS
        ):
            return True

    return False


# =========================================================
# Arithmetic derivation
# =========================================================

def _is_supported_daily_arithmetic(
    answer_sentence: str,
    evidence: str,
) -> bool:
    """
    Allow an explicit arithmetic derivation when:

        evidence -> supported per-day rate
        answer   -> same rate × explicit number of days
        answer   -> explicit resulting total

    This avoids blocking legitimate simple calculations.
    """
    answer_rates = tuple(
        float(
            (
                match.group("rupee")
                or match.group("rs")
            ).replace(",", "")
        )
        for match in PER_DAY_RATE_PATTERN.finditer(
            answer_sentence
        )
    )

    if not answer_rates:
        return False

    durations = _duration_values(
        answer_sentence
    )

    day_durations = tuple(
        number
        for number, unit in durations
        if _normalize_duration_unit(unit)
        == "day"
    )

    answer_money = _money_values(
        answer_sentence
    )

    if not day_durations:
        return False

    if len(answer_money) < 2:
        return False

    evidence_rates = tuple(
        float(
            (
                match.group("rupee")
                or match.group("rs")
            ).replace(",", "")
        )
        for match in PER_DAY_RATE_PATTERN.finditer(
            evidence
        )
    )

    if not evidence_rates:
        return False

    # The rate used in the answer must exist in evidence.
    valid_rates = [
        rate
        for rate in answer_rates
        if any(
            abs(rate - evidence_rate)
            < 1e-9
            for evidence_rate in evidence_rates
        )
    ]

    if not valid_rates:
        return False

    rate = valid_rates[0]
    days = day_durations[0]

    expected_total = (
        rate * days
    )

    # Find a money value in the answer that equals the calculated total.
    return any(
        abs(
            value
            - expected_total
        ) < 1e-9
        for value in answer_money
    )


# =========================================================
# High-risk scope check
# =========================================================

def _check_numeric_scope(
    answer_sentence: str,
    evidence: str,
) -> list[ScopeIssue]:
    """
    Check all monetary claims in one answer sentence.
    """
    issues: list[ScopeIssue] = []

    money_values = _money_values(
        answer_sentence
    )

    if not money_values:
        return issues

    # -----------------------------------------------------
    # Explicit arithmetic derivation is allowed when fully
    # supported by an evidence-backed daily rate.
    # -----------------------------------------------------
    if _is_supported_daily_arithmetic(
        answer_sentence,
        evidence,
    ):
        return issues

    answer_durations = _duration_values(
        answer_sentence
    )

    answer_occupancies = tuple(
        match.group(1).casefold()
        for match in OCCUPANCY_PATTERN.finditer(
            answer_sentence
        )
    )

    has_exact_or_entire_scope = any(
        pattern.search(
            answer_sentence
        )
        for pattern in ENTIRE_STAY_PATTERNS
    )

    has_personal_cost = any(
        pattern.search(
            answer_sentence
        )
        for pattern in PERSONAL_COST_PATTERNS
    )

    for money_value in money_values:

        # -------------------------------------------------
        # Duration attachment.
        # -------------------------------------------------
        if (
            answer_durations
            and (
                has_exact_or_entire_scope
                or has_personal_cost
                or len(answer_durations) > 0
            )
            and not _money_explicitly_has_duration_scope(
                evidence,
                money_value,
            )
        ):
            issues.append(
                ScopeIssue(
                    issue_type=(
                        "numeric_duration_scope"
                    ),
                    answer_claim=answer_sentence,
                    evidence_basis=evidence,
                    reason=(
                        "The answer attaches a monetary value to a "
                        "specific stay/duration, but the evidence does "
                        "not explicitly attach that value to the same "
                        "duration."
                    ),
                )
            )

        # -------------------------------------------------
        # Occupancy attachment.
        # -------------------------------------------------
        if (
            answer_occupancies
            and not _money_explicitly_has_occupancy_scope(
                evidence,
                money_value,
            )
        ):
            issues.append(
                ScopeIssue(
                    issue_type=(
                        "numeric_occupancy_scope"
                    ),
                    answer_claim=answer_sentence,
                    evidence_basis=evidence,
                    reason=(
                        "The answer attaches a monetary value to an "
                        "occupancy type, but the evidence does not "
                        "explicitly attach that value to the same "
                        "occupancy type."
                    ),
                )
            )

        # -------------------------------------------------
        # "Entire stay" / exact-total interpretation.
        # -------------------------------------------------
        if (
            has_exact_or_entire_scope
            and not _money_is_explicitly_personal(
                evidence,
                money_value,
            )
            and not _money_explicitly_has_duration_scope(
                evidence,
                money_value,
            )
        ):
            issues.append(
                ScopeIssue(
                    issue_type=(
                        "numeric_total_scope"
                    ),
                    answer_claim=answer_sentence,
                    evidence_basis=evidence,
                    reason=(
                        "The answer interprets the monetary value as "
                        "an exact/entire-stay total, but the evidence "
                        "does not establish that interpretation."
                    ),
                )
            )

    return issues


# =========================================================
# Public API
# =========================================================

def assess_numeric_scope(
    *,
    answer: str,
    evidence: str,
) -> dict:
    """
    Assess high-risk numeric scope attachment.

    Status:
        grounded
            No unsupported numeric-scope relationship found.

        review
            One or more numeric-scope relationships require review.
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

    issues: list[ScopeIssue] = []

    for sentence in _sentences(
        answer
    ):
        issues.extend(
            _check_numeric_scope(
                sentence,
                evidence,
            )
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


def has_numeric_scope_issue(
    *,
    answer: str,
    evidence: str,
) -> bool:
    """Return True when numeric scope needs review."""
    return (
        assess_numeric_scope(
            answer=answer,
            evidence=evidence,
        )["status"]
        == "review"
    )
