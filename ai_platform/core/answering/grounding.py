"""Deterministic answer-to-evidence grounding checks for the reusable RAG core.

Pipeline
--------
E6 verified evidence package -> E7 generated answer -> E7.2 grounding -> E8 guard.

The module checks high-risk relationships that can be wrong even when the answer
contains words or values also present in evidence. It never retrieves, calls an
LLM, rewrites the answer, or depends on institution-specific vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GroundingIssue:
    issue_type: str
    answer_claim: str
    evidence_basis: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "issue_type": self.issue_type,
            "answer_claim": self.answer_claim,
            "evidence_basis": self.evidence_basis,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GroundingResult:
    status: str
    issues: tuple[GroundingIssue, ...] = ()

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "issue_count": self.issue_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _normalize(text: object) -> str:
    value = str(text or "").replace("–", "-").replace("—", "-")
    return " ".join(value.split())


def _lower(text: object) -> str:
    return _normalize(text).casefold()


def _sentences(text: str) -> tuple[str, ...]:
    value = _normalize(text)
    if not value:
        return ()
    return tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", value)
        if part.strip()
    )


# ---------------------------------------------------------------------------
# Generic factual marker extraction
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_MONEY_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?",
    re.I,
)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_RATIO_RE = re.compile(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ACADEMIC_YEAR_RE = re.compile(
    r"\b(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\b"
)
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:day|days|week|weeks|month|months|year|years|semester|semesters)\b",
    re.I,
)


def _money_normalized(markers: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for marker in markers:
        value = _lower(marker)
        value = re.sub(
            r"^(?:inr|rs\.?|₹)\s*",
            "rs ",
            value,
        )
        value = re.sub(r"\s+", " ", value).strip()

        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return tuple(result)


def _markers(text: str) -> dict[str, tuple[str, ...]]:
    normalized = _normalize(text)

    for word, number in _NUMBER_WORDS.items():
        normalized = re.sub(
            rf"\b{re.escape(word)}\b",
            str(number),
            normalized,
            flags=re.I,
        )

    return {
        "money": _money_normalized(
            _MONEY_RE.findall(normalized)
        ),
        "percentages": tuple(
            dict.fromkeys(
                _lower(x)
                for x in _PERCENT_RE.findall(normalized)
            )
        ),
        "ratios": tuple(
            dict.fromkeys(
                _lower(x)
                for x in _RATIO_RE.findall(normalized)
            )
        ),
        "years": tuple(
            dict.fromkeys(
                _YEAR_RE.findall(normalized)
            )
        ),
        "academic_years": tuple(
            dict.fromkeys(
                _lower(x)
                for x in _ACADEMIC_YEAR_RE.findall(normalized)
            )
        ),
        "durations": tuple(
            dict.fromkeys(
                _lower(x)
                for x in _DURATION_RE.findall(normalized)
            )
        ),
    }


def _numeric_signature(value: str) -> str:
    return re.sub(
        r"[^0-9.]",
        "",
        _lower(value),
    )


def _unsupported_numeric_markers(
    answer: str,
    evidence: str,
) -> tuple[str, ...]:
    answer_markers = _markers(answer)
    evidence_markers = _markers(evidence)
    unsupported: list[str] = []

    derived_totals: set[str] = set()

    answer_rates = [
        float(
            m.group("amount").replace(",", "")
        )
        for m in _PER_DAY_RATE_PATTERN.finditer(answer)
    ]

    evidence_rates = [
        float(
            m.group("amount").replace(",", "")
        )
        for m in _PER_DAY_RATE_PATTERN.finditer(evidence)
    ]

    day_counts = [
        number
        for number, unit in _durations(answer)
        if unit in {"day", "days"}
    ]

    supported_rates = [
        rate
        for rate in answer_rates
        if any(
            math.isclose(
                rate,
                evidence_rate,
                rel_tol=0,
                abs_tol=1e-9,
            )
            for evidence_rate in evidence_rates
        )
    ]

    for rate in supported_rates:
        for days in day_counts:
            total = rate * days

            if float(total).is_integer():
                derived_totals.add(str(int(total)))
            else:
                derived_totals.add(str(total))

    for category in (
        "money",
        "percentages",
        "ratios",
        "years",
        "academic_years",
    ):
        evidence_exact = set(
            evidence_markers[category]
        )
        evidence_numeric = {
            _numeric_signature(value)
            for value in evidence_exact
        }

        for value in answer_markers[category]:
            if value in evidence_exact:
                continue

            signature = _numeric_signature(value)

            if signature and signature in evidence_numeric:
                continue

            if (
                category == "money"
                and signature in derived_totals
            ):
                continue

            unsupported.append(value)

    return tuple(
        dict.fromkeys(unsupported)
    )


def _money_values(text: str) -> tuple[float, ...]:
    values: list[float] = []

    for marker in _markers(text)["money"]:
        match = re.search(
            r"\brs\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            marker,
        )

        if match:
            try:
                values.append(
                    float(
                        match.group(1).replace(",", "")
                    )
                )
            except ValueError:
                pass

    return tuple(values)


def _durations(
    text: str,
) -> tuple[tuple[float, str], ...]:
    result: list[tuple[float, str]] = []

    for marker in _markers(text)["durations"]:
        match = re.fullmatch(
            r"([0-9]+(?:\.[0-9]+)?)\s*"
            r"(day|days|week|weeks|month|months|year|years|semester|semesters)",
            marker,
        )

        if match:
            result.append(
                (
                    float(match.group(1)),
                    match.group(2),
                )
            )

    return tuple(result)


def _contains_any(
    text: str,
    patterns: Iterable[re.Pattern[str]],
) -> bool:
    return any(
        pattern.search(text)
        for pattern in patterns
    )


# ---------------------------------------------------------------------------
# High-risk relationship patterns
# ---------------------------------------------------------------------------

_PERSONAL_ELIGIBILITY_PATTERNS = (
    re.compile(
        r"\b(?:you|your)\b.{0,150}"
        r"\b(?:are|would be|will be|should be|may be|might be)\b"
        r".{0,90}\b(?:eligible|qualified)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:you|your)\b.{0,100}"
        r"\bcan\b.{0,70}\bapply\b",
        re.I,
    ),
)

_PERSONAL_EXEMPTION_PATTERNS = (
    re.compile(
        r"\b(?:you|your)\b.{0,150}"
        r"\b(?:are|would be|will be|should be|may be|might be)\b"
        r".{0,90}\b(?:exempt|exempted)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:you|your)\b.{0,130}"
        r"\b(?:qualify|qualifies)\b.{0,90}"
        r"\bexemption\b",
        re.I,
    ),
)

_PERSONAL_COST_PATTERNS = (
    re.compile(
        r"\b(?:your|you will|you would)\b.{0,120}"
        r"\b(?:exact|total)\b.{0,90}"
        r"\b(?:cost|fee|charge|amount|rent|payment)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:your|you will|you would)\b.{0,120}"
        r"\b(?:pay|cost|spend)\b.{0,70}"
        r"(?:rs\.?|inr|₹)\s*\d",
        re.I,
    ),
    re.compile(
        r"\bthe\s+(?:exact|total)\s+"
        r"(?:cost|fee|charge|amount)\b",
        re.I,
    ),
)

# Deliberately excludes bare "so".
#
# "So" appears naturally in ordinary explanatory language and is not, by
# itself, reliable evidence that a sentence contains an unsupported causal
# relationship. Stronger explicit causal markers remain protected.
_CAUSAL_PATTERNS = (
    re.compile(
        r"\b(?:because|since|therefore|thus|hence|due to|as a result)\b",
        re.I,
    ),
    re.compile(
        r"\bthis\s+(?:means|results?|causes?)\b",
        re.I,
    ),
)

_EXACT_SCOPE_PATTERNS = (
    re.compile(
        r"\b(?:exactly|exact|entire stay|whole stay|"
        r"total stay|for the whole period|for the full period)\b",
        re.I,
    ),
)

_OCCUPANCY_PATTERN = re.compile(
    r"\b(single|double|triple|shared|private)\s*"
    r"(?:occupancy|room|sharing)\b",
    re.I,
)

_PER_DAY_RATE_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)?\s*"
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:per|/|each)\s*day\b",
    re.I,
)

_PERSONAL_EVIDENCE_PATTERN = re.compile(
    r"\b(?:you|your)\b.{0,120}"
    r"\b(?:eligible|qualified|exempt|exempted|pay|payment|"
    r"cost|fee|charge|amount|rent)\b",
    re.I,
)

_REQUIREMENT_EVIDENCE_PATTERN = re.compile(
    r"\b(?:applicant|applicants|candidate|candidates)\b.{0,110}"
    r"\b(?:must|required|requires|required to|minimum|qualification|"
    r"qualifications|criteria|eligible|eligibility)\b",
    re.I,
)

_EXEMPTION_EVIDENCE_PATTERN = re.compile(
    r"\b(?:exempt|exempted|exemption)\b.{0,110}"
    r"\b(?:fee|fees|tuition|charge|charges|payment|pay)\b"
    r"|"
    r"\b(?:fee|fees|tuition|charge|charges|payment|pay)\b.{0,110}"
    r"\b(?:exempt|exempted|exemption)\b",
    re.I,
)


def _supported_daily_arithmetic(
    answer: str,
    evidence: str,
) -> bool:
    answer_rates = tuple(
        float(
            m.group("amount").replace(",", "")
        )
        for m in _PER_DAY_RATE_PATTERN.finditer(answer)
    )

    evidence_rates = tuple(
        float(
            m.group("amount").replace(",", "")
        )
        for m in _PER_DAY_RATE_PATTERN.finditer(evidence)
    )

    if not answer_rates or not evidence_rates:
        return False

    supported_rates = [
        rate
        for rate in answer_rates
        if any(
            math.isclose(
                rate,
                evidence_rate,
                rel_tol=0,
                abs_tol=1e-9,
            )
            for evidence_rate in evidence_rates
        )
    ]

    day_counts = [
        number
        for number, unit in _durations(answer)
        if unit in {"day", "days"}
    ]

    money = _money_values(answer)

    if not supported_rates or not day_counts or len(money) < 2:
        return False

    rate = supported_rates[0]

    return any(
        any(
            math.isclose(
                amount,
                rate * days,
                rel_tol=0,
                abs_tol=1e-9,
            )
            for amount in money
        )
        for days in day_counts
    )


def _evidence_scope_for_money(
    answer_sentence: str,
    evidence: str,
) -> tuple[bool, str | None]:
    money = _money_values(answer_sentence)

    if not money:
        return True, None

    if _supported_daily_arithmetic(
        answer_sentence,
        evidence,
    ):
        return True, None

    answer_durations = _durations(
        answer_sentence
    )

    answer_occupancies = [
        match.group(1).casefold()
        for match in _OCCUPANCY_PATTERN.finditer(
            answer_sentence
        )
    ]

    exact_scope = _contains_any(
        answer_sentence,
        _EXACT_SCOPE_PATTERNS,
    )

    personal_scope = _contains_any(
        answer_sentence,
        _PERSONAL_COST_PATTERNS,
    )

    evidence_lower = _lower(evidence)
    _ = evidence_lower  # retained for compatibility/readability

    for amount in money:
        matching_marker_windows: list[str] = []

        signature = (
            str(int(amount))
            if amount.is_integer()
            else str(amount)
        )

        for raw_match in _MONEY_RE.finditer(
            evidence
        ):
            raw_marker = raw_match.group(0)

            if (
                _numeric_signature(raw_marker)
                != signature
            ):
                continue

            start = max(
                0,
                raw_match.start() - 160,
            )
            end = min(
                len(evidence),
                raw_match.end() + 160,
            )

            matching_marker_windows.append(
                _lower(
                    evidence[start:end]
                )
            )

        # Unknown money values are handled by
        # unsupported_numeric_value.
        if not matching_marker_windows:
            continue

        if answer_durations:
            for number, unit in answer_durations:
                duration_pattern = (
                    rf"\b"
                    rf"{int(number) if number.is_integer() else number:g}"
                    rf"\s*{re.escape(unit)}\b"
                )

                if not any(
                    re.search(
                        duration_pattern,
                        window,
                        re.I,
                    )
                    for window in matching_marker_windows
                ):
                    return False, "numeric_duration_scope"

        if answer_occupancies and not any(
            occupancy in window
            for occupancy in answer_occupancies
            for window in matching_marker_windows
        ):
            return False, "numeric_occupancy_scope"

        if exact_scope or personal_scope:
            if not any(
                re.search(
                    r"\b(?:your|you|applicant|candidate|student|"
                    r"individual|personal|total|entire|whole)\b",
                    window,
                    re.I,
                )
                for window in matching_marker_windows
            ):
                return False, "personal_cost_interpretation"

    return True, None


def _check_sentence(
    sentence: str,
    evidence: str,
) -> list[GroundingIssue]:
    issues: list[GroundingIssue] = []

    if _contains_any(
        sentence,
        _PERSONAL_ELIGIBILITY_PATTERNS,
    ):
        # Policy eligibility is not the same as a personal eligibility
        # conclusion.
        if not _PERSONAL_EVIDENCE_PATTERN.search(
            evidence
        ):
            issues.append(
                GroundingIssue(
                    "eligibility_conclusion",
                    sentence,
                    evidence,
                    (
                        "The answer makes a personal "
                        "eligibility/application conclusion, "
                        "but the evidence does not establish "
                        "facts about the user."
                    ),
                )
            )

    if _contains_any(
        sentence,
        _PERSONAL_EXEMPTION_PATTERNS,
    ):
        if not _PERSONAL_EVIDENCE_PATTERN.search(
            evidence
        ):
            issues.append(
                GroundingIssue(
                    "exemption_conclusion",
                    sentence,
                    evidence,
                    (
                        "The answer applies an exemption to "
                        "the user, but the evidence does not "
                        "establish the user's personal applicability."
                    ),
                )
            )

    if _contains_any(
        sentence,
        _PERSONAL_COST_PATTERNS,
    ):
        supported, reason = _evidence_scope_for_money(
            sentence,
            evidence,
        )

        if not supported:
            issues.append(
                GroundingIssue(
                    reason
                    or "personal_cost_interpretation",
                    sentence,
                    evidence,
                    (
                        "The answer gives a personal, exact, "
                        "duration-specific, or occupancy-specific "
                        "monetary interpretation not established "
                        "by the evidence."
                    ),
                )
            )

    if _contains_any(
        sentence,
        _CAUSAL_PATTERNS,
    ):
        if not _contains_any(
            _lower(evidence),
            _CAUSAL_PATTERNS,
        ):
            issues.append(
                GroundingIssue(
                    "unsupported_causal_claim",
                    sentence,
                    evidence,
                    (
                        "The answer introduces a causal "
                        "relationship that is not explicitly "
                        "represented in the evidence."
                    ),
                )
            )

    return issues


def assess_answer_grounding(
    *,
    answer: str,
    evidence: str,
) -> GroundingResult:
    """Return grounded/review plus precise deterministic grounding issues."""
    answer = _normalize(answer)
    evidence = _normalize(evidence)

    if not answer:
        return GroundingResult(
            status="grounded"
        )

    issues: list[GroundingIssue] = []

    unsupported = _unsupported_numeric_markers(
        answer,
        evidence,
    )

    if unsupported:
        issues.append(
            GroundingIssue(
                "unsupported_numeric_value",
                answer,
                evidence,
                (
                    "The answer contains one or more "
                    "numeric values that are not present "
                    "in the supplied evidence."
                ),
            )
        )

    for sentence in _sentences(answer):
        issues.extend(
            _check_sentence(
                sentence,
                evidence,
            )
        )

    deduped: list[GroundingIssue] = []
    seen: set[tuple[str, str]] = set()

    for issue in issues:
        key = (
            issue.issue_type,
            issue.answer_claim.casefold(),
        )

        if key not in seen:
            seen.add(key)
            deduped.append(issue)

    return GroundingResult(
        status=(
            "review"
            if deduped
            else "grounded"
        ),
        issues=tuple(deduped),
    )


def has_grounding_issue(
    *,
    answer: str,
    evidence: str,
) -> bool:
    return (
        assess_answer_grounding(
            answer=answer,
            evidence=evidence,
        ).status
        == "review"
    )


__all__ = [
    "GroundingIssue",
    "GroundingResult",
    "assess_answer_grounding",
    "has_grounding_issue",
]