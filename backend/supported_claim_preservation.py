"""
IIT Jodhpur V1 — Supported Claim Preservation

Purpose
-------
Check whether important factual markers present in final evidence are
preserved by the generated answer.

This is an answer-grounding control.

It does NOT require literal sentence matching.

It checks high-risk factual markers where omission can materially
change an answer:

- money
- percentages
- CGPA / CPI
- durations
- program names

Design principles
-----------------
- Deterministic.
- Lightweight.
- No LLM call.
- Conservative.
- No institution-specific values.
- Exact duplicates are normalized.
- Only materially useful factual markers are audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet


# =========================================================
# Marker container
# =========================================================

@dataclass(frozen=True)
class FactualMarkers:
    """
    Normalized high-risk factual markers extracted from text.
    """

    money: FrozenSet[str]
    percentages: FrozenSet[str]
    scores: FrozenSet[str]
    durations: FrozenSet[str]
    program_terms: FrozenSet[str]


# =========================================================
# Normalization
# =========================================================

def _normalize(
    text: str,
) -> str:
    """
    Normalize text for marker extraction.
    """

    text = str(
        text or ""
    ).lower()

    text = text.replace(
        "–",
        "-",
    )

    text = text.replace(
        "—",
        "-",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# Money
# =========================================================

MONEY_PATTERN = re.compile(
    r"""
    (?:
        ₹
        |
        rs\.?
        |
        inr
    )
    \s*
    (
        \d+(?:,\d{3})*(?:\.\d+)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# =========================================================
# Percentages
# =========================================================

PERCENTAGE_PATTERN = re.compile(
    r"""
    (
        \d+(?:\.\d+)?
    )
    \s*
    %
    """,
    re.IGNORECASE | re.VERBOSE,
)


# =========================================================
# Scores
# =========================================================

CGPA_PATTERN = re.compile(
    r"""
    (
        \d+(?:\.\d+)?
    )
    \s*
    (?:
        /
        \s*
        10
    )?
    \s*
    (?:
        cgpa
        |
        cpi
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


SCALE_PATTERN = re.compile(
    r"""
    (
        \d+(?:\.\d+)?
    )
    \s*
    /
    \s*
    (
        10
        |
        8
        |
        4
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# =========================================================
# Duration
# =========================================================

DURATION_PATTERN = re.compile(
    r"""
    (
        \d+(?:\.\d+)?
    )
    \s*
    (
        day
        |
        days
        |
        month
        |
        months
        |
        year
        |
        years
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


WORD_DURATION_PATTERN = re.compile(
    r"""
    \b
    (
        one
        |
        two
        |
        three
        |
        four
        |
        five
        |
        six
        |
        seven
        |
        eight
        |
        nine
        |
        ten
    )
    -
    year
    s?
    \b
    |
    \b
    (
        one
        |
        two
        |
        three
        |
        four
        |
        five
        |
        six
        |
        seven
        |
        eight
        |
        nine
        |
        ten
    )
    \s+
    (
        year
        |
        years
        |
        month
        |
        months
        |
        day
        |
        days
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


WORD_NUMBER_MAP = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


# =========================================================
# Program markers
# =========================================================

PROGRAM_PATTERNS = {
    "btech": re.compile(
        r"\bb\.?\s*tech\.?\b",
        re.IGNORECASE,
    ),

    "mtech": re.compile(
        r"\bm\.?\s*tech\.?\b",
        re.IGNORECASE,
    ),

    "msc": re.compile(
        r"\bm\.?\s*sc\.?\b",
        re.IGNORECASE,
    ),

    "bsc": re.compile(
        r"\bb\.?\s*sc\.?\b",
        re.IGNORECASE,
    ),

    "phd": re.compile(
        r"\bph\.?\s*d\.?\b",
        re.IGNORECASE,
    ),

    "mba": re.compile(
        r"\bmba\b",
        re.IGNORECASE,
    ),

    "ms_by_research": re.compile(
        r"\bm\.?\s*s\.?\s+by\s+research\b",
        re.IGNORECASE,
    ),
}


# =========================================================
# Marker extraction helpers
# =========================================================

def _money_markers(
    normalized: str,
) -> set[str]:
    """
    Extract normalized money values.

    Example:
        ₹2,175
        INR 2175

    become:
        "2175"
    """

    markers = set()

    for match in MONEY_PATTERN.finditer(
        normalized
    ):

        value = match.group(
            1
        ).replace(
            ",",
            "",
        )

        markers.add(
            value
        )

    return markers


def _percentage_markers(
    normalized: str,
) -> set[str]:
    """
    Extract normalized percentage values.
    """

    return {
        match.group(
            1
        )
        + "%"
        for match in PERCENTAGE_PATTERN.finditer(
            normalized
        )
    }


def _score_markers(
    normalized: str,
) -> set[str]:
    """
    Extract CGPA/CPI/scaled-score markers.

    Representation:
        6.0/10
        5.5/10
        4.8/8
    """

    markers = set()

    for match in CGPA_PATTERN.finditer(
        normalized
    ):

        value = match.group(
            1
        )

        suffix = (
            "/10"
            if "/10" in match.group(0)
            else ""
        )

        markers.add(
            f"{value}{suffix}"
        )

    for match in SCALE_PATTERN.finditer(
        normalized
    ):

        markers.add(
            f"{match.group(1)}/{match.group(2)}"
        )

    return markers


def _duration_markers(
    normalized: str,
) -> set[str]:
    """
    Extract normalized duration markers.
    """

    markers = set()

    for match in DURATION_PATTERN.finditer(
        normalized
    ):

        number = match.group(
            1
        )

        unit = match.group(
            2
        ).lower()

        unit = unit.rstrip(
            "s"
        )

        markers.add(
            f"{number} {unit}"
        )

    for match in WORD_DURATION_PATTERN.finditer(
        normalized
    ):

        groups = match.groups()

        word = next(
            (
                value
                for value in groups
                if value
                in WORD_NUMBER_MAP
            ),
            None,
        )

        unit = next(
            (
                value
                for value in groups
                if value
                in {
                    "day",
                    "days",
                    "month",
                    "months",
                    "year",
                    "years",
                }
            ),
            None,
        )

        if (
            word
            and
            unit
        ):

            unit = unit.rstrip(
                "s"
            )

            markers.add(
                f"{WORD_NUMBER_MAP[word]} {unit}"
            )

    return markers


def _program_markers(
    normalized: str,
) -> set[str]:
    """
    Extract normalized program markers.
    """

    return {
        name
        for name, pattern in PROGRAM_PATTERNS.items()
        if pattern.search(
            normalized
        )
    }


# =========================================================
# Public extraction API
# =========================================================

def extract_factual_markers(
    text: str,
) -> FactualMarkers:
    """
    Extract high-risk factual markers from text.
    """

    normalized = _normalize(
        text
    )

    return FactualMarkers(
        money=frozenset(
            _money_markers(
                normalized
            )
        ),
        percentages=frozenset(
            _percentage_markers(
                normalized
            )
        ),
        scores=frozenset(
            _score_markers(
                normalized
            )
        ),
        durations=frozenset(
            _duration_markers(
                normalized
            )
        ),
        program_terms=frozenset(
            _program_markers(
                normalized
            )
        ),
    )


# =========================================================
# Marker comparison
# =========================================================

def _missing_markers(
    answer_markers: FactualMarkers,
    context_markers: FactualMarkers,
) -> list[str]:
    """
    Determine which supported markers are absent from the answer.

    Output strings are intentionally human-readable so diagnostics can
    show exactly what was omitted.
    """

    missing = []

    for value in sorted(
        context_markers.money
        -
        answer_markers.money
    ):
        missing.append(
            value
        )

    for value in sorted(
        context_markers.percentages
        -
        answer_markers.percentages
    ):
        missing.append(
            value
        )

    for value in sorted(
        context_markers.scores
        -
        answer_markers.scores
    ):
        missing.append(
            value
        )

    for value in sorted(
        context_markers.durations
        -
        answer_markers.durations
    ):
        missing.append(
            value
        )

    for value in sorted(
        context_markers.program_terms
        -
        answer_markers.program_terms
    ):
        missing.append(
            value
        )

    return missing


# =========================================================
# Main audit
# =========================================================

def assess_supported_claim_preservation(
    answer: str,
    context: str,
) -> dict:
    """
    Audit preservation of important supported factual markers.

    Status:
        supported
            No material high-risk supported marker is missing.

        incomplete
            At least one high-risk supported marker is absent from
            the answer.

    Important limitation:
        This is an omission detector, not a complete semantic truth
        verifier. It should therefore complement, not replace, the
        existing answer claim audit.
    """

    context_markers = (
        extract_factual_markers(
            context
        )
    )

    answer_markers = (
        extract_factual_markers(
            answer
        )
    )

    missing = _missing_markers(
        answer_markers=answer_markers,
        context_markers=context_markers,
    )

    if missing:

        return {
            "status": "incomplete",
            "missing_claims": missing,
            "context_markers": context_markers,
            "answer_markers": answer_markers,
        }

    return {
        "status": "supported",
        "missing_claims": [],
        "context_markers": context_markers,
        "answer_markers": answer_markers,
    }
