"""
Phase 4 — Answer Claim Audit

Purpose
-------
Perform deterministic post-generation checks on high-risk factual
markers in an answer.

The audit focuses on factual values that are dangerous when invented:

    - percentages
    - score / CGPA-style values
    - monetary amounts
    - years
    - durations
    - admission-mode markers

Degree/program terminology is extracted for diagnostics but is NOT
treated as an automatic failure condition. A generated answer may use
the program/degree name as contextual framing even when that exact
term is absent from a short evidence excerpt.

Important invariants
--------------------
- No additional LLM call.
- No institution-specific hardcoding.
- Does not modify the answer.
- Equivalent textual forms are normalized where safely possible.
- Only high-risk factual markers can make the audit unsafe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# =========================================================
# Number normalization
# =========================================================

NUMBER_WORDS = {
    "zero": "0",
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
    "eleven": "11",
    "twelve": "12",
}


def _normalize_number_words(
    text: str,
) -> str:
    """
    Normalize common English number words.
    """

    result = text

    for word, number in NUMBER_WORDS.items():

        result = re.sub(
            rf"\b{word}\b",
            number,
            result,
            flags=re.IGNORECASE,
        )

    return result


# =========================================================
# Text normalization
# =========================================================

def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for deterministic marker comparison.
    """

    text = str(
        text or ""
    ).lower()

    replacements = {
        "₹": " rs ",
        "rs.": " rs ",
        "inr": " rs ",
        "b.tech.": " btech ",
        "b.tech": " btech ",
        "m.tech.": " mtech ",
        "m.tech": " mtech ",
        "m.sc.": " msc ",
        "m.sc": " msc ",
        "m.s.": " ms ",
        "m.s": " ms ",
        "ph.d.": " phd ",
        "ph.d": " phd ",
        "b.s.": " bs ",
        "b.s": " bs ",
        "c.g.p.a.": " cgpa ",
        "cpi/cgpa": " cpi cgpa ",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    text = _normalize_number_words(
        text
    )

    text = re.sub(
        r"[-–—]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# Marker model
# =========================================================

@dataclass(frozen=True)
class FactualMarkers:
    """
    Extracted factual markers.

    program_terms are diagnostic only. They are not independently
    considered unsupported because contextual labels may be supplied by
    the question itself rather than repeated in the evidence excerpt.
    """

    percentages: frozenset[str]
    scores: frozenset[str]
    money: frozenset[str]
    years: frozenset[str]
    durations: frozenset[str]
    program_terms: frozenset[str]
    admission_modes: frozenset[str]


# =========================================================
# Marker patterns
# =========================================================

PERCENTAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*%"
)

SCORE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b"
)

MONEY_PATTERN = re.compile(
    r"""
    (?:
        rs
        |
        ₹
    )
    \s*
    \d[\d,\s]*(?:\.\d+)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

YEAR_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b"
)

DURATION_PATTERN = re.compile(
    r"""
    \b
    \d+(?:\.\d+)?
    \s*
    (?:days?|weeks?|months?|years?|semesters?)
    \b
    |
    \b
    \d+(?:\.\d+)?\s*year
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

PROGRAM_PATTERN = re.compile(
    r"""
    \b
    (?:
        btech |
        mtech |
        msc |
        ms |
        phd |
        bs |
        bachelor |
        master |
        doctoral |
        doctorate
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

ADMISSION_MODE_PATTERN = re.compile(
    r"""
    \b
    (?:
        regular |
        full\s+time |
        part\s+time |
        sponsored |
        external |
        executive |
        dual\s+degree
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# =========================================================
# Marker canonicalization
# =========================================================

def _normalize_marker(
    marker: str,
) -> str:
    return _normalize_text(
        marker
    )


def _normalize_duration(
    marker: str,
) -> str:
    """
    Canonicalize duration expressions.

    Examples:
        4 years  -> 4 years
        four-year -> 4 years
    """

    value = _normalize_text(
        marker
    )

    value = value.replace(
        "year",
        "years",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _normalize_program_term(
    marker: str,
) -> str:
    """
    Canonicalize program/degree terminology.
    """

    value = _normalize_text(
        marker
    )

    aliases = {
        "btech": "btech",
        "mtech": "mtech",
        "msc": "msc",
        "ms": "ms",
        "phd": "phd",
        "bs": "bs",
        "bachelor": "bachelor",
        "master": "master",
        "doctoral": "doctoral",
        "doctorate": "doctoral",
    }

    return aliases.get(
        value,
        value,
    )


def _normalize_admission_mode(
    marker: str,
) -> str:
    """
    Canonicalize admission mode terminology.
    """

    value = _normalize_text(
        marker
    )

    value = value.replace(
        "full time",
        "full-time",
    )

    value = value.replace(
        "part time",
        "part-time",
    )

    value = value.replace(
        "dual degree",
        "dual-degree",
    )

    return value


# =========================================================
# Marker extraction
# =========================================================

def extract_factual_markers(
    text: str,
) -> FactualMarkers:
    """
    Extract high-risk factual markers from text.
    """

    normalized = _normalize_text(
        text
    )

    percentages = frozenset(
        _normalize_marker(
            marker
        )
        for marker in PERCENTAGE_PATTERN.findall(
            normalized
        )
    )

    scores = frozenset(
        _normalize_marker(
            marker
        )
        for marker in SCORE_PATTERN.findall(
            normalized
        )
    )

    money = frozenset(
        _normalize_marker(
            marker
        )
        for marker in MONEY_PATTERN.findall(
            normalized
        )
    )

    years = frozenset(
        YEAR_PATTERN.findall(
            normalized
        )
    )

    durations = frozenset(
        _normalize_duration(
            marker
        )
        for marker in DURATION_PATTERN.findall(
            normalized
        )
    )

    program_terms = frozenset(
        _normalize_program_term(
            marker
        )
        for marker in PROGRAM_PATTERN.findall(
            normalized
        )
    )

    admission_modes = frozenset(
        _normalize_admission_mode(
            marker
        )
        for marker in ADMISSION_MODE_PATTERN.findall(
            normalized
        )
    )

    return FactualMarkers(
        percentages=percentages,
        scores=scores,
        money=money,
        years=years,
        durations=durations,
        program_terms=program_terms,
        admission_modes=admission_modes,
    )


# =========================================================
# Generic compatibility
# =========================================================

def _unsupported(
    answer_values: Iterable[str],
    evidence_values: Iterable[str],
) -> frozenset[str]:
    """
    Return answer markers absent from evidence.
    """

    evidence = set(
        evidence_values
    )

    return frozenset(
        value
        for value in answer_values
        if value not in evidence
    )


def _duration_supported(
    answer_duration: str,
    evidence_durations: Iterable[str],
) -> bool:
    """
    Determine whether a duration in the answer is represented in the
    evidence, allowing equivalent forms.
    """

    answer_normalized = _normalize_duration(
        answer_duration
    )

    answer_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*years?\b",
        answer_normalized,
    )

    if not answer_match:
        return (
            answer_normalized
            in set(evidence_durations)
        )

    answer_number = answer_match.group(
        1
    )

    for evidence_duration in evidence_durations:

        evidence_normalized = _normalize_duration(
            evidence_duration
        )

        if re.search(
            rf"\b{re.escape(answer_number)}\s*years?\b",
            evidence_normalized,
        ):
            return True

    return False


def _admission_mode_conflicts(
    answer_modes: Iterable[str],
    evidence_modes: Iterable[str],
) -> frozenset[str]:
    """
    Detect admission-mode markers that are explicitly different from
    the available evidence modes.

    Absence of a mode in a short evidence excerpt is not itself an
    error. A conflict is raised only when the evidence explicitly
    identifies a different mode.
    """

    answer_modes = set(
        answer_modes
    )

    evidence_modes = set(
        evidence_modes
    )

    if not answer_modes:
        return frozenset()

    if not evidence_modes:
        return frozenset()

    conflicts = set()

    # Normalize full-time/regular as related but not identical labels.
    normalized_evidence = set(
        evidence_modes
    )

    for answer_mode in answer_modes:

        if answer_mode in normalized_evidence:
            continue

        # An explicit alternative admission mode in the evidence is
        # enough to treat a different generated mode as risky.
        if answer_mode == "regular":
            if {
                "part-time",
                "sponsored",
                "external",
            } & normalized_evidence:
                conflicts.add(
                    answer_mode
                )

        elif answer_mode in {
            "part-time",
            "sponsored",
            "external",
        }:
            if "regular" in normalized_evidence:
                conflicts.add(
                    answer_mode
                )

    return frozenset(
        conflicts
    )


# =========================================================
# Main audit
# =========================================================

def audit_answer_claims(
    answer: str,
    context: str,
) -> dict:
    """
    Audit high-risk factual markers in a generated answer.

    Program/degree terms are retained for diagnostics but do not by
    themselves make an answer unsafe.
    """

    answer_markers = extract_factual_markers(
        answer
    )

    evidence_markers = extract_factual_markers(
        context
    )

    unsupported_percentages = _unsupported(
        answer_markers.percentages,
        evidence_markers.percentages,
    )

    unsupported_scores = _unsupported(
        answer_markers.scores,
        evidence_markers.scores,
    )

    unsupported_money = _unsupported(
        answer_markers.money,
        evidence_markers.money,
    )

    unsupported_years = _unsupported(
        answer_markers.years,
        evidence_markers.years,
    )

    unsupported_durations = frozenset(
        duration
        for duration in answer_markers.durations
        if not _duration_supported(
            duration,
            evidence_markers.durations,
        )
    )

    conflicting_admission_modes = (
        _admission_mode_conflicts(
            answer_markers.admission_modes,
            evidence_markers.admission_modes,
        )
    )

    # Program terms are intentionally diagnostic-only.
    unsupported_program_terms = frozenset()

    unsupported = {
        "percentages": unsupported_percentages,
        "scores": unsupported_scores,
        "money": unsupported_money,
        "years": unsupported_years,
        "durations": unsupported_durations,
        "program_terms": unsupported_program_terms,
        "admission_modes": conflicting_admission_modes,
    }

    total_unsupported = sum(
        len(values)
        for values in unsupported.values()
    )

    return {
        "status": (
            "unsafe"
            if total_unsupported
            else "supported"
        ),
        "answer_markers": answer_markers,
        "evidence_markers": evidence_markers,
        "unsupported": unsupported,
        "unsupported_count": total_unsupported,
    }


# =========================================================
# Production predicate
# =========================================================

def has_unsupported_high_risk_claims(
    answer: str,
    context: str,
) -> bool:
    """
    Return True when the generated answer contains a high-risk factual
    marker that is unsupported or explicitly conflicts with evidence.
    """

    result = audit_answer_claims(
        answer=answer,
        context=context,
    )

    return (
        result["status"]
        == "unsafe"
    )
