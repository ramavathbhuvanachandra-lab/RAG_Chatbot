"""
Phase 4 — Numeric and Temporal Evidence Compatibility

Purpose
-------
Determine whether quantitative and temporal evidence matches what the
user actually asked.

Important invariants
--------------------
- Units are matched semantically.
- Academic-year ranges are preserved from the original text.
- A request for "latest 2027" must not be satisfied merely because
  2027 is the endpoint of an AY 2026-2027 range.
- Deterministic and lightweight.
"""

import re
from typing import List, Set, Tuple

from backend.retriever import normalize_text


# =========================================================
# Unit categories
# =========================================================

UNIT_DAILY = "daily"
UNIT_WEEKLY = "weekly"
UNIT_MONTHLY = "monthly"
UNIT_YEARLY = "yearly"
UNIT_HOURLY = "hourly"
UNIT_UNKNOWN = "unknown"


UNIT_PATTERNS = {
    UNIT_DAILY: {
        r"\bper\s+day\b",
        r"\bper\s+diem\b",
        r"\bdaily\b",
        r"\bday\b",
    },
    UNIT_WEEKLY: {
        r"\bper\s+week\b",
        r"\bweekly\b",
        r"\bweek\b",
    },
    UNIT_MONTHLY: {
        r"\bper\s+month\b",
        r"\bmonthly\b",
        r"\bmonth\b",
    },
    UNIT_YEARLY: {
        r"\bper\s+year\b",
        r"\bannually\b",
        r"\bannual\b",
        r"\byearly\b",
        r"\byear\b",
    },
    UNIT_HOURLY: {
        r"\bper\s+hour\b",
        r"\bhourly\b",
        r"\bhour\b",
    },
}


# =========================================================
# Latest/current markers
# =========================================================

LATEST_MARKERS = {
    "latest",
    "current",
    "currently",
    "most recent",
    "newest",
    "updated",
}


# =========================================================
# Normalization
# =========================================================

def _normalized(
    text: str,
) -> str:
    return normalize_text(
        text
    )


# =========================================================
# Unit detection
# =========================================================

def detect_units(
    text: str,
) -> Set[str]:
    """
    Detect explicit time/rate units.
    """

    normalized = _normalized(
        text
    )

    units = set()

    for unit, patterns in UNIT_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                units.add(
                    unit
                )
                break

    return units


def detect_requested_unit(
    question: str,
) -> str:
    """
    Detect the requested rate unit.
    """

    normalized = _normalized(
        question
    )

    priority = [
        UNIT_DAILY,
        UNIT_WEEKLY,
        UNIT_MONTHLY,
        UNIT_YEARLY,
        UNIT_HOURLY,
    ]

    for unit in priority:

        for pattern in UNIT_PATTERNS[unit]:

            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                return unit

    return UNIT_UNKNOWN


def units_compatible(
    question: str,
    document_text: str,
) -> bool:
    """
    Determine whether explicit requested and evidence units agree.
    """

    requested_unit = detect_requested_unit(
        question
    )

    evidence_units = detect_units(
        document_text
    )

    if requested_unit == UNIT_UNKNOWN:
        return True

    if not evidence_units:
        return True

    return requested_unit in evidence_units


# =========================================================
# Year range detection
# =========================================================

def detect_year_ranges(
    text: str,
) -> List[Tuple[int, int]]:
    """
    Extract academic/calendar year ranges from the ORIGINAL text.

    Examples:
        2026-2027
        2026/2027
        2026 to 2027
    """

    raw = str(
        text or ""
    )

    ranges = []

    patterns = [
        r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b",
        r"\b((?:19|20)\d{2})\s*/\s*((?:19|20)\d{2})\b",
        r"\b((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b",
    ]

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            raw,
            flags=re.IGNORECASE,
        ):

            start = int(
                match.group(1)
            )

            end = int(
                match.group(2)
            )

            if end < start:
                start, end = end, start

            ranges.append(
                (
                    start,
                    end,
                )
            )

    return ranges


def detect_years(
    text: str,
) -> Set[int]:
    """
    Extract standalone years while excluding years already belonging
    to a detected range.
    """

    raw = str(
        text or ""
    )

    ranges = detect_year_ranges(
        raw
    )

    range_spans = []

    for pattern in [
        r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b",
        r"\b((?:19|20)\d{2})\s*/\s*((?:19|20)\d{2})\b",
        r"\b((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b",
    ]:

        for match in re.finditer(
            pattern,
            raw,
            flags=re.IGNORECASE,
        ):
            range_spans.append(
                (
                    match.start(),
                    match.end(),
                )
            )

    years = set()

    for match in re.finditer(
        r"\b(?:19|20)\d{2}\b",
        raw,
    ):

        position = match.start()

        inside_range = any(
            start <= position < end
            for start, end in range_spans
        )

        if inside_range:
            continue

        years.add(
            int(
                match.group(0)
            )
        )

    return years


# =========================================================
# Latest/current detection
# =========================================================

def asks_for_latest(
    question: str,
) -> bool:
    """
    Detect whether the question requests latest/current information.
    """

    normalized = _normalized(
        question
    )

    return any(
        marker in normalized
        for marker in LATEST_MARKERS
    )


# =========================================================
# Temporal compatibility
# =========================================================

def temporal_evidence_compatible(
    question: str,
    document_text: str,
) -> bool:
    """
    Determine whether dated evidence can answer the temporal question.

    Important rule:

        "latest fee for 2027"
        is NOT satisfied by
        "AY 2026-2027 fee"

    merely because 2027 occurs at the end of the academic-year range.

    An exact academic-year query can match an exact academic-year
    evidence range.
    """

    query_ranges = detect_year_ranges(
        question
    )

    query_years = detect_years(
        question
    )

    evidence_ranges = detect_year_ranges(
        document_text
    )

    evidence_years = detect_years(
        document_text
    )

    # -----------------------------------------------------
    # No explicit temporal target
    # -----------------------------------------------------

    if (
        not query_ranges
        and not query_years
    ):
        return True

    # -----------------------------------------------------
    # Exact academic-year query
    # -----------------------------------------------------

    if query_ranges:

        if not evidence_ranges:
            return False

        return any(
            query_range == evidence_range
            for query_range in query_ranges
            for evidence_range in evidence_ranges
        )

    # -----------------------------------------------------
    # Explicit calendar-year query
    # -----------------------------------------------------

    requested_year = max(
        query_years
    )

    # -----------------------------------------------------
    # Latest/current + explicit year
    # -----------------------------------------------------
    #
    # A year range is not treated as a standalone exact-year claim.
    # Therefore:
    #
    # query    = latest fee for 2027
    # evidence = AY 2026-2027
    #
    # => False
    # -----------------------------------------------------

    if asks_for_latest(
        question
    ):

        if requested_year in evidence_years:
            return True

        return False

    # -----------------------------------------------------
    # Non-latest explicit year
    #
    # A dated range can support a normal question about a year covered
    # by that range.
    # -----------------------------------------------------

    if requested_year in evidence_years:
        return True

    for start, end in evidence_ranges:

        if (
            start
            <= requested_year
            <= end
        ):
            return True

    return False


# =========================================================
# Combined compatibility
# =========================================================

def quantitative_temporal_compatible(
    question: str,
    document_text: str,
) -> bool:
    """
    Combined quantity + temporal compatibility.
    """

    if not units_compatible(
        question,
        document_text,
    ):
        return False

    if not temporal_evidence_compatible(
        question,
        document_text,
    ):
        return False

    return True