"""
IIT Jodhpur V1 — Contradiction Handler

Purpose
-------
Detect material conflicts between evidence claims before answer
generation.

A contradiction requires the same factual slot:
    same subject
    + same context
    + same unit
    + compatible version
    + different value

The detector is intentionally conservative.
"""

from __future__ import annotations

import re
from typing import Any


# =========================================================
# Normalization
# =========================================================

def normalize(text: str) -> str:
    """Normalize whitespace and dash variants."""
    text = str(text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# Subject detection
# =========================================================

def subject(text: str) -> str | None:
    """
    Identify the specific factual subject represented by a claim.

    The important design rule is that subject detection must work on
    compact claims as well as full sentences.

    Example:
        "Regular Ph.D. applicants require at least 60% marks"
    must map to the Ph.D. eligibility subject even though the word
    "admission" is absent.
    """
    text = normalize(text)

    # Specific subjects first.
    if "gate" in text and (
        "phd" in text
        or "ph.d" in text
        or "doctoral" in text
    ):
        return "phd_gate"

    if (
        "phd" in text
        or "ph.d" in text
        or "doctoral" in text
    ):
        if (
            "eligib" in text
            or "applicant" in text
            or "qualifying degree" in text
            or "admission" in text
            or "marks" in text
            or "%" in text
            or "cgpa" in text
            or "cpi" in text
        ):
            return "phd_eligibility"

    if (
        "hostel room rent" in text
        or "room rent" in text
    ):
        return "hostel_room_rent"

    if (
        "hostel" in text
        and (
            "fee" in text
            or "charge" in text
            or "accommodation" in text
        )
    ):
        return "hostel_fee"

    if (
        "tuition fee" in text
        or "tuition fees" in text
    ):
        return "tuition_fee"

    if (
        "semester fee" in text
        or "semester fees" in text
    ):
        return "semester_fee"

    if (
        "application fee" in text
        or "application fees" in text
    ):
        return "application_fee"

    if (
        "admission fee" in text
        or "admission fees" in text
    ):
        return "admission_fee"

    if (
        "convocation fee" in text
        or "convocation fees" in text
    ):
        return "convocation_fee"

    if (
        "refundable deposit" in text
        or "security deposit" in text
    ):
        return "deposit"

    if "eligibility" in text:
        return "eligibility"

    return None


# =========================================================
# Context / slot dimensions
# =========================================================

def context_slot(text: str) -> dict[str, str | None]:
    """
    Extract explicit contextual dimensions.

    Missing dimensions remain None.
    """
    text = normalize(text)

    return {
        "occupancy": (
            "single"
            if "single occupancy" in text
            else (
                "double"
                if "double occupancy" in text
                else None
            )
        ),
        "resident": (
            "visitor"
            if re.search(r"\bvisitors?\b", text)
            else (
                "student"
                if re.search(r"\bstudents?\b", text)
                else (
                    "staff"
                    if re.search(r"\bstaff\b", text)
                    else (
                        "faculty"
                        if re.search(r"\bfaculty\b", text)
                        else None
                    )
                )
            )
        ),
        "bedding": (
            "without"
            if "without bedding" in text
            else (
                "with"
                if (
                    "with bedding" in text
                    or "with 1 bedding set" in text
                    or "one bedding set" in text
                )
                else None
            )
        ),
        "route": (
            "master"
            if (
                "master's degree" in text
                or "master degree" in text
                or "master's" in text
            )
            else (
                "bachelor"
                if (
                    "bachelor's degree" in text
                    or "bachelor degree" in text
                    or "bachelor's" in text
                    or "four-year bachelor's" in text
                )
                else None
            )
        ),
        "mode": (
            "part-time"
            if "part-time" in text
            else (
                "full-time"
                if "full-time" in text
                else (
                    "regular"
                    if re.search(r"\bregular\b", text)
                    else None
                )
            )
        ),
    }


def different_explicit_context(
    left: dict[str, str | None],
    right: dict[str, str | None],
) -> bool:
    """
    Return True only when the same explicit slot dimension is known on
    both sides and the values differ.
    """
    for key in (
        "occupancy",
        "resident",
        "bedding",
        "route",
        "mode",
    ):
        left_value = left.get(key)
        right_value = right.get(key)

        if (
            left_value is not None
            and right_value is not None
            and left_value != right_value
        ):
            return True

    return False


# =========================================================
# Version / time
# =========================================================

def version(text: str) -> set[str]:
    """Extract explicit academic-year ranges."""
    text = normalize(text)

    return {
        re.sub(
            r"\s+",
            "",
            value,
        )
        for value in re.findall(
            r"\b20\d{2}\s*-\s*20\d{2}\b",
            text,
        )
    }


def different_version(
    left: set[str],
    right: set[str],
) -> bool:
    """Return True when both claims explicitly use different versions."""
    return bool(
        left
        and right
        and left.isdisjoint(right)
    )


# =========================================================
# Units
# =========================================================

def unit(text: str) -> str:
    """
    Determine the factual unit represented by the claim.
    """
    text = normalize(text)

    if re.search(
        r"\bper\s+day\b|\b/day\b",
        text,
    ):
        return "per_day"

    if re.search(
        r"\bper\s+month\b|\bmonthly\b|\b/month\b",
        text,
    ):
        return "per_month"

    if re.search(
        r"\bper\s+semester\b|\b/semester\b",
        text,
    ):
        return "per_semester"

    if (
        "one-time" in text
        or "one time" in text
    ):
        return "one_time"

    if "%" in text:
        return "percentage"

    if "cgpa" in text or "cpi" in text:
        return "score"

    return "unknown"


# =========================================================
# Numeric extraction
# =========================================================

MONEY_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*"
    r"((?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)

PERCENT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%"
)

SCORE_RE = re.compile(
    r"(\d+(?:\.\d+)?)"
    r"(?:\s*/\s*\d+(?:\.\d+)?)?"
    r"\s*(?:cgpa|cpi)",
    re.IGNORECASE,
)


def numeric_claims(text: str) -> list[dict[str, Any]]:
    """Extract money, percentage, and score claims."""
    text = normalize(text)
    claims = []

    for match in MONEY_RE.finditer(text):
        claims.append({
            "value": float(
                match.group(1).replace(",", "")
            ),
            "kind": "money",
        })

    for match in PERCENT_RE.finditer(text):
        claims.append({
            "value": float(
                match.group(1)
            ),
            "kind": "percentage",
        })

    for match in SCORE_RE.finditer(text):
        claims.append({
            "value": float(
                match.group(1)
            ),
            "kind": "score",
        })

    return claims


# =========================================================
# Policy polarity
# =========================================================

def policy_polarity(
    text: str,
) -> str | None:
    """
    Extract explicit positive/negative policy language.
    """
    text = normalize(text)

    if re.search(
        r"\bnot\s+required\b"
        r"|\bnot\s+mandatory\b"
        r"|\bnot\s+permitted\b"
        r"|\bnot\s+allowed\b"
        r"|\bcannot\b",
        text,
    ):
        return "negative"

    if re.search(
        r"\brequired\b"
        r"|\bmandatory\b"
        r"|\bpermitted\b"
        r"|\ballowed\b",
        text,
    ):
        return "positive"

    return None


# =========================================================
# Pairwise contradiction
# =========================================================

def documents_contradict(
    left: str,
    right: str,
) -> bool:
    """
    Decide whether two documents contain materially conflicting claims.
    """
    left = normalize(left)
    right = normalize(right)

    left_subject = subject(left)
    right_subject = subject(right)

    if (
        left_subject is None
        or right_subject is None
        or left_subject != right_subject
    ):
        return False

    # Explicit versions distinguish claims.
    if different_version(
        version(left),
        version(right),
    ):
        return False

    left_context = context_slot(left)
    right_context = context_slot(right)

    # Different occupancy/resident/bedding/etc. means different facts.
    if different_explicit_context(
        left_context,
        right_context,
    ):
        return False

    # Numeric contradiction.
    left_numbers = numeric_claims(left)
    right_numbers = numeric_claims(right)

    left_unit = unit(left)
    right_unit = unit(right)

    if left_unit != right_unit:
        return False

    for left_claim in left_numbers:
        for right_claim in right_numbers:

            if (
                left_claim["kind"]
                == right_claim["kind"]
                and
                left_claim["value"]
                != right_claim["value"]
            ):
                return True

    # Policy contradiction.
    left_polarity = policy_polarity(left)
    right_polarity = policy_polarity(right)

    if (
        left_polarity is not None
        and right_polarity is not None
        and left_polarity != right_polarity
    ):
        return True

    return False


# =========================================================
# Public API
# =========================================================

def detect_contradictions(
    documents,
) -> dict:
    """
    Detect pairwise material contradictions.
    """
    unique = []
    seen = set()

    for document in documents or []:
        text = normalize(
            getattr(
                document,
                "page_content",
                "",
            )
        )

        if not text or text in seen:
            continue

        seen.add(text)
        unique.append(text)

    contradictions = []

    for i in range(
        len(unique)
    ):
        for j in range(
            i + 1,
            len(unique),
        ):
            if documents_contradict(
                unique[i],
                unique[j],
            ):
                contradictions.append({
                    "type": "material",
                    "document_a": i + 1,
                    "document_b": j + 1,
                    "claim_a": unique[i],
                    "claim_b": unique[j],
                })

    return {
        "has_contradiction": bool(
            contradictions
        ),
        "contradiction_count": len(
            contradictions
        ),
        "contradictions": contradictions,
    }
