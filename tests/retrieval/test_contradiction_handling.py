"""
Phase 4 — Contradiction Handling Tests

Purpose
-------
Verify that materially conflicting evidence is detected while
legitimate contextual differences are preserved.

The system must:

- detect conflicting numeric claims about the same fact,
- detect conflicting boolean/policy claims,
- avoid flagging legitimate category differences,
- avoid flagging unrelated numbers,
- remain deterministic,
- remain conservative.
"""

from langchain_core.documents import Document

from backend.contradiction_handler import (
    detect_contradictions,
)


def doc(text):
    return Document(
        page_content=text,
        metadata={},
    )


# =========================================================
# Numeric contradictions
# =========================================================

def test_conflicting_percentage_claims_are_detected():

    result = detect_contradictions(
        [
            doc(
                "Regular Ph.D. applicants require at least "
                "60% marks in the qualifying degree."
            ),
            doc(
                "Regular Ph.D. applicants require at least "
                "70% marks in the qualifying degree."
            ),
        ]
    )

    assert result["has_contradiction"] is True
    assert result["contradiction_count"] >= 1


def test_conflicting_fee_claims_are_detected():

    result = detect_contradictions(
        [
            doc(
                "Hostel room rent is ₹300 per day "
                "for double occupancy."
            ),
            doc(
                "Hostel room rent is ₹500 per day "
                "for double occupancy."
            ),
        ]
    )

    assert result["has_contradiction"] is True


# =========================================================
# Legitimate contextual differences
# =========================================================

def test_different_hostel_occupancy_rates_are_not_contradictions():

    result = detect_contradictions(
        [
            doc(
                "Double occupancy hostel room rent is ₹300 per day."
            ),
            doc(
                "Single occupancy hostel room rent is ₹500 per day."
            ),
        ]
    )

    assert result["has_contradiction"] is False


def test_different_resident_categories_are_not_contradictions():

    result = detect_contradictions(
        [
            doc(
                "Hostel room rent for students is ₹300 per day."
            ),
            doc(
                "Hostel room rent for visitors is ₹450 per day."
            ),
        ]
    )

    assert result["has_contradiction"] is False


def test_different_degree_routes_are_not_contradictions():

    result = detect_contradictions(
        [
            doc(
                "Applicants with a master's degree require "
                "at least 60% marks."
            ),
            doc(
                "Applicants using the four-year bachelor's route "
                "require at least 70% marks."
            ),
        ]
    )

    assert result["has_contradiction"] is False


# =========================================================
# Boolean / policy contradictions
# =========================================================

def test_conflicting_policy_claims_are_detected():

    result = detect_contradictions(
        [
            doc(
                "GATE is required for regular Ph.D. admission."
            ),
            doc(
                "GATE is not required for regular Ph.D. admission."
            ),
        ]
    )

    assert result["has_contradiction"] is True


# =========================================================
# Unrelated numbers
# =========================================================

def test_unrelated_numbers_are_not_contradictions():

    result = detect_contradictions(
        [
            doc(
                "The institute has 17 student hostels."
            ),
            doc(
                "The Central Research Facility was established in 2018."
            ),
        ]
    )

    assert result["has_contradiction"] is False


# =========================================================
# Empty / single evidence
# =========================================================

def test_empty_documents_are_safe():

    result = detect_contradictions([])

    assert result["has_contradiction"] is False
    assert result["contradiction_count"] == 0


def test_single_document_cannot_contradict_itself():

    result = detect_contradictions(
        [
            doc(
                "The hostel provides Wi-Fi and LAN connectivity."
            )
        ]
    )

    assert result["has_contradiction"] is False
    assert result["contradiction_count"] == 0


# =========================================================
# Result contract
# =========================================================

def test_result_contains_required_fields():

    result = detect_contradictions(
        [
            doc("The minimum requirement is 60%."),
            doc("The minimum requirement is 70%."),
        ]
    )

    assert "has_contradiction" in result
    assert "contradiction_count" in result
    assert "contradictions" in result
