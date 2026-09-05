"""
Phase 4 — Supported Claim Preservation Tests

Purpose
-------
Verify that materially important factual markers supported by the final
evidence are preserved by the generated answer.

This is not a generic text-overlap test.

It focuses on high-risk factual markers:
- money
- percentages
- CGPA / CPI
- durations
- program names

The system must tolerate normal wording differences while preventing
silent omission of important supported facts.
"""

from backend.supported_claim_preservation import (
    assess_supported_claim_preservation,
)


# =========================================================
# Money
# =========================================================

def test_supported_money_claim_is_preserved():

    result = assess_supported_claim_preservation(
        answer=(
            "Double occupancy costs ₹300 per day."
        ),
        context=(
            "Hostel room rent for double occupancy "
            "is ₹300 per day."
        ),
    )

    assert result["status"] == "supported"
    assert result["missing_claims"] == []


def test_missing_supported_money_claim_is_detected():

    result = assess_supported_claim_preservation(
        answer=(
            "Double occupancy costs ₹300 per day."
        ),
        context=(
            "Double occupancy costs ₹300 per day. "
            "Single occupancy costs ₹500 per day."
        ),
    )

    assert result["status"] == "incomplete"
    assert "500" in result["missing_claims"]


# =========================================================
# Percentages
# =========================================================

def test_supported_percentage_claims_are_preserved():

    result = assess_supported_claim_preservation(
        answer=(
            "The requirement is 60% for GEN/OBC "
            "and 55% for SC/ST."
        ),
        context=(
            "Applicants require 60% marks for GEN/OBC "
            "and 55% marks for SC/ST."
        ),
    )

    assert result["status"] == "supported"
    assert result["missing_claims"] == []


def test_missing_supported_percentage_is_detected():

    result = assess_supported_claim_preservation(
        answer=(
            "The requirement is 60% for GEN/OBC."
        ),
        context=(
            "Applicants require 60% for GEN/OBC "
            "and 55% for SC/ST."
        ),
    )

    assert result["status"] == "incomplete"
    assert "55%" in result["missing_claims"]


# =========================================================
# CGPA / CPI
# =========================================================

def test_cgpa_is_preserved():

    result = assess_supported_claim_preservation(
        answer=(
            "The minimum CGPA is 6.0/10."
        ),
        context=(
            "The minimum requirement is 6.0/10 CGPA."
        ),
    )

    assert result["status"] == "supported"


def test_missing_cgpa_is_detected():

    result = assess_supported_claim_preservation(
        answer=(
            "The minimum requirement is 60% marks."
        ),
        context=(
            "The minimum requirement is 60% marks "
            "or 6.0/10 CGPA."
        ),
    )

    assert result["status"] == "incomplete"
    assert "6.0/10" in result["missing_claims"]


# =========================================================
# Duration
# =========================================================

def test_duration_is_preserved():

    result = assess_supported_claim_preservation(
        answer=(
            "The program lasts four years."
        ),
        context=(
            "The program has a duration of four years."
        ),
    )

    assert result["status"] == "supported"


# =========================================================
# Program names
# =========================================================

def test_program_terms_are_preserved():

    result = assess_supported_claim_preservation(
        answer=(
            "The institute offers B.Tech and Ph.D. programs."
        ),
        context=(
            "The institute offers B.Tech and Ph.D. programs."
        ),
    )

    assert result["status"] == "supported"


def test_missing_program_term_is_detected():

    result = assess_supported_claim_preservation(
        answer=(
            "The institute offers B.Tech programs."
        ),
        context=(
            "The institute offers B.Tech and M.Tech programs."
        ),
    )

    assert result["status"] == "incomplete"
    assert "mtech" in result["missing_claims"]


# =========================================================
# No high-risk claims
# =========================================================

def test_no_high_risk_claims_is_safe():

    result = assess_supported_claim_preservation(
        answer=(
            "The institute provides academic and research opportunities."
        ),
        context=(
            "The institute provides academic and research opportunities."
        ),
    )

    assert result["status"] == "supported"
    assert result["missing_claims"] == []


# =========================================================
# Empty answer
# =========================================================

def test_empty_answer_is_incomplete_when_context_has_claims():

    result = assess_supported_claim_preservation(
        answer="",
        context=(
            "The hostel charge is ₹300 per day."
        ),
    )

    assert result["status"] == "incomplete"
    assert "300" in result["missing_claims"]


# =========================================================
# Exact false-positive protection
# =========================================================

def test_unrelated_context_number_is_not_required():

    result = assess_supported_claim_preservation(
        answer=(
            "The hostel has Wi-Fi."
        ),
        context=(
            "The hostel has Wi-Fi. "
            "The campus has 17 hostels."
        ),
    )

    assert result["status"] == "supported"
