"""
Phase 4 — Answer Claim Audit Integration Tests

These tests verify the production answer node refuses generated answers
that introduce unsupported high-risk factual markers.
"""

from backend.answer_claim_audit import (
    audit_answer_claims,
)


def test_supported_answer_passes_audit():

    result = audit_answer_claims(
        answer=(
            "The Ph.D. eligibility requirement is at least 60% "
            "marks or 6.0/10 CGPA."
        ),
        context=(
            "Applicants require at least 60% marks or 6.0/10 CGPA "
            "for the relevant master's degree route."
        ),
    )

    assert result["status"] == "supported"


def test_unsupported_percentage_fails_audit():

    result = audit_answer_claims(
        answer=(
            "Applicants need at least 70% marks."
        ),
        context=(
            "Applicants need at least 60% marks."
        ),
    )

    assert result["status"] == "unsafe"


def test_unsupported_fee_fails_audit():

    result = audit_answer_claims(
        answer=(
            "The hostel charge is ₹900 per day."
        ),
        context=(
            "Hostel room rent is ₹300 per day."
        ),
    )

    assert result["status"] == "unsafe"


def test_equivalent_duration_passes():

    result = audit_answer_claims(
        answer=(
            "The program lasts 4 years."
        ),
        context=(
            "The programme is a four-year program."
        ),
    )

    assert result["status"] == "supported"
