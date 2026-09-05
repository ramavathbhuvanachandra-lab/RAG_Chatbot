"""
Phase 4 — Answer Claim Audit Tests
"""

from backend.answer_claim_audit import (
    extract_factual_markers,
    audit_answer_claims,
    has_unsupported_high_risk_claims,
)


def test_percentage_supported_by_context():

    result = audit_answer_claims(
        answer="The requirement is 60%.",
        context="Applicants need at least 60% marks.",
    )

    assert result["status"] == "supported"


def test_unsupported_percentage_is_detected():

    result = audit_answer_claims(
        answer="The requirement is 70%.",
        context="Applicants need at least 60% marks.",
    )

    assert result["status"] == "unsafe"
    assert "70%" in result["unsupported"]["percentages"]


def test_money_supported_by_context():

    result = audit_answer_claims(
        answer="The fee is ₹300 per day.",
        context="Hostel charges are ₹300 per day.",
    )

    assert result["status"] == "supported"


def test_unsupported_money_is_detected():

    result = audit_answer_claims(
        answer="The fee is ₹900 per day.",
        context="Hostel charges are ₹300 per day.",
    )

    assert result["status"] == "unsafe"


def test_year_supported_by_context():

    result = audit_answer_claims(
        answer="The fee applies for 2026.",
        context="AY 2026-2027 charges are listed.",
    )

    assert result["status"] == "supported"


def test_unsupported_year_is_detected():

    result = audit_answer_claims(
        answer="The fee applies for 2028.",
        context="AY 2026-2027 charges are listed.",
    )

    assert result["status"] == "unsafe"


def test_duration_supported():

    result = audit_answer_claims(
        answer="The program lasts 4 years.",
        context="The programme is a four-year program.",
    )

    assert result["status"] == "supported"


def test_admission_mode_conflict_is_detected():

    result = audit_answer_claims(
        answer="Regular admission requires two years of work experience.",
        context="Part-time admission requires two years of work experience.",
    )

    assert result["status"] == "unsafe"


def test_program_marker_extraction():

    markers = extract_factual_markers(
        "The B.Tech and Ph.D. programs are available."
    )

    assert "btech" in markers.program_terms
    assert "phd" in markers.program_terms


def test_clean_answer_has_no_high_risk_issue():

    assert not has_unsupported_high_risk_claims(
        answer="Hostel facilities include Wi-Fi.",
        context="Hostel facilities include Wi-Fi.",
    )
