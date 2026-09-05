"""
Phase 5.5.1 — Numeric Scope Attachment Tests

These tests deliberately target production-style ambiguity.
"""

from backend.answer_grounding_scope import (
    assess_numeric_scope,
    has_numeric_scope_issue,
)


def test_published_long_stay_rate_is_allowed():
    result = assess_numeric_scope(
        answer=(
            "The published charge for stays exceeding 11 days "
            "but less than a month is ₹2,175 excluding GST."
        ),
        evidence=(
            "For stays exceeding 11 days but less than a month, "
            "Rs. 2175 is charged excluding GST."
        ),
    )

    assert result["status"] == "grounded"


def test_twenty_day_total_from_published_bracket_is_reviewed():
    result = assess_numeric_scope(
        answer=(
            "For a 20-day stay, the charge is ₹2,175. "
            "The cost is ₹2,175 for the entire stay."
        ),
        evidence=(
            "For stays exceeding 11 days but less than a month, "
            "Rs. 2175 is charged excluding GST."
        ),
    )

    assert result["status"] == "review"

    assert any(
        issue.issue_type
        == "numeric_duration_scope"
        for issue in result["issues"]
    )


def test_flat_fee_claim_requires_evidence():
    result = assess_numeric_scope(
        answer=(
            "The rate is a flat fee of ₹2,175 regardless of exact days."
        ),
        evidence=(
            "For stays exceeding 11 days but less than a month, "
            "Rs. 2175 is charged excluding GST."
        ),
    )

    assert result["status"] == "review"


def test_double_occupancy_attachment_is_reviewed_when_amount_is_not_scoped():
    result = assess_numeric_scope(
        answer=(
            "₹2,175 applies to double occupancy."
        ),
        evidence=(
            "Double occupancy hostel room rent is ₹300 per day. "
            "For stays exceeding 11 days but less than a month, "
            "Rs. 2175 is charged excluding GST."
        ),
    )

    assert result["status"] == "review"

    assert any(
        issue.issue_type
        == "numeric_occupancy_scope"
        for issue in result["issues"]
    )


def test_double_occupancy_amount_is_allowed_when_evidence_links_it():
    result = assess_numeric_scope(
        answer=(
            "Double occupancy costs ₹300 per day."
        ),
        evidence=(
            "Hostel room rent for double occupancy is ₹300 per day."
        ),
    )

    assert result["status"] == "grounded"


def test_single_occupancy_amount_is_not_confused_with_double():
    result = assess_numeric_scope(
        answer=(
            "Single occupancy costs ₹500 per day."
        ),
        evidence=(
            "Double occupancy costs ₹300 per day. "
            "Single occupancy costs ₹500 per day."
        ),
    )

    assert result["status"] == "grounded"


def test_supported_daily_arithmetic_is_allowed():
    result = assess_numeric_scope(
        answer=(
            "At ₹500 per day, five days would total ₹2,500."
        ),
        evidence=(
            "Single occupancy hostel room rent is ₹500 per day."
        ),
    )

    assert result["status"] == "grounded"


def test_unsupported_daily_rate_arithmetic_is_reviewed():
    result = assess_numeric_scope(
        answer=(
            "At ₹600 per day, five days would total ₹3,000."
        ),
        evidence=(
            "Single occupancy hostel room rent is ₹500 per day."
        ),
    )

    assert result["status"] == "review"


def test_wrong_arithmetic_is_not_accepted():
    result = assess_numeric_scope(
        answer=(
            "At ₹500 per day, five days would total ₹2,600."
        ),
        evidence=(
            "Single occupancy hostel room rent is ₹500 per day."
        ),
    )

    assert result["status"] == "review"


def test_personal_total_requires_explicit_evidence_mapping():
    result = assess_numeric_scope(
        answer=(
            "Your total hostel fee is ₹2,175."
        ),
        evidence=(
            "The hostel charge is ₹2,175 for stays exceeding "
            "11 days but less than a month."
        ),
    )

    assert result["status"] == "review"


def test_explicit_personal_total_can_pass():
    result = assess_numeric_scope(
        answer=(
            "Your total hostel fee is ₹2,175."
        ),
        evidence=(
            "Your total hostel fee is ₹2,175 for the approved stay."
        ),
    )

    assert result["status"] == "grounded"


def test_duration_without_exact_total_language_can_still_require_scope():
    result = assess_numeric_scope(
        answer=(
            "For a 20-day stay, the applicable charge is ₹2,175."
        ),
        evidence=(
            "Rs. 2175 applies to stays exceeding 11 days but less "
            "than a month."
        ),
    )

    assert result["status"] == "review"


def test_ordinary_non_numeric_answer_is_grounded():
    result = assess_numeric_scope(
        answer=(
            "The hostel provides Wi-Fi and common rooms."
        ),
        evidence=(
            "The hostel provides Wi-Fi and common rooms."
        ),
    )

    assert result["status"] == "grounded"


def test_boolean_predicate_matches_review():
    assert has_numeric_scope_issue(
        answer=(
            "Your 20-day stay costs ₹2,175."
        ),
        evidence=(
            "₹2,175 applies to stays exceeding 11 days "
            "but less than a month."
        ),
    )


def test_boolean_predicate_matches_grounded():
    assert not has_numeric_scope_issue(
        answer=(
            "The published charge is ₹2,175 for stays "
            "exceeding 11 days but less than a month."
        ),
        evidence=(
            "₹2,175 applies to stays exceeding 11 days "
            "but less than a month."
        ),
    )


def test_multiple_scope_failures_are_reported():
    result = assess_numeric_scope(
        answer=(
            "For a 20-day stay, ₹2,175 applies to double occupancy "
            "and is a flat fee for the entire stay."
        ),
        evidence=(
            "₹2,175 applies to stays exceeding 11 days but less "
            "than a month."
        ),
    )

    assert result["status"] == "review"
    assert result["issue_count"] >= 2


def test_realistic_student_room_rate_is_grounded():
    result = assess_numeric_scope(
        answer=(
            "For students, single occupancy without bedding is "
            "₹500 per day."
        ),
        evidence=(
            "Hostel accommodation charges for students/staff/faculty: "
            "single occupancy without bedding ₹500 per day."
        ),
    )

    assert result["status"] == "grounded"


def test_value_must_be_present_in_evidence_for_numeric_scope():
    result = assess_numeric_scope(
        answer=(
            "The hostel costs ₹900 per day."
        ),
        evidence=(
            "The hostel costs ₹500 per day."
        ),
    )

    assert result["status"] == "review"


def test_exact_total_language_without_duration_is_still_high_risk():
    result = assess_numeric_scope(
        answer=(
            "The exact total is ₹2,175."
        ),
        evidence=(
            "₹2,175 is the published charge for stays exceeding "
            "11 days but less than a month."
        ),
    )

    assert result["status"] == "review"
