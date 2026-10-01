"""Hard deterministic tests for E7.2 answer grounding in the reusable RAG core.

This test belongs to the new reusable architecture.

Legacy tests under tests/phase5 are intentionally not used here.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Make the repository root importable regardless of pytest import mode.
#
# This keeps the test self-contained and avoids depending on pytest's
# rootdir/sys.path discovery behavior.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.core.answering.grounding import (  # noqa: E402
    assess_answer_grounding,
    has_grounding_issue,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def assert_review(
    answer: str,
    evidence: str,
    issue_type: str | None = None,
) -> None:
    result = assess_answer_grounding(
        answer=answer,
        evidence=evidence,
    )

    assert result.status == "review", result.to_dict()

    if issue_type is not None:
        assert any(
            issue.issue_type == issue_type
            for issue in result.issues
        ), result.to_dict()


def assert_grounded(
    answer: str,
    evidence: str,
) -> None:
    result = assess_answer_grounding(
        answer=answer,
        evidence=evidence,
    )

    assert result.status == "grounded", result.to_dict()


# ---------------------------------------------------------------------------
# Eligibility / policy grounding
# ---------------------------------------------------------------------------

def test_supported_eligibility_policy_is_grounded() -> None:
    assert_grounded(
        "Applicants must have a qualifying degree.",
        "Applicants must have a qualifying degree and meet the stated criteria.",
    )


def test_personal_eligibility_conclusion_is_rejected() -> None:
    assert_review(
        "Based on that, you are eligible.",
        "Applicants must have a qualifying degree.",
        "eligibility_conclusion",
    )


def test_personal_application_claim_is_rejected_without_personal_facts() -> None:
    assert_review(
        "You can apply for this program.",
        "Applicants must hold the stated qualification.",
        "eligibility_conclusion",
    )


def test_personal_exemption_is_rejected_without_personal_category() -> None:
    assert_review(
        "You are exempt from the fee.",
        "Certain categories are exempt from tuition fees.",
        "exemption_conclusion",
    )


def test_supported_exemption_statement_is_grounded() -> None:
    assert_grounded(
        "The specified category is exempt from the fee.",
        "The specified category is exempt from tuition fees.",
    )


def test_do_not_flag_supported_program_policy_wording_as_personal_eligibility() -> None:
    assert_grounded(
        "Applicants are eligible when they meet the stated criteria.",
        "Applicants are eligible when they meet the stated criteria.",
    )


def test_personal_eligibility_with_explicit_evidence_about_the_user_can_be_grounded() -> None:
    assert_grounded(
        "You are eligible because you meet the stated criteria.",
        "You are eligible because you meet the stated criteria.",
    )


# ---------------------------------------------------------------------------
# Monetary grounding
# ---------------------------------------------------------------------------

def test_fabricated_money_value_is_rejected() -> None:
    assert_review(
        "The fee is ₹99,999.",
        "The fee is ₹50,000.",
        "unsupported_numeric_value",
    )


def test_supported_money_value_is_grounded() -> None:
    assert_grounded(
        "The fee is ₹50,000.",
        "Tuition Fee* ₹50,000/-",
    )


def test_unsupported_percent_is_rejected() -> None:
    assert_review(
        "The minimum is 75%.",
        "The minimum is 60%.",
        "unsupported_numeric_value",
    )


def test_unsupported_year_is_rejected() -> None:
    assert_review(
        "This applies in 2099.",
        "This applies in 2026.",
        "unsupported_numeric_value",
    )


def test_supported_year_is_grounded() -> None:
    assert_grounded(
        "This applies in 2026.",
        "The policy applies in 2026.",
    )


# ---------------------------------------------------------------------------
# Supported arithmetic / duration behavior
# ---------------------------------------------------------------------------

def test_number_word_duration_is_normalized_for_arithmetic() -> None:
    assert_grounded(
        "At ₹500 per day, five days would total ₹2,500.",
        "Single occupancy room rent is ₹500 per day.",
    )


def test_daily_rate_arithmetic_is_allowed_when_calculated_correctly() -> None:
    assert_grounded(
        "At ₹500 per day, five days would total ₹2,500.",
        "Single occupancy room rent is ₹500 per day.",
    )


def test_daily_rate_arithmetic_wrong_total_is_rejected() -> None:
    assert_review(
        "At ₹500 per day, five days would total ₹3,000.",
        "Single occupancy room rent is ₹500 per day.",
        "unsupported_numeric_value",
    )


def test_personal_total_is_rejected_when_existing_amount_lacks_personal_scope() -> None:
    assert_review(
        "Your exact total cost is ₹500.",
        "Room rent is ₹500 per day.",
        "personal_cost_interpretation",
    )


def test_new_personal_total_amount_is_rejected_as_unsupported_numeric_value() -> None:
    assert_review(
        "Your exact total cost is ₹2,500.",
        "Room rent is ₹500 per day.",
        "unsupported_numeric_value",
    )


def test_duration_specific_cost_is_rejected_when_duration_not_attached() -> None:
    assert_review(
        "Your stay for five days will cost ₹2,500.",
        "The room rate is ₹2,500.",
        "numeric_duration_scope",
    )


def test_occupancy_specific_cost_is_rejected_when_occupancy_not_attached() -> None:
    assert_review(
        "Your single occupancy room will cost ₹2,500.",
        "Room rent is ₹2,500.",
        "numeric_occupancy_scope",
    )


def test_explicit_total_scope_can_be_grounded_when_evidence_is_personal() -> None:
    assert_grounded(
        "Your total payment is ₹2,500.",
        "Your total payment is ₹2,500 for the stated service.",
    )


# ---------------------------------------------------------------------------
# Causal grounding
# ---------------------------------------------------------------------------

def test_unsupported_causal_claim_is_rejected() -> None:
    assert_review(
        "The fee is lower because the program is shorter.",
        "The fee is ₹50,000 and the program has four semesters.",
        "unsupported_causal_claim",
    )


def test_explicit_causal_evidence_is_allowed() -> None:
    assert_grounded(
        "The fee is lower because the program is shorter.",
        "The fee is lower because the program is shorter.",
    )


def test_causal_word_without_causal_assertion_is_not_falsely_flagged() -> None:
    assert_grounded(
        "The page lists the reason for the fee change.",
        "The page lists the reason for the fee change.",
    )


# ---------------------------------------------------------------------------
# IMPORTANT REGRESSION:
# Bare "so" must NOT be treated as a causal claim.
#
# The real M.Tech E2E answer previously contained normal language using
# "so", which was incorrectly converted into unsupported_causal_claim.
# ---------------------------------------------------------------------------

def test_bare_so_is_not_treated_as_unsupported_causal_claim() -> None:
    assert_grounded(
        "The required degree is a four-year engineering degree, so applicants "
        "should check the stated qualification.",
        "The required degree is a four-year engineering degree. "
        "Applicants should check the stated qualification.",
    )


def test_sentence_starting_with_so_is_not_treated_as_causal_evidence_failure() -> None:
    assert_grounded(
        "So the admission requirements are listed on the page.",
        "The admission requirements are listed on the page.",
    )


def test_realistic_answer_with_so_remains_grounded() -> None:
    assert_grounded(
        "Applicants need the required qualifying degree, so they should "
        "check the admission criteria before applying.",
        "Applicants need the required qualifying degree. "
        "The admission criteria are provided on the admission page.",
    )


# ---------------------------------------------------------------------------
# Multiple simultaneous issues
# ---------------------------------------------------------------------------

def test_multiple_grounding_issues_are_reported() -> None:
    result = assess_answer_grounding(
        answer=(
            "You are eligible and the fee is ₹99,999 "
            "because the course is shorter."
        ),
        evidence=(
            "Applicants must hold a qualifying degree. "
            "The fee is ₹50,000."
        ),
    )

    assert result.status == "review", result.to_dict()

    assert any(
        issue.issue_type == "eligibility_conclusion"
        for issue in result.issues
    ), result.to_dict()

    assert any(
        issue.issue_type == "unsupported_numeric_value"
        for issue in result.issues
    ), result.to_dict()

    assert any(
        issue.issue_type == "unsupported_causal_claim"
        for issue in result.issues
    ), result.to_dict()


# ---------------------------------------------------------------------------
# Normalization / utility behavior
# ---------------------------------------------------------------------------

def test_case_and_spacing_normalization() -> None:
    assert_grounded(
        "THE FEE IS ₹ 50,000.",
        "tuition fee ₹50,000/-",
    )


def test_empty_answer_has_no_grounding_issue() -> None:
    result = assess_answer_grounding(
        answer="",
        evidence="anything",
    )

    assert result.status == "grounded"
    assert result.issue_count == 0


def test_helper_matches_assessment() -> None:
    assert has_grounding_issue(
        answer="The fee is ₹99,999.",
        evidence="The fee is ₹50,000.",
    )

    assert not has_grounding_issue(
        answer="The fee is ₹50,000.",
        evidence="The fee is ₹50,000.",
    )


def test_plain_factual_sentence_without_high_risk_relation_is_not_blocked() -> None:
    assert_grounded(
        "The program has four semesters.",
        "The program has four semesters.",
    )


def test_generic_numeric_ratio_is_checked() -> None:
    assert_review(
        "The required score is 7/10.",
        "The required score is 6/10.",
        "unsupported_numeric_value",
    )


def test_supported_numeric_ratio_is_grounded() -> None:
    assert_grounded(
        "The required score is 6/10.",
        "The required score is 6/10.",
    )


# ---------------------------------------------------------------------------
# Direct execution support
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main([__file__, "-q"])
    )