"""
Phase 5.5 — Hard Evidence → Answer Grounding Tests

These tests target semantic relationship failures rather than simple
number presence.
"""

from backend.core.answering.grounding (
    assess_answer_grounding,
    has_grounding_issue,
)


# =========================================================
# Eligibility conclusions
# =========================================================

def test_percentage_does_not_automatically_prove_eligibility():
    result = assess_answer_grounding(
        answer=(
            "You have 74%, so you are eligible for regular Ph.D. admission."
        ),
        evidence=(
            "Applicants with a four-year bachelor's degree require "
            "at least 70% marks."
        ),
    )

    assert result["status"] == "review"
    assert any(
        issue.issue_type
        == "eligibility_conclusion"
        for issue in result["issues"]
    )


def test_explicit_eligibility_evidence_allows_eligibility_statement():
    result = assess_answer_grounding(
        answer=(
            "You are eligible for regular Ph.D. admission."
        ),
        evidence=(
            "Applicants with a four-year bachelor's degree and at least "
            "70% marks are eligible for regular Ph.D. admission."
        ),
    )

    assert result["status"] == "grounded"


def test_higher_percentage_does_not_invent_gates_exemption():
    result = assess_answer_grounding(
        answer=(
            "Since you have 74%, you are exempt from GATE."
        ),
        evidence=(
            "Applicants with a four-year bachelor's degree require "
            "at least 70% marks."
        ),
    )

    assert result["status"] == "review"

    assert any(
        issue.issue_type
        == "exemption_conclusion"
        for issue in result["issues"]
    )


# =========================================================
# Exemption / exception claims
# =========================================================

def test_explicit_exemption_evidence_is_allowed():
    result = assess_answer_grounding(
        answer=(
            "SC, ST and PwD students are exempt from tuition fees."
        ),
        evidence=(
            "SC, ST and PwD students are exempt from paying tuition fees."
        ),
    )

    assert result["status"] == "grounded"


def test_personal_exemption_claim_requires_explicit_basis():
    result = assess_answer_grounding(
        answer=(
            "You are exempt from tuition fees."
        ),
        evidence=(
            "Students in SC, ST and PwD categories are exempt from "
            "tuition fees."
        ),
    )

    assert result["status"] == "review"


# =========================================================
# Numeric interpretation
# =========================================================

def test_published_long_stay_amount_is_not_automatically_exact_personal_total():
    result = assess_answer_grounding(
        answer=(
            "Your 20-day hostel stay will cost exactly ₹2,175."
        ),
        evidence=(
            "For stays exceeding 11 days but less than a month, "
            "the charge is ₹2,175 excluding GST."
        ),
    )

    assert result["status"] == "review"

    assert any(
        issue.issue_type
        in {
            "personal_cost_interpretation",
            "unsupported_numeric_relationship",
        }
        for issue in result["issues"]
    )


def test_plain_published_rate_is_allowed():
    result = assess_answer_grounding(
        answer=(
            "The published charge for stays exceeding 11 days but less "
            "than a month is ₹2,175 excluding GST."
        ),
        evidence=(
            "For stays exceeding 11 days but less than a month, "
            "the charge is ₹2,175 excluding GST."
        ),
    )

    assert result["status"] == "grounded"


def test_supported_five_day_personal_calculation_can_remain_grounded():
    result = assess_answer_grounding(
        answer=(
            "At ₹500 per day, five days would total ₹2,500."
        ),
        evidence=(
            "Single occupancy hostel room rent is ₹500 per day."
        ),
    )

    # 5.5 does not attempt arithmetic verification. That belongs to
    # a later/explicit calculation control if needed.
    assert result["status"] == "grounded"


# =========================================================
# Causal relationships
# =========================================================

def test_unsupported_because_relationship_is_flagged():
    result = assess_answer_grounding(
        answer=(
            "Because your percentage is above the threshold, "
            "you are therefore exempt from GATE."
        ),
        evidence=(
            "The minimum percentage requirement is 70%."
        ),
    )

    assert result["status"] == "review"


def test_evidence_with_same_causal_relationship_can_pass():
    result = assess_answer_grounding(
        answer=(
            "You qualify because the policy states that applicants "
            "with a four-year degree and 70% marks are eligible."
        ),
        evidence=(
            "Applicants with a four-year degree and 70% marks are "
            "eligible because this route accepts four-year bachelor's "
            "degree holders."
        ),
    )

    # The evidence explicitly contains the causal relation.
    assert result["status"] == "grounded"


# =========================================================
# Safe ordinary answers
# =========================================================

def test_supported_numeric_answer_without_new_relationship_is_safe():
    result = assess_answer_grounding(
        answer=(
            "The hostel room rent is ₹300 per day for double occupancy."
        ),
        evidence=(
            "Double occupancy hostel room rent is ₹300 per day."
        ),
    )

    assert result["status"] == "grounded"


def test_descriptive_answer_is_safe():
    result = assess_answer_grounding(
        answer=(
            "The institute has hostel accommodation for students."
        ),
        evidence=(
            "Hostel accommodation is available for students."
        ),
    )

    assert result["status"] == "grounded"


def test_unrelated_personal_story_does_not_trigger_without_risky_claim():
    result = assess_answer_grounding(
        answer=(
            "You mentioned that you play football, but the hostel "
            "information covers accommodation charges."
        ),
        evidence=(
            "Hostel accommodation charges are listed by occupancy type."
        ),
    )

    assert result["status"] == "grounded"


# =========================================================
# Multiple issues
# =========================================================

def test_multiple_high_risk_relationships_are_reported():
    result = assess_answer_grounding(
        answer=(
            "You have 74%, so you are eligible for Ph.D. admission. "
            "You are also exempt from GATE."
        ),
        evidence=(
            "Applicants need at least 70% marks for the qualifying degree."
        ),
    )

    assert result["status"] == "review"
    assert result["issue_count"] >= 2


# =========================================================
# Production predicate
# =========================================================

def test_predicate_returns_true_for_review_case():
    assert has_grounding_issue(
        answer=(
            "You have 74%, so you are eligible."
        ),
        evidence=(
            "The minimum requirement is 70%."
        ),
    )


def test_predicate_returns_false_for_grounded_case():
    assert not has_grounding_issue(
        answer=(
            "The minimum requirement is 70%."
        ),
        evidence=(
            "Applicants need at least 70% marks."
        ),
    )


def test_empty_answer_is_safe():
    result = assess_answer_grounding(
        answer="",
        evidence=(
            "The hostel charge is ₹300 per day."
        ),
    )

    assert result["status"] == "grounded"
    assert result["issue_count"] == 0


# =========================================================
# Adversarial wording variants
# =========================================================

def test_hence_eligibility_is_detected():
    result = assess_answer_grounding(
        answer=(
            "The threshold is 70%, hence you are eligible."
        ),
        evidence=(
            "Applicants require at least 70% marks."
        ),
    )

    assert result["status"] == "review"


def test_therefore_exemption_is_detected():
    result = assess_answer_grounding(
        answer=(
            "You meet the percentage, therefore you are exempt."
        ),
        evidence=(
            "Applicants require at least 70% marks."
        ),
    )

    assert result["status"] == "review"


def test_personal_exact_fee_wording_is_high_risk():
    result = assess_answer_grounding(
        answer=(
            "Your exact hostel fee is ₹2,175."
        ),
        evidence=(
            "₹2,175 applies to stays exceeding 11 days but less than "
            "a month."
        ),
    )

    assert result["status"] == "review"


def test_published_fee_wording_is_not_personalized():
    result = assess_answer_grounding(
        answer=(
            "The published hostel charge is ₹2,175 for stays exceeding "
            "11 days but less than a month."
        ),
        evidence=(
            "₹2,175 applies to stays exceeding 11 days but less than "
            "a month."
        ),
    )

    assert result["status"] == "grounded"


# =========================================================
# Conservative boundary
# =========================================================

def test_unknown_semantic_statement_is_not_rejected_without_high_risk_pattern():
    result = assess_answer_grounding(
        answer=(
            "This option may be more suitable depending on your situation."
        ),
        evidence=(
            "The institute offers several academic options."
        ),
    )

    assert result["status"] == "grounded"
