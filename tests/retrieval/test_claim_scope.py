"""
Phase 4 — Claim Scope Unit Tests
"""

from backend.claim_scope import (
    detect_claim_scopes,
    detect_admission_modes,
    has_scope_conflict,
)


def test_admission_scope():

    scopes = detect_claim_scopes(
        "What are the eligibility requirements for Ph.D. admission?"
    )

    assert "admission" in scopes


def test_financial_assistance_scope():

    scopes = detect_claim_scopes(
        "Can a student receive Ph.D. financial assistance?"
    )

    assert "financial_assistance" in scopes


def test_regular_admission_mode():

    modes = detect_admission_modes(
        "What are the requirements for regular Ph.D. admission?"
    )

    assert "regular" in modes


def test_part_time_admission_mode():

    modes = detect_admission_modes(
        "What are the requirements for part-time Ph.D. admission?"
    )

    assert "part_time" in modes


def test_financial_assistance_conflicts_with_admission():

    assert has_scope_conflict(
        "Can someone with a four-year bachelor's degree "
        "apply for regular Ph.D. admission?",
        "A B.Tech/B.S. holder with GATE may receive "
        "financial assistance."
    )


def test_part_time_conflicts_with_regular():

    assert has_scope_conflict(
        "What are the requirements for regular Ph.D. admission?",
        "For sponsored or part-time Ph.D. admission, "
        "two years of work experience is required."
    )


def test_direct_admission_does_not_conflict():

    assert not has_scope_conflict(
        "Can someone with a four-year bachelor's degree "
        "apply for regular Ph.D. admission?",
        "A bachelor's degree of minimum four-year duration "
        "with the required marks is eligible for regular "
        "Ph.D. admission."
    )
