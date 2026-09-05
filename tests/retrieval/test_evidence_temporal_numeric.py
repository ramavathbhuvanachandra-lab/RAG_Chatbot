"""
Phase 4 — Numeric / Temporal Evidence Tests
"""

from backend.evidence_temporal_numeric import (
    detect_requested_unit,
    units_compatible,
    detect_years,
    asks_for_latest,
    temporal_evidence_compatible,
    quantitative_temporal_compatible,
)


def test_detect_daily_unit():

    assert (
        detect_requested_unit(
            "What is the daily hostel fee?"
        )
        == "daily"
    )


def test_detect_monthly_unit():

    assert (
        detect_requested_unit(
            "What is the monthly hostel fee?"
        )
        == "monthly"
    )


def test_daily_and_monthly_are_incompatible():

    assert not units_compatible(
        "What is the daily hostel fee?",
        "The charge is ₹2,175 per month.",
    )


def test_monthly_and_monthly_are_compatible():

    assert units_compatible(
        "What is the monthly hostel fee?",
        "The charge is ₹2,175 per month.",
    )


def test_requested_year_must_match_evidence():

    assert not temporal_evidence_compatible(
        "What is the latest hostel fee for 2027?",
        "AY 2026-2027 charges are ₹2,175 per month.",
    )

def test_matching_academic_year_is_temporally_compatible():

    assert temporal_evidence_compatible(
        "What is the hostel fee for AY 2025-2026?",
        "AY 2025-2026 charges are ₹2,175 per month.",
    )

def test_latest_without_explicit_year_remains_permissive():

    assert temporal_evidence_compatible(
        "What is the latest hostel fee?",
        "AY 2026-2027 charges are ₹2,175 per month.",
    )


def test_combined_numeric_temporal_compatibility():

    assert not quantitative_temporal_compatible(
        "What is the daily hostel fee for 2027?",
        "AY 2026-2027 charges are ₹2,175 per month.",
    )
