"""
Phase 6 — Organizational Scope Regression Tests

Purpose
-------
Verify that organizational identity matching is precise enough to allow
legitimate department/school evidence while preserving protection against
scope collisions.
"""

from backend.claim_scope import (
    organizational_scope_conflict,
    has_scope_conflict,
)


def test_named_department_matches_query_without_department_word():

    query = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    evidence = (
        "The Department of Electrical Engineering "
        "research areas include MIMO communications, "
        "control systems, signal processing, VLSI, "
        "and cyber-physical systems."
    )

    assert not organizational_scope_conflict(
        query,
        evidence,
    )

    assert not has_scope_conflict(
        query,
        evidence,
    )


def test_generic_query_cannot_use_narrow_department_evidence():

    query = (
        "What research areas are available?"
    )

    evidence = (
        "The Department of Electrical Engineering "
        "research areas include control systems."
    )

    assert organizational_scope_conflict(
        query,
        evidence,
    )


def test_matching_school_identity_is_allowed():

    query = (
        "What are the Ph.D. requirements "
        "in the School of Artificial Intelligence "
        "and Data Science?"
    )

    evidence = (
        "The School of Artificial Intelligence "
        "and Data Science has specific Ph.D. "
        "eligibility requirements."
    )

    assert not organizational_scope_conflict(
        query,
        evidence,
    )


def test_different_school_identity_is_rejected():

    query = (
        "What are the Ph.D. requirements "
        "in the School of Artificial Intelligence "
        "and Data Science?"
    )

    evidence = (
        "The School of Electrical Engineering "
        "has specific Ph.D. eligibility requirements."
    )

    assert organizational_scope_conflict(
        query,
        evidence,
    )


def test_different_department_identity_is_rejected():

    query = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    evidence = (
        "The Department of Physics "
        "research areas include condensed matter."
    )

    assert organizational_scope_conflict(
        query,
        evidence,
    )


def test_matching_department_with_explicit_department_label_is_allowed():

    query = (
        "What research areas are available "
        "in the Department of Electrical Engineering?"
    )

    evidence = (
        "The Department of Electrical Engineering "
        "research areas include control systems."
    )

    assert not organizational_scope_conflict(
        query,
        evidence,
    )


def test_unrelated_generic_document_scope_is_not_blocked():

    query = (
        "What research areas are available?"
    )

    evidence = (
        "Research at the institute includes interdisciplinary "
        "work across engineering and science."
    )

    assert not organizational_scope_conflict(
        query,
        evidence,
    )


def test_generic_named_organization_matching():

    query = (
        "What research is available in Mechanical Engineering?"
    )

    evidence = (
        "The Department of Mechanical Engineering "
        "works on thermal systems and robotics."
    )

    assert not organizational_scope_conflict(
        query,
        evidence,
    )


def test_generic_different_organization_is_rejected():

    query = (
        "What research is available in Mechanical Engineering?"
    )

    evidence = (
        "The Department of Civil Engineering "
        "works on structural engineering."
    )

    assert organizational_scope_conflict(
        query,
        evidence,
    )