"""
IIT Jodhpur V1 — Phase 8D1
Entity Boundary Preservation Regression Tests

Protect the invariant that configured entities are matched as complete
entities rather than accidental substrings.

Compound configured entities are treated as the more specific entity for
the same text span. Therefore:

    "M.S. by Research" -> "m.s. by research"

and not an additional nested "m.s." entity.

A separate "M.S." occurrence elsewhere remains independently detectable.
"""

from __future__ import annotations

from backend.student_situation import (
    understand_student_situation,
)


def _entities(
    text: str,
) -> tuple[str, ...]:
    """Return extracted entities for one natural-language message."""
    situation = understand_student_situation(
        text
    )
    return situation.entities


def test_msc_does_not_match_ms():
    entities = _entities(
        "What are the admission routes for M.Sc.?"
    )

    assert "m.sc" in entities
    assert "m.s." not in entities


def test_msc_without_period_does_not_match_ms():
    entities = _entities(
        "What are the admission routes for M.Sc?"
    )

    assert "m.sc" in entities
    assert "m.s." not in entities


def test_msc_plain_alias_does_not_match_ms():
    entities = _entities(
        "Tell me about MSC admission."
    )

    assert "m.sc" in entities
    assert "m.s." not in entities


def test_ms_by_research_is_specific_compound_entity():
    entities = _entities(
        "Does IIT Jodhpur offer M.S. by Research?"
    )

    assert entities == (
        "m.s. by research",
    )


def test_ms_by_research_without_periods_is_specific_compound_entity():
    entities = _entities(
        "Does IIT Jodhpur offer MS by Research?"
    )

    assert entities == (
        "m.s. by research",
    )


def test_ms_by_research_with_partial_punctuation_is_specific_compound_entity():
    entities = _entities(
        "Does IIT Jodhpur offer M.S by Research?"
    )

    assert entities == (
        "m.s. by research",
    )


def test_short_ms_is_not_extracted_inside_compound_entity():
    entities = _entities(
        "M.S. by Research admissions"
    )

    assert entities == (
        "m.s. by research",
    )


def test_standalone_ms_is_detected():
    entities = _entities(
        "Does IIT Jodhpur offer M.S. admission?"
    )

    assert entities == (
        "m.s.",
    )


def test_mtech_is_preserved():
    entities = _entities(
        "What are the M.Tech admission requirements?"
    )

    assert "m.tech" in entities


def test_btech_is_preserved():
    entities = _entities(
        "What are the B.Tech admission requirements?"
    )

    assert "b.tech" in entities


def test_phd_is_preserved():
    entities = _entities(
        "What are the Ph.D. admission requirements?"
    )

    assert "phd" in entities


def test_multiple_distinct_entities_are_preserved():
    entities = _entities(
        "I am comparing M.Sc. and M.S. by Research."
    )

    assert entities == (
        "m.sc",
        "m.s. by research",
    )