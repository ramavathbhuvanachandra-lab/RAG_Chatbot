"""
Phase 4 — Multi-Intent Decomposition Tests

These tests aggressively probe:
- true multi-intent questions
- coordinated single intents
- anaphoric references
- misleading conjunctions
- concise user language
- scope inheritance
"""

from backend.multi_intent import (
    decompose_multi_intent,
    intent_questions,
    is_multi_intent,
)


def questions(
    text: str,
):
    return [
        unit.question
        for unit in decompose_multi_intent(
            text
        )
    ]


# =========================================================
# True multi-intent questions
# =========================================================

def test_phd_and_hostel_are_split():

    result = questions(
        "What are the Ph.D. eligibility requirements "
        "and what are the hostel fees?"
    )

    assert len(result) == 2
    assert (
        "ph.d" in result[0].lower()
        or "phd" in result[0].lower()
    )
    assert "hostel" in result[1].lower()


def test_two_explicit_questions_are_split():

    result = questions(
        "What are the hostel facilities? "
        "And what are the hostel fees?"
    )

    assert len(result) == 2


def test_admission_and_fee_are_split():

    result = questions(
        "What are the Ph.D. admission requirements "
        "and what is the application fee?"
    )

    assert len(result) == 2


def test_research_and_programs_are_split():

    result = questions(
        "What research areas are available in Electrical Engineering "
        "and what programs are offered there?"
    )

    assert len(result) == 2


def test_anaphoric_scope_is_inherited():

    result = questions(
        "What research areas are available in Electrical Engineering "
        "and what programs are offered there?"
    )

    assert (
        "electrical engineering"
        in result[1].lower()
    )


def test_school_scope_is_inherited():

    result = questions(
        "What programs are available through the School of "
        "Artificial Intelligence and Data Science and what research "
        "is being done there?"
    )

    assert len(result) == 2

    assert (
        "artificial intelligence"
        in result[1].lower()
    )


# =========================================================
# Single-intent coordinated language
# =========================================================

def test_fees_and_charges_are_one_intent():

    result = questions(
        "What are the fees and charges for hostel accommodation?"
    )

    assert len(result) == 1


def test_costs_and_charges_are_one_intent():

    result = questions(
        "What are the costs and charges for hostel accommodation?"
    )

    assert len(result) == 1


def test_research_themes_and_areas_are_one_intent():

    result = questions(
        "What research themes and areas are available?"
    )

    assert len(result) == 1


def test_facilities_and_amenities_are_one_intent():

    result = questions(
        "What facilities and amenities are available?"
    )

    assert len(result) == 1


# =========================================================
# Single questions
# =========================================================

def test_single_question_remains_single():

    result = questions(
        "What research areas are available in Electrical Engineering?"
    )

    assert len(result) == 1


def test_single_program_question_remains_single():

    result = questions(
        "What programs are offered at IIT Jodhpur?"
    )

    assert len(result) == 1


# =========================================================
# Conservative false-positive protection
# =========================================================

def test_random_and_does_not_force_split():

    result = questions(
        "Tell me about the hostel and campus."
    )

    assert len(result) == 1


def test_and_inside_statement_does_not_force_split():

    result = questions(
        "The hostel has Wi-Fi and dining facilities."
    )

    assert len(result) == 1


def test_research_and_facilities_compound():

    result = questions(
        "What are the research facilities and laboratories?"
    )

    assert len(result) == 1


# =========================================================
# Concise user-style language
# =========================================================

def test_concise_two_part_question():

    result = questions(
        "Hostel fees and Ph.D. eligibility?"
    )

    assert len(result) == 2


def test_question_with_three_compact_parts_is_conservative():

    result = questions(
        "What are the hostel facilities, fees, and rules?"
    )

    assert len(result) in {
        1,
        2,
    }


# =========================================================
# Empty input
# =========================================================

def test_empty_question():

    result = decompose_multi_intent(
        ""
    )

    assert result == []


def test_whitespace_question():

    result = decompose_multi_intent(
        "   "
    )

    assert result == []


# =========================================================
# Helpers
# =========================================================

def test_is_multi_intent():

    assert is_multi_intent(
        "What are Ph.D. eligibility requirements "
        "and what are hostel fees?"
    )


def test_is_not_multi_intent():

    assert not is_multi_intent(
        "What are the fees and charges for hostel accommodation?"
    )


def test_intent_questions_returns_strings():

    result = intent_questions(
        "What are the Ph.D. requirements "
        "and what are the hostel fees?"
    )

    assert all(
        isinstance(
            value,
            str,
        )
        for value in result
    )
