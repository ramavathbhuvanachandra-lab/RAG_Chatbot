"""Standalone tests for the IITJ institution lexical adapter."""

from __future__ import annotations

from backend.institutions.iitj.lexical import (
    canonical_terms,
    expand_query,
    get_lexical_profile,
    matched_phrases,
    normalize_for_institution,
    score_pair,
    validate_lexical_profile,
)


def test_profile_is_valid() -> None:
    validate_lexical_profile()
    profile = get_lexical_profile()
    assert profile.institution_id == "iitj"
    assert "mtech" in profile.aliases
    assert "admission" in profile.aliases
    assert "iitj" in profile.acronyms


def test_program_aliases_canonicalize() -> None:
    assert "mtech" in canonical_terms("What are the M.Tech eligibility requirements?")
    assert "msc" in canonical_terms("What is the M.Sc admission route?")
    assert "phd" in canonical_terms("What are the Ph.D requirements?")
    assert "btech" in canonical_terms("B.Tech admission requirements")


def test_institution_name_is_canonicalized() -> None:
    normalized = normalize_for_institution("Indian Institute of Technology Jodhpur")
    assert "iitj" in normalized


def test_phrase_hits_are_shared_only_when_present_in_both_texts() -> None:
    query = "What are the M.Tech eligibility requirements?"
    document = "The M.Tech eligibility requirements are listed in the admission document."
    assert "eligibility requirements" in matched_phrases(query, document)

    unrelated = "The hostel facilities are available to students."
    assert "eligibility requirements" not in matched_phrases(query, unrelated)


def test_score_is_recall_signal_only() -> None:
    result = score_pair(
        "What are the M.Tech eligibility requirements?",
        "M.Tech eligibility requirements are stated in the admission document.",
    )
    assert result.query_overlap > 0.0
    assert "mtech" in result.canonical_terms
    assert result.matched_phrases


def test_query_expansion_is_bounded_and_original_first() -> None:
    values = expand_query("What are the M.Tech eligibility requirements?")
    assert values
    assert values[0] == "What are the M.Tech eligibility requirements?"
    assert len(values) <= 3


if __name__ == "__main__":
    tests = (
        test_profile_is_valid,
        test_program_aliases_canonicalize,
        test_institution_name_is_canonicalized,
        test_phrase_hits_are_shared_only_when_present_in_both_texts,
        test_score_is_recall_signal_only,
        test_query_expansion_is_bounded_and_original_first,
    )
    for test in tests:
        test()
    print(f"PASS: {len(tests)} IITJ lexical adapter tests")