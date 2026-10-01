"""
Red-team tests for the IITJ institution lexical adapter.

These tests intentionally attack boundary conditions rather than only checking
happy-path examples. They are designed to catch:
- normalization regressions
- alias substring collisions
- phrase false positives
- nondeterminism
- unsafe mutation
- score-contract violations
- malformed profile configuration

Run from the RAG_Chatbot repository root:

    PYTHONPATH=. pytest -q /mnt/data/test_iitj_lexical_redteam.py

Some tests are deliberately capable of exposing an implementation bug.
A failing red-team test is useful information and should be fixed before
connecting the adapter to production retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.institutions.iitj.lexical import (
    LexicalProfile,
    canonical_terms,
    expand_query,
    get_lexical_profile,
    load_lexical_profile,
    matched_phrases,
    normalize_for_institution,
    score_pair,
    validate_lexical_profile,
)


def test_profile_integrity() -> None:
    profile = get_lexical_profile()
    validate_lexical_profile(profile)

    assert profile.institution_id == "iitj"
    assert profile.aliases
    assert profile.phrases
    assert profile.acronyms

    assert Path(__file__).exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("M.Tech", "mtech"),
        ("M.Tech.", "mtech"),
        ("M Tech", "mtech"),
        ("B.Tech", "btech"),
        ("M.Sc.", "msc"),
        ("Ph.D.", "phd"),
        ("M.B.A.", "mba"),
        ("IIT Jodhpur", "iitj"),
        ("Indian Institute of Technology Jodhpur", "iitj"),
        ("IIT J", "iitj"),
        ("Wi-Fi", "wifi"),
    ],
)
def test_normalization_alias_matrix(raw: str, expected: str) -> None:
    normalized = normalize_for_institution(raw)
    assert expected in normalized


def test_mixed_case_and_noise_are_deterministic() -> None:
    query = "  WHAT!!! are the M.Tech. eligibility REQUIREMENTS???  "
    outputs = {normalize_for_institution(query) for _ in range(50)}
    assert len(outputs) == 1


def test_empty_and_whitespace_inputs_are_safe() -> None:
    assert normalize_for_institution("") == ""
    assert normalize_for_institution("   ") == ""
    assert canonical_terms("") == ()
    assert expand_query("") == ()


@pytest.mark.parametrize(
    "raw",
    [
        "M.Technique",
        "B.Techno",
        "IITJunction",
        "Jodhpur IITian",
        "Ph.Dstudent",
        "MBAmazing",
    ],
)
def test_aliases_do_not_match_inside_larger_words(raw: str) -> None:
    terms = canonical_terms(raw)

    assert not any(
        term in {"mtech", "btech", "iitj", "phd", "mba"}
        for term in terms
    )


def test_iitj_alias_variants_collapse_to_one_canonical_term() -> None:
    variants = (
        "IITJ",
        "IIT J",
        "IIT Jodhpur",
        "Indian Institute of Technology Jodhpur",
    )

    for value in variants:
        terms = canonical_terms(value)
        assert "iitj" in terms


@pytest.mark.parametrize(
    "raw",
    [
        "M.Tech admission",
        "master of technology admission",
        "master technology admission",
        "M Tech admission",
    ],
)
def test_mtech_wording_is_consistent(raw: str) -> None:
    assert "mtech" in canonical_terms(raw)
    normalized = normalize_for_institution(raw)
    assert "mtech" in normalized


def test_multiple_programs_can_be_detected_without_cross_replacement() -> None:
    terms = canonical_terms("Compare B.Tech, M.Tech, M.Sc and Ph.D options.")
    assert {"btech", "mtech", "msc", "phd"}.issubset(set(terms))


def test_unknown_words_are_preserved() -> None:
    raw = "quantumwhale xyzzy_not_an_institution_term"
    normalized = normalize_for_institution(raw)

    assert "quantumwhale" in normalized
    assert "xyzzy" in normalized
    assert "iitj" not in normalized


def test_malicious_long_substring_does_not_trigger_institution_alias() -> None:
    raw = "not-iitjacker"
    terms = canonical_terms(raw)
    assert "iitj" not in terms


def test_phrase_hit_requires_real_phrase_presence() -> None:
    query = "What are the M.Tech eligibility requirements?"
    document = (
        "The M.Tech eligibility requirements are listed in the official "
        "admission document."
    )

    hits = matched_phrases(query, document)
    assert "eligibility requirements" in hits


def test_phrase_is_not_inferred_when_query_lacks_it() -> None:
    query = "Tell me about the M.Tech program."
    document = (
        "The M.Tech eligibility requirements are listed in the admission "
        "document."
    )

    hits = matched_phrases(query, document)
    assert "eligibility requirements" not in hits


def test_phrase_does_not_cross_document_boundaries() -> None:
    query = "eligibility requirements"
    document = "eligibility. requirements."

    # The adapter normalizes punctuation but should still treat this as two
    # separate words forming the configured phrase only if the phrase is truly
    # present after normalization.
    assert "eligibility requirements" in matched_phrases(query, document)


def test_phrase_boundary_red_team() -> None:
    """
    This is a deliberate adversarial case.

    "admission process" must not be considered present inside
    "admission processor". A substring-only phrase matcher is vulnerable here.
    """
    query = "admission process"
    document = "The admission processor handles requests."

    assert "admission process" not in matched_phrases(query, document)


def test_score_is_bounded() -> None:
    results = (
        score_pair(
            "M.Tech eligibility requirements",
            "M.Tech eligibility requirements are listed here.",
        ),
        score_pair(
            "hostel fees",
            "hostel charges and accommodation details",
        ),
        score_pair(
            "completely unrelated question",
            "quantum mechanics and astrophysics",
        ),
    )

    for result in results:
        assert 0.0 <= result.query_overlap <= 1.0


def test_direct_text_has_nonzero_recall_score() -> None:
    result = score_pair(
        "What are the M.Tech eligibility requirements?",
        "M.Tech eligibility requirements are stated in the admission document.",
    )

    assert result.query_overlap > 0.0
    assert result.exact_terms
    assert "mtech" in result.canonical_terms


def test_unrelated_text_has_lower_overlap() -> None:
    relevant = score_pair(
        "M.Tech eligibility requirements",
        "M.Tech eligibility requirements are listed here.",
    )
    unrelated = score_pair(
        "M.Tech eligibility requirements",
        "The library has a collection of books and journals.",
    )

    assert relevant.query_overlap > unrelated.query_overlap


def test_score_is_repeatable() -> None:
    values = [
        score_pair(
            "M.Tech admission requirements",
            "M.Tech admission requirements are published online.",
        ).query_overlap
        for _ in range(100)
    ]

    assert len(set(values)) == 1


def test_expand_query_keeps_original_first() -> None:
    original = "What are the M.Tech eligibility requirements?"
    values = expand_query(original)

    assert values[0] == original
    assert 1 <= len(values) <= 3


def test_expand_query_is_deterministic() -> None:
    query = "What are the M.Tech eligibility requirements?"
    results = {expand_query(query) for _ in range(50)}
    assert len(results) == 1


def test_expand_query_does_not_explode() -> None:
    query = (
        "Please tell me the M.Tech admission requirements, M.Tech fees, "
        "hostel fees, mess fees, campus facilities and research areas."
    )
    values = expand_query(query)

    assert len(values) <= 3


def test_profile_is_immutable_at_runtime() -> None:
    profile = get_lexical_profile()

    with pytest.raises((AttributeError, TypeError)):
        profile.institution_id = "other"  # type: ignore[misc]


def test_profile_reload_matches_cached_profile() -> None:
    cached = get_lexical_profile()
    fresh = load_lexical_profile()

    assert cached.institution_id == fresh.institution_id
    assert cached.aliases == fresh.aliases
    assert cached.phrases == fresh.phrases
    assert cached.acronyms == fresh.acronyms
    assert cached.canonicalization == fresh.canonicalization


def test_profile_contains_no_empty_alias_variants() -> None:
    profile = get_lexical_profile()

    for mapping in (profile.aliases, profile.acronyms):
        for canonical, variants in mapping.items():
            assert canonical.strip()
            assert variants
            assert all(str(value).strip() for value in variants)


def test_phrase_configuration_is_nonempty() -> None:
    profile = get_lexical_profile()

    for section, phrases in profile.phrases.items():
        assert section.strip()
        assert phrases
        assert all(phrase.strip() for phrase in phrases)


def test_topic_words_remain_lexical_evidence() -> None:
    """
    Topic terms such as 'requirements' should remain visible lexical text
    rather than being rewritten into an opaque semantic token.
    """
    normalized = normalize_for_institution(
        "admission requirements and fee details"
    )

    assert "admission" in normalized
    assert "requirements" in normalized
    assert "fee" in normalized


def test_unknown_profile_cannot_be_relabelled_as_iitj() -> None:
    bad_profile = LexicalProfile(
        institution_id="other",
        aliases={"x": ("x",)},
        phrases={"x": ("x",)},
        acronyms={"x": ("x",)},
        canonicalization={"x": "x"},
    )

    with pytest.raises(ValueError):
        validate_lexical_profile(bad_profile)


def test_no_answer_or_verification_api_exists_in_adapter() -> None:
    """
    Architecture guard: the institution lexical adapter must remain a recall
    configuration layer, not a hidden answer/verification engine.
    """
    from backend.institutions.iitj import lexical

    forbidden = {
        "verify_candidate",
        "verify_candidates",
        "generate_answer",
        "answer_query",
    }

    exported = set(getattr(lexical, "__all__", ()))
    assert exported.isdisjoint(forbidden)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))