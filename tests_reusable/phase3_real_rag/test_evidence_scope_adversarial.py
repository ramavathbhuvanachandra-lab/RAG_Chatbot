"""Adversarial stress suite for reusable evidence scope/conflict protection.

This suite intentionally attacks the scope layer with near-miss documents,
contradictory metadata, mixed domains, malformed inputs, target-only semantic
contracts, ordering/mutation checks, and boundary conditions.

The goal is not to prove that every possible interpretation is correct. The
goal is to make unsafe scope decisions difficult to introduce silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.evidence.scope import (
    assess_scope,
    filter_scope_conflicts,
    is_scope_compatible,
    signals_from_semantic,
)
from backend.core.query.models import Query, Target


@dataclass
class Document:
    page_content: str
    metadata: dict | None = None


def test_casefold_and_punctuation_are_not_scope_conflicts():
    assert is_scope_compatible(
        "WHAT ARE THE M.TECH FEES?",
        Document("m.tech fees and charges are published.")
    )


def test_program_degree_variants_normalize_consistently():
    for query, content in (
        ("What are M.Tech fees?", "M Tech fees are published."),
        ("What are M.Sc fees?", "M.Sc. fees are published."),
        ("What are Ph.D. fees?", "Ph D fees are published."),
        ("What are B.Tech fees?", "B.tech fees are published."),
    ):
        assert is_scope_compatible(query, Document(content))


def test_btech_does_not_match_mtech():
    assert not is_scope_compatible(
        "What are B.Tech fees?",
        Document("M.Tech fees are published here.")
    )


def test_mtech_does_not_match_msc():
    assert not is_scope_compatible(
        "What are M.Tech admission requirements?",
        Document("M.Sc. admission requirements are published here.")
    )


def test_phd_does_not_match_mtech():
    assert not is_scope_compatible(
        "What are Ph.D. admission requirements?",
        Document("M.Tech admission requirements are published here.")
    )


def test_hostel_domain_does_not_match_admission_domain_even_when_fee_matches():
    decision = assess_scope(
        "What are the hostel fees?",
        Document("Admission fee payment instructions are listed here.")
    )
    assert not decision.compatible
    assert "explicit_topic_mismatch" in decision.reasons


def test_research_domain_does_not_match_admission_domain():
    assert not is_scope_compatible(
        "What research areas are available?",
        Document("Admission requirements and eligibility are published here.")
    )


def test_facilities_domain_does_not_match_finance_only_document():
    assert not is_scope_compatible(
        "What facilities are available?",
        Document("Tuition fees and payment schedules are published here.")
    )


def test_hostel_fee_document_remains_compatible_with_hostel_facilities_query():
    # Scope protects domain, not attribute-level completeness. Coverage handles
    # whether a particular requested attribute is sufficiently represented.
    assert is_scope_compatible(
        "What hostel facilities are available?",
        Document("Hostel accommodation and room fee information are provided.")
    )


def test_broad_fee_query_allows_specialized_program_fee_document():
    assert is_scope_compatible(
        "What are the fees?",
        Document("M.Tech tuition fees and semester charges are listed here.")
    )


def test_broad_research_query_allows_any_department():
    for content in (
        "Department of Physics research areas include condensed matter.",
        "Department of Chemistry research areas include catalysis.",
        "Department of Electrical Engineering research areas include control.",
    ):
        assert is_scope_compatible("What research areas are available?", Document(content))


def test_explicit_department_mismatch_is_hard_conflict():
    decision = assess_scope(
        "What research areas are available in Electrical Engineering?",
        Document("Department of Mechanical Engineering research areas include thermofluids.")
    )
    assert not decision.compatible
    assert "explicit_organization_mismatch" in decision.reasons


def test_department_name_is_not_confused_by_shared_word_engineering():
    assert not is_scope_compatible(
        "What research areas are available in Electrical Engineering?",
        Document("Department of Chemical Engineering research areas include reaction engineering.")
    )


def test_same_department_with_extra_words_is_compatible():
    assert is_scope_compatible(
        "What research areas are available in Electrical Engineering?",
        Document("Department of Electrical Engineering and its research groups publish several research areas.")
    )


def test_school_mismatch_is_hard_conflict():
    assert not is_scope_compatible(
        "What programs are available in the School of Management?",
        Document("School of Engineering programs are listed here.")
    )


def test_same_school_is_compatible():
    assert is_scope_compatible(
        "What programs are available in the School of Management?",
        Document("School of Management programs include several academic offerings.")
    )


def test_regular_vs_part_time_is_hard_conflict():
    assert not is_scope_compatible(
        "What are the regular Ph.D. requirements?",
        Document("The part-time Ph.D. route has separate requirements.")
    )


def test_part_time_vs_regular_is_hard_conflict():
    assert not is_scope_compatible(
        "What are the part-time Ph.D. requirements?",
        Document("The regular Ph.D. route has separate requirements.")
    )


def test_same_mode_is_compatible():
    assert is_scope_compatible(
        "What are the regular Ph.D. requirements?",
        Document("The regular Ph.D. admission route requirements are described here.")
    )


def test_exact_year_range_rejects_wrong_range():
    assert not is_scope_compatible(
        "What are the fees for AY 2026-2027?",
        Document("Fee schedule for AY 2025-2026.")
    )


def test_exact_year_range_accepts_same_range_with_different_formatting():
    assert is_scope_compatible(
        "What are the fees for AY 2026-2027?",
        Document("Fees for AY 2026/2027 are listed here.")
    )


def test_ordinary_year_query_accepts_containing_academic_range():
    assert is_scope_compatible(
        "What were the fees in 2026?",
        Document("Fees for AY 2026-2027 are listed here.")
    )


def test_ordinary_year_query_rejects_unrelated_year():
    assert not is_scope_compatible(
        "What were the fees in 2026?",
        Document("Fees for AY 2024-2025 are listed here.")
    )


def test_unknown_terms_do_not_trigger_false_conflict():
    assert is_scope_compatible(
        "What is the zorbax policy?",
        Document("Student rules and regulations are published here.")
    )


def test_unknown_term_does_not_override_known_program_conflict():
    assert not is_scope_compatible(
        "What are M.Tech zorbax requirements?",
        Document("M.Sc. admission requirements are published here.")
    )


def test_metadata_cannot_override_explicit_program_in_text():
    decision = assess_scope(
        "What are M.Tech admission requirements?",
        Document(
            "M.Sc. admission requirements are published here.",
            {"meaning": {"programs": ["mtech"], "topics": ["admission"]}},
        ),
    )
    assert not decision.compatible
    assert "explicit_program_mismatch" in decision.reasons


def test_metadata_cannot_override_explicit_department_in_text():
    decision = assess_scope(
        "What research areas are available in Electrical Engineering?",
        Document(
            "Department of Mechanical Engineering research areas are published here.",
            {"meaning": {"entities": ["Electrical Engineering"], "topics": ["research"]}},
        ),
    )
    assert not decision.compatible
    assert "explicit_organization_mismatch" in decision.reasons


def test_metadata_cannot_override_explicit_mode_in_text():
    decision = assess_scope(
        "What are the regular Ph.D. requirements?",
        Document(
            "The part-time Ph.D. route has separate requirements.",
            {"meaning": {"qualifiers": ["regular"], "topics": ["admission"]}},
        ),
    )
    assert not decision.compatible
    assert "explicit_mode_mismatch" in decision.reasons


def test_metadata_can_enrich_missing_program_scope():
    doc = Document(
        "Admission requirements are published here.",
        {"meaning": {"programs": ["mtech"], "topics": ["admission"]}},
    )
    assert is_scope_compatible("What are M.Tech admission requirements?", doc)


def test_metadata_can_enrich_missing_organization_scope():
    doc = Document(
        "Research areas are published here.",
        {"meaning": {"entities": ["Electrical Engineering"], "topics": ["research"]}},
    )
    assert is_scope_compatible("What research areas are available in Electrical Engineering?", doc)


def test_metadata_non_mapping_is_safe():
    doc = Document("Hostel fees are listed here.", {"meaning": 42})
    assert is_scope_compatible("What are hostel fees?", doc)


def test_metadata_nested_mapping_is_safe():
    doc = Document(
        "Hostel fees are listed here.",
        {"meaning": {"programs": {"name": "mtech"}, "entities": {"label": "hostel"}}},
    )
    assert is_scope_compatible("What are hostel fees?", doc)


def test_structured_target_program_is_used_even_when_original_query_omits_it():
    query = Query(
        original_query="What are the admission requirements?",
        target=Target(text="M.Tech", entity_type="program", confidence=1.0, resolution_state="resolved"),
    )
    doc = Document("M.Sc. admission requirements are published here.")
    decision = assess_scope(query, doc)
    assert not decision.compatible
    assert "explicit_program_mismatch" in decision.reasons


def test_structured_target_program_accepts_matching_document():
    query = Query(
        original_query="What are the admission requirements?",
        target=Target(text="M.Tech", entity_type="program", confidence=1.0, resolution_state="resolved"),
    )
    assert is_scope_compatible(query, Document("M.Tech admission requirements are published here."))


def test_structured_target_department_is_used_even_when_original_query_omits_it():
    query = Query(
        original_query="What research areas are available?",
        target=Target(text="Electrical Engineering", entity_type="department", confidence=1.0, resolution_state="resolved"),
    )
    decision = assess_scope(query, Document("Department of Mechanical Engineering research areas are published here."))
    assert not decision.compatible
    assert "explicit_organization_mismatch" in decision.reasons


def test_unresolved_generic_target_does_not_create_accidental_program_conflict():
    query = Query(
        original_query="What are the admission requirements?",
        target=Target(text="advanced pathway", confidence=0.2, resolution_state="unresolved"),
    )
    assert is_scope_compatible(query, Document("Admission requirements are published here."))


def test_query_scope_object_is_supported():
    query = Query(
        original_query="What research areas are available?",
        scope=None,
    )
    assert is_scope_compatible(query, Document("Department research areas are published here."))


def test_filter_removes_only_conflicts_and_preserves_order():
    docs = [
        Document("M.Sc. admission requirements."),
        Document("M.Tech admission requirements."),
        Document("General admission information."),
        Document("M.Sc. admissions are listed here."),
    ]
    kept = filter_scope_conflicts("What are M.Tech admission requirements?", docs)
    assert [d.page_content for d in kept] == [
        "M.Tech admission requirements.",
        "General admission information.",
    ]


def test_filter_does_not_mutate_input_sequence():
    docs = [
        Document("M.Sc. admission requirements."),
        Document("M.Tech admission requirements."),
    ]
    snapshot = list(docs)
    filter_scope_conflicts("What are M.Tech admission requirements?", docs)
    assert docs == snapshot


def test_filter_preserves_object_identity_for_kept_documents():
    keep = Document("M.Tech admission requirements.")
    drop = Document("M.Sc. admission requirements.")
    kept = filter_scope_conflicts("What are M.Tech admission requirements?", [drop, keep])
    assert kept[0] is keep


def test_empty_query_is_safe():
    assert is_scope_compatible("", Document("Admission information."))


def test_empty_document_is_safe():
    assert is_scope_compatible("What is the fee?", Document(""))


def test_none_metadata_is_safe():
    assert is_scope_compatible("What is the fee?", Document("Fee information.", None))


def test_weird_non_string_content_is_safe():
    class OddDocument:
        page_content = 12345
        metadata = {"meaning": None}

    decision = assess_scope("What is the fee?", OddDocument())
    assert decision.compatible


def test_weird_query_mapping_is_safe():
    query = {"original_query": "What is the fee?", "target": None}
    assert is_scope_compatible(query, Document("Fee information."))


def test_multiple_query_domains_can_match_a_multidomain_document():
    assert is_scope_compatible(
        "Tell me about hostel rules and facilities.",
        Document("Hostel rules and facilities for residents are described here."),
    )


def test_multiple_query_domains_reject_wrong_single_domain():
    assert not is_scope_compatible(
        "Tell me about hostel rules and facilities.",
        Document("Admission requirements and eligibility are described here."),
    )


def test_contact_attribute_does_not_match_wrong_domain_when_domain_is_explicit():
    assert not is_scope_compatible(
        "How can I contact the hostel office?",
        Document("Admissions office contact details are listed here."),
    )


def test_contact_query_allows_general_office_contact_evidence_when_domain_not_explicit():
    assert is_scope_compatible(
        "How can I contact the office?",
        Document("The office contact details are listed here."),
    )


def test_mixed_program_document_requires_later_claim_level_filtering():
    # Scope can see that the requested program is present, but it cannot prove
    # that every sentence in a mixed-program document belongs to that target.
    # That distinction belongs to claims/coverage, not scope.
    assert is_scope_compatible(
        "What are M.Tech admission requirements?",
        Document("M.Tech programs overview. M.Sc. admission requirements are described separately."),
    )


def test_mixed_program_document_with_both_programs_is_not_rejected_by_program_scope_alone():
    # Both programs are explicitly present, so scope cannot decide which claim
    # is the relevant one. Coverage/claims must disambiguate later.
    assert is_scope_compatible(
        "What are M.Tech admission requirements?",
        Document("M.Tech programs are offered. M.Sc. admission requirements are described separately."),
    )


def test_exact_range_is_case_insensitive_to_ay_prefix():
    assert is_scope_compatible(
        "What are fees for 2026-2027?",
        Document("AY 2026-2027 fees are listed here."),
    )


def test_year_range_does_not_conflict_with_same_year_mention():
    assert is_scope_compatible(
        "What are fees for 2026-2027?",
        Document("In 2026, the fee schedule applies; AY 2026-2027 is current."),
    )


def test_signal_extraction_is_deterministic():
    first = signals_from_semantic({"programs": ["M.Tech"], "topics": ["admission"]})
    second = signals_from_semantic({"programs": ["M.Tech"], "topics": ["admission"]})
    assert first == second


def test_no_state_leaks_between_decisions():
    first = assess_scope("What are M.Tech fees?", Document("M.Tech fees are listed."))
    second = assess_scope("What are M.Tech fees?", Document("M.Sc. fees are listed."))
    third = assess_scope("What are M.Sc fees?", Document("M.Sc. fees are listed."))
    assert first.compatible is True
    assert second.compatible is False
    assert third.compatible is True


def main() -> int:
    tests = [
        value for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - test harness needs full failure report
            failures.append((test.__name__, exc))

    print(f"EVIDENCE SCOPE ADVERSARIAL HARD TESTS: {len(tests) - len(failures)}/{len(tests)} PASS")
    if failures:
        for name, exc in failures:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())