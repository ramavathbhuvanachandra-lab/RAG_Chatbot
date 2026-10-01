"""Hard deterministic regression suite for the reusable retrieval ranker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.query.models import Query, RetrievalRequirement, Target
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
    RetrievalProvenance,
)
from backend.core.retrieval.ranking import (
    detect_query_scopes,
    rank_candidates,
    score_candidate,
)


@dataclass(frozen=True)
class Doc:
    page_content: str
    metadata: dict


def candidate(
    *,
    source: str,
    content: str,
    rrf: float,
    program: str | None = None,
    entity: str | None = None,
    topic: str | None = None,
    attributes: tuple[str, ...] = (),
    semantic: float = 0.0,
    coverage: float = 0.0,
    conflicts: tuple[str, ...] = (),
    noise: float = 0.0,
) -> RetrievalCandidate:
    return RetrievalCandidate.from_document(
        Doc(content, {"source": source}),
        source=source,
        meaning=DocumentMeaning(
            programs=(program,) if program else (),
            entities=(entity,) if entity else (),
            topics=(topic,) if topic else (),
            attributes=attributes,
        ),
        alignment=CandidateAlignment(
            semantic_match=semantic,
            coverage=coverage,
            conflicts=conflicts,
        ),
        quality=EvidenceQuality(noise=noise),
        provenance=RetrievalProvenance(
            rrf_score=rrf,
            primary_query="",
        ),
    )


def query_with_target(question: str, target: str) -> Query:
    return Query(
        original_query=question,
        target=Target(
            text=target,
            confidence=1.0,
            resolution_state="resolved",
        ),
        retrieval_requirement=RetrievalRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            reject_explicit_conflict=True,
            allow_partial_evidence=False,
            max_unmatched_required_facets=0,
        ),
    )


def test_scope_detection() -> None:
    assert "admissions" in detect_query_scopes("What is the admission process?")
    assert "fees" in detect_query_scopes("What is the student hostel fee?")
    assert "hostel" in detect_query_scopes("What are the hostel rules?")
    assert "research" in detect_query_scopes("What are the research areas?")


def test_broad_admission_prefers_general_institution_source() -> None:
    general = candidate(
        source="data/content/admissions/general_information.docx",
        content="General admissions information and application process for the institute.",
        rrf=0.010,
    )
    specialized = candidate(
        source="data/content/admissions/msc_admissions.docx",
        content="Admission process and selection procedure for the M.Sc. program.",
        rrf=0.020,
        program="msc",
    )
    ranked = rank_candidates("What is the admission process?", [specialized, general], top_k=2)
    assert ranked[0].source.endswith("admissions/general_information.docx"), ranked


def test_specific_target_beats_generic_source() -> None:
    general = candidate(
        source="data/content/admissions/general_information.docx",
        content="General admission requirements and application process.",
        rrf=0.020,
    )
    target = candidate(
        source="data/content/programs/program-a/admissions.docx",
        content="Admission requirements and application procedure for Program A.",
        rrf=0.010,
        program="program-a",
        semantic=1.0,
        coverage=1.0,
    )
    frame = query_with_target("What is the admission process for Program A?", "Program A")
    ranked = rank_candidates(frame.original_query, [general, target], query_frame=frame, top_k=2)
    assert ranked[0].meaning.programs == ("program-a",), ranked


def test_documents_attribute_beats_generic_admissions() -> None:
    generic = candidate(
        source="data/content/admissions/general_admissions.docx",
        content="General admission information and academic eligibility.",
        rrf=0.020,
    )
    documents = candidate(
        source="data/content/academic_administration/registration.docx",
        content="For registration, bring originals and copies of all necessary certificates, score cards, and proof of address.",
        rrf=0.010,
    )
    ranked = rank_candidates("What documents are required during admission?", [generic, documents], top_k=2)
    assert ranked[0].source.endswith("academic_administration/registration.docx"), ranked


def test_contact_attribute_beats_generic_admissions() -> None:
    generic = candidate(
        source="data/content/admissions/general_admissions.docx",
        content="Admission eligibility and application information.",
        rrf=0.020,
    )
    contact = candidate(
        source="data/content/programs/phd/general_information.docx",
        content="For online application queries, contact the academic office by email. General queries may be directed to the program coordinator.",
        rrf=0.010,
    )
    ranked = rank_candidates("How can I contact the admissions office?", [generic, contact], top_k=2)
    assert ranked[0].source.endswith("programs/phd/general_information.docx"), ranked


def test_student_hostel_fee_beats_booking_charges() -> None:
    student = candidate(
        source="data/content/finance/fees.docx",
        content="Student hostel fee is included in the semester fee; hostel room charges apply to students.",
        rrf=0.010,
    )
    booking = candidate(
        source="data/content/finance/fees.docx",
        content="Short-term accommodation booking charges are calculated per day per person.",
        rrf=0.020,
    )
    ranked = rank_candidates("What is the hostel fee for students?", [booking, student], top_k=2)
    assert "student" in ranked[0].document.page_content.casefold()
    assert "semester fee" in ranked[0].document.page_content.casefold()


def test_booking_query_allows_booking_charge_material() -> None:
    student = candidate(
        source="data/content/finance/fees.docx",
        content="Student hostel accommodation and semester fee information.",
        rrf=0.020,
    )
    booking = candidate(
        source="data/content/finance/fees.docx",
        content="Short-term accommodation booking charges are calculated per day per person.",
        rrf=0.010,
    )
    ranked = rank_candidates("What are the short-term hostel booking charges?", [student, booking], top_k=2)
    assert "booking charges" in ranked[0].document.page_content.casefold()


def test_hostel_rules_beats_finance_noise() -> None:
    finance = candidate(
        source="data/content/finance/fees.docx",
        content="Semester fees, hostel fee, refundable deposit and tuition charges.",
        rrf=0.020,
    )
    hostel = candidate(
        source="data/content/hostel_accommodation/general_information.docx",
        content="Hostel accommodation guidelines and rules for students, including permitted and prohibited activities.",
        rrf=0.010,
    )
    ranked = rank_candidates("hostel mein students ko kya rules follow karne hote hain?", [finance, hostel], top_k=2)
    assert ranked[0].source.endswith("hostel_accommodation/general_information.docx"), ranked


def test_minor_programs_prefers_program_source() -> None:
    programs = candidate(
        source="data/content/programs/minor_programs.docx",
        content="Minor Programs are offered to undergraduate students as a complement to their majors.",
        rrf=0.010,
    )
    dining = candidate(
        source="data/content/hostel_accommodation/general_information.docx",
        content="Dining options include vegetarian and non-vegetarian messes.",
        rrf=0.020,
    )
    ranked = rank_candidates("What minor programs are available?", [dining, programs], top_k=2)
    assert ranked[0].source.endswith("programs/minor_programs.docx"), ranked


def test_research_target_prefers_target_department() -> None:
    ee = candidate(
        source="data/content/departments/electrical_engineering/research.docx",
        content="Research areas include VLSI systems, signal integrity, neuromorphic computing and communications.",
        rrf=0.010,
        entity="electrical engineering",
    )
    other = candidate(
        source="data/content/departments/economics/research.docx",
        content="Research activities are organised around thematic areas in economics.",
        rrf=0.020,
        entity="economics",
    )
    frame = query_with_target(
        "What are the research areas offered by the Electrical Engineering department?",
        "Electrical Engineering department",
    )
    ranked = rank_candidates(frame.original_query, [other, ee], query_frame=frame, top_k=2)
    assert ranked[0].meaning.entities == ("electrical engineering",), ranked


def test_dining_beats_unrelated_admissions_content() -> None:
    dining = candidate(
        source="data/content/hostel_accommodation/general_information.docx",
        content="Dining Options include vegetarian and non-vegetarian messes and meal services.",
        rrf=0.010,
    )
    admissions = candidate(
        source="data/content/admissions/general_admissions.docx",
        content="Admission to academic programs and application requirements.",
        rrf=0.020,
    )
    ranked = rank_candidates("What food and dining facilities are available on campus?", [admissions, dining], top_k=2)
    assert ranked[0].source.endswith("hostel_accommodation/general_information.docx"), ranked


def test_emergency_medical_prefers_facilities() -> None:
    facilities = candidate(
        source="data/content/research_and_technology_facilities/facilities.docx",
        content="The Health Centre provides round-the-clock medical care and emergency healthcare services.",
        rrf=0.010,
    )
    fallback = candidate(
        source="data/content/fallback/knowledge_buffer_general.docx",
        content="General campus information with references to several schools and offices.",
        rrf=0.020,
    )
    ranked = rank_candidates("What emergency medical facilities are available?", [fallback, facilities], top_k=2)
    assert ranked[0].source.endswith("research_and_technology_facilities/facilities.docx"), ranked


def test_mtech_typo_keeps_admission_target() -> None:
    target = candidate(
        source="data/content/admissions/mtech_admissions.docx",
        content="Admission to M Tech Program. Applicants need a qualifying degree and follow the admission process.",
        rrf=0.010,
    )
    unrelated = candidate(
        source="data/content/departments/mechanical_engineering/general_information.docx",
        content="The department offers an M.Tech program and several academic courses.",
        rrf=0.020,
    )
    ranked = rank_candidates("Mtech regstration kaise hota hai?", [unrelated, target], top_k=2)
    assert ranked[0].source.endswith("admissions/mtech_admissions.docx"), ranked


def test_exact_duplicate_content_is_removed() -> None:
    first = candidate(
        source="/Users/example/data/content/programs/minor_programs.docx",
        content="Minor Programs are offered to undergraduate students.",
        rrf=0.020,
    )
    duplicate = candidate(
        source="data/content/programs/minor_programs.docx",
        content="Minor Programs are offered to undergraduate students.",
        rrf=0.019,
    )
    second_source = candidate(
        source="data/content/office/programs.docx",
        content="The institute offers undergraduate and postgraduate programs.",
        rrf=0.010,
    )
    ranked = rank_candidates("What programs are available?", [duplicate, first, second_source], top_k=3)
    fingerprints = [" ".join(item.document.page_content.split()).casefold() for item in ranked]
    assert fingerprints.count("minor programs are offered to undergraduate students.") == 1, ranked


def test_source_diversity_caps_repeated_chunks() -> None:
    source_items = [
        candidate(
            source="data/content/finance/fees.docx",
            content=f"Student fee section {i} with tuition and semester fee information.",
            rrf=0.030 - i * 0.001,
        )
        for i in range(4)
    ]
    other = candidate(
        source="data/content/hostel_accommodation/general_information.docx",
        content="Student hostel accommodation fee information.",
        rrf=0.010,
    )
    ranked = rank_candidates("What is the hostel fee for students?", source_items + [other], top_k=3)
    assert sum(item.source.endswith("finance/fees.docx") for item in ranked) <= 2, ranked


def test_conflicting_candidate_cannot_beat_aligned_candidate() -> None:
    aligned = candidate(
        source="data/content/admissions/program-a.docx",
        content="Program A admission requirements and application information.",
        rrf=0.010,
        program="program-a",
        semantic=1.0,
        coverage=1.0,
    )
    conflicting = candidate(
        source="data/content/admissions/program-b.docx",
        content="Program B admission requirements and application information.",
        rrf=0.050,
        program="program-b",
        semantic=1.0,
        coverage=1.0,
        conflicts=("target_mismatch",),
    )
    frame = query_with_target("What are the requirements for Program A?", "Program A")
    ranked = rank_candidates(frame.original_query, [conflicting, aligned], query_frame=frame, top_k=2)
    assert ranked[0].meaning.programs == ("program-a",), ranked


def test_unknown_term_does_not_create_semantic_alignment() -> None:
    item = candidate(
        source="data/content/facilities/general.docx",
        content="Laboratory facilities and campus infrastructure.",
        rrf=0.010,
        semantic=0.0,
        coverage=0.0,
    )
    scored = score_candidate(
        "Where is the qzxvultra laboratory?",
        item,
        candidates=[item],
    )
    # The opaque token cannot appear in semantic metadata or generate target
    # alignment. Legitimate lexical/facility evidence may still score.
    assert item.meaning.attributes == ()
    assert scored < 10.0


def test_provenance_survives_ranking() -> None:
    item = candidate(
        source="data/content/admissions/general_information.docx",
        content="General admissions information.",
        rrf=0.012345,
    )
    ranked = rank_candidates("What is the admission process?", [item], top_k=1)
    assert ranked[0].provenance.rrf_score == 0.012345
    assert ranked[0].document_id == item.document_id
    assert ranked[0].final_score >= 0.0


def test_contact_requires_actionable_contact_evidence() -> None:
    generic = candidate(
        source="data/content/admissions/general_admissions.docx",
        content="Admission eligibility and application information; applicants may contact offices as needed.",
        rrf=0.030,
    )
    contact = candidate(
        source="data/content/programs/phd/general_information.docx",
        content="For application queries, contact the academic office by email at office@example.edu; general queries may be directed to the program coordinator.",
        rrf=0.010,
    )
    ranked = rank_candidates("How can I contact the admissions office?", [generic, contact], top_k=2)
    assert ranked[0].source.endswith("programs/phd/general_information.docx"), ranked


def test_minor_program_target_beats_generic_program_index() -> None:
    generic = candidate(
        source="data/content/programs/general_information.docx",
        content="Programs General Information Doctor of Philosophy Programs and Core Sciences Departments.",
        rrf=0.030,
    )
    minor = candidate(
        source="data/content/schools/management_and_entrepreneurship/programs.docx",
        content="Minor Programs are offered to undergraduate students to complement their majors.",
        rrf=0.010,
    )
    ranked = rank_candidates("What minor programs are available?", [generic, minor], top_k=2)
    assert ranked[0].source.endswith("schools/management_and_entrepreneurship/programs.docx"), ranked


def test_raw_query_mtech_target_beats_rules_without_query_frame() -> None:
    rules = candidate(
        source="data/content/offices_and_administration/office_of_academics/rules_and_regulations.docx",
        content="Reservation Policy and amendments to academic regulations.",
        rrf=0.040,
    )
    programs = candidate(
        source="data/content/offices_and_administration/office_of_academics/programs.docx",
        content="M Tech programs and academic programs offered by the institute.",
        rrf=0.030,
    )
    target = candidate(
        source="data/content/admissions/mtech_admissions.docx",
        content="Admission to M Tech Program (Regular). Applicants follow the M Tech admission process.",
        rrf=0.010,
    )
    ranked = rank_candidates("Mtech regstration kaise hota hai?", [rules, programs, target], top_k=3)
    assert ranked[0].source.endswith("admissions/mtech_admissions.docx"), ranked


def test_short_term_booking_query_prefers_booking_evidence() -> None:
    student = candidate(
        source="data/content/finance/fees.docx",
        content="Student hostel fee is included in the semester fee.",
        rrf=0.030,
    )
    booking = candidate(
        source="data/content/finance/fees.docx",
        content="Accommodation Charges for short-term bookings not exceeding 10 days are calculated per day.",
        rrf=0.010,
    )
    ranked = rank_candidates("What are the short-term hostel booking charges?", [student, booking], top_k=2)
    assert "short-term bookings" in ranked[0].document.page_content.casefold(), ranked


def main() -> int:
    tests = [
        test_scope_detection,
        test_broad_admission_prefers_general_institution_source,
        test_specific_target_beats_generic_source,
        test_documents_attribute_beats_generic_admissions,
        test_contact_attribute_beats_generic_admissions,
        test_student_hostel_fee_beats_booking_charges,
        test_booking_query_allows_booking_charge_material,
        test_hostel_rules_beats_finance_noise,
        test_minor_programs_prefers_program_source,
        test_research_target_prefers_target_department,
        test_dining_beats_unrelated_admissions_content,
        test_emergency_medical_prefers_facilities,
        test_mtech_typo_keeps_admission_target,
        test_exact_duplicate_content_is_removed,
        test_source_diversity_caps_repeated_chunks,
        test_conflicting_candidate_cannot_beat_aligned_candidate,
        test_unknown_term_does_not_create_semantic_alignment,
        test_provenance_survives_ranking,
        test_contact_requires_actionable_contact_evidence,
        test_minor_program_target_beats_generic_program_index,
        test_raw_query_mtech_target_beats_rules_without_query_frame,
        test_short_term_booking_query_prefers_booking_evidence,
    ]
    for test in tests:
        test()
    print(f"RETRIEVAL RANKING HARD TESTS: PASS ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())