"""
Phase 8B — Broad Institutional Retrieval Tests
"""

from langchain_core.documents import Document

from backend.broad_institutional_retrieval import (
    assemble_broad_candidates,
    detect_broad_scope,
    is_broad_institutional_question,
    score_broad_candidate,
)


# =========================================================
# Helpers
# =========================================================

def make_doc(
    text: str,
    source: str,
) -> Document:

    return Document(
        page_content=text,
        metadata={
            "source": source,
        },
    )


# =========================================================
# Scope Detection
# =========================================================

def test_program_broad_scope_is_detected():

    scope = detect_broad_scope(
        "What academic programs are available?"
    )

    assert scope is not None
    assert scope.name == "programs"


def test_research_broad_scope_is_detected():

    scope = detect_broad_scope(
        "What research opportunities are available?"
    )

    assert scope is not None
    assert scope.name == "research"


def test_focused_question_is_not_broad():

    assert not is_broad_institutional_question(
        "What is the eligibility for M.Tech?"
    )


# =========================================================
# Scoring
# =========================================================

def test_program_index_beats_specific_admission_document():

    query = (
        "What academic programs are available?"
    )

    index_doc = make_doc(
        (
            "Academic programs include B.Tech, "
            "M.Tech, M.Sc., Ph.D., and MBA."
        ),
        "/programs/programs.docx",
    )

    specific_doc = make_doc(
        (
            "M.Tech admission requires an application."
        ),
        "/admissions/mtech_admissions.docx",
    )

    assert score_broad_candidate(
        query,
        index_doc,
    ) > score_broad_candidate(
        query,
        specific_doc,
    )


def test_general_information_filename_does_not_create_index_bonus():

    query = (
        "What academic programs are available?"
    )

    hostel = make_doc(
        (
            "Hostel accommodation and room facilities."
        ),
        "/hostel_accommodation/general_information.docx",
    )

    assert score_broad_candidate(
        query,
        hostel,
    ) < 1.0


def test_unrelated_research_hostel_document_is_not_positive():

    query = (
        "What research opportunities are available?"
    )

    hostel = make_doc(
        (
            "Hostel accommodation facilities and room amenities."
        ),
        "/hostel_accommodation/general_information.docx",
    )

    assert score_broad_candidate(
        query,
        hostel,
    ) < 1.0


# =========================================================
# Candidate Assembly
# =========================================================

def test_program_selection_excludes_unrelated_hostel():

    query = (
        "What academic programs are available?"
    )

    documents = [
        make_doc(
            (
                "Academic programs overview "
                "B.Tech M.Tech M.Sc Ph.D."
            ),
            "/programs/programs.docx",
        ),
        make_doc(
            (
                "M.Tech eligibility and admission."
            ),
            "/admissions/mtech_admissions.docx",
        ),
        make_doc(
            (
                "M.Sc programs and admission."
            ),
            "/programs/msc_programs.docx",
        ),
        make_doc(
            (
                "Hostel accommodation details."
            ),
            "/hostel_accommodation/general_information.docx",
        ),
    ]

    selected = assemble_broad_candidates(
        query,
        documents,
        max_candidates=3,
    )

    sources = {
        document.metadata["source"]
        for document in selected
    }

    assert (
        "/programs/programs.docx"
        in sources
    )

    assert (
        "/hostel_accommodation/general_information.docx"
        not in sources
    )


def test_research_selection_preserves_multiple_relevant_sources():

    query = (
        "What research opportunities are available?"
    )

    electrical = make_doc(
        (
            "Research areas and research themes."
        ),
        "/departments/electrical/research.docx",
    )

    aide = make_doc(
        (
            "Research opportunities and laboratories."
        ),
        "/schools/aide/research.docx",
    )

    hostel = make_doc(
        (
            "Hostel accommodation facilities."
        ),
        "/hostel_accommodation/general_information.docx",
    )

    selected = assemble_broad_candidates(
        query,
        [
            electrical,
            aide,
            hostel,
        ],
        max_candidates=3,
    )

    selected_sources = {
        document.metadata["source"]
        for document in selected
    }

    assert (
        electrical.metadata["source"]
        in selected_sources
    )

    assert (
        aide.metadata["source"]
        in selected_sources
    )

    assert (
        hostel.metadata["source"]
        not in selected_sources
    )


def test_unrelated_documents_cannot_fill_empty_slots():

    query = (
        "What academic programs are available?"
    )

    program_doc = make_doc(
        (
            "Academic programs include B.Tech and M.Tech."
        ),
        "/programs/programs.docx",
    )

    hostel_doc = make_doc(
        "Hostel accommodation details.",
        "/hostel_accommodation/a.docx",
    )

    placement_doc = make_doc(
        "Placement statistics.",
        "/placements/b.docx",
    )

    selected = assemble_broad_candidates(
        query,
        [
            program_doc,
            hostel_doc,
            placement_doc,
        ],
        max_candidates=5,
    )

    sources = {
        document.metadata["source"]
        for document in selected
    }

    assert sources == {
        "/programs/programs.docx"
    }


def test_facilities_can_include_hostel_when_query_is_facilities():

    query = (
        "What facilities are available?"
    )

    hostel = make_doc(
        (
            "Hostel facilities include furnished rooms, "
            "laundry, Wi-Fi, and common amenities."
        ),
        "/hostel_accommodation/general_information.docx",
    )

    selected = assemble_broad_candidates(
        query,
        [
            hostel,
        ],
        max_candidates=3,
    )

    assert (
        hostel in selected
    )


def test_non_broad_query_preserves_input_order():

    query = (
        "What is the M.Tech eligibility?"
    )

    documents = [
        make_doc(
            "first",
            "a.docx",
        ),
        make_doc(
            "second",
            "b.docx",
        ),
    ]

    result = assemble_broad_candidates(
        query,
        documents,
    )

    assert [
        document.page_content
        for document in result
    ] == [
        "first",
        "second",
    ]