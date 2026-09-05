"""Phase 8B + 8C production regression tests."""

from langchain_core.documents import Document

from backend.broad_institutional_retrieval import (
    assemble_broad_candidates,
    build_broad_retrieval_queries,
    detect_broad_scope,
)
from backend.institutional_coverage import (
    assess_institutional_coverage,
)


def doc(
    content: str,
    source: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source},
    )


def test_program_scope():
    assert (
        detect_broad_scope(
            "What academic programs are available?"
        ).name
        == "programs"
    )


def test_department_scope():
    assert (
        detect_broad_scope(
            "What departments are there?"
        ).name
        == "departments"
    )


def test_broad_query_expansion_is_bounded():
    queries = build_broad_retrieval_queries(
        "What academic programs are available?"
    )
    assert 1 <= len(queries) <= 2


def test_focused_question_is_not_expanded():
    assert build_broad_retrieval_queries(
        "What is the M.Tech eligibility?"
    ) == []


def test_programs_reject_unrelated_hostel():
    program = doc(
        (
            "Academic programs include B.Tech, M.Tech, "
            "M.Sc. and Ph.D."
        ),
        "/programs/programs_overview.docx",
    )

    hostel = doc(
        "Hostel accommodation and room facilities.",
        "/hostel_accommodation/general_information.docx",
    )

    selected = assemble_broad_candidates(
        "What academic programs are available?",
        [program, hostel],
    )

    assert program in selected
    assert hostel not in selected


def test_research_preserves_complementary_sources():
    first = doc(
        "Research areas and research themes.",
        "/departments/electrical/research.docx",
    )

    second = doc(
        "Research groups and research opportunities.",
        "/schools/aide/research.docx",
    )

    unrelated = doc(
        "Hostel accommodation facilities.",
        "/hostel_accommodation/general_information.docx",
    )

    selected = assemble_broad_candidates(
        "What research opportunities are available?",
        [first, second, unrelated],
    )

    assert first in selected
    assert second in selected
    assert unrelated not in selected


def test_program_coverage_without_overview_is_partial():
    result = assess_institutional_coverage(
        "What academic programs are available?",
        [
            doc(
                "M.Tech program details.",
                "/admissions/mtech.docx",
            ),
            doc(
                "M.Sc. program details.",
                "/admissions/msc.docx",
            ),
        ],
    )

    assert result["status"] == "partially_supported"


def test_program_coverage_with_overview_is_supported():
    result = assess_institutional_coverage(
        "What academic programs are available?",
        [
            doc(
                (
                    "Academic programs overview. "
                    "B.Tech M.Tech M.Sc. Ph.D. MBA."
                ),
                "/programs/programs_overview.docx",
            ),
        ],
    )

    assert result["status"] == "supported"
    assert result["question_type"] == "list"


def test_department_coverage_requires_real_breadth():
    result = assess_institutional_coverage(
        "What departments are there?",
        [
            doc(
                (
                    "Academic departments and schools. "
                    "Chemistry department. CSE department."
                ),
                "/departments/departments_overview.docx",
            ),
        ],
    )

    assert result["status"] == "supported"


def test_irrelevant_documents_do_not_create_coverage():
    result = assess_institutional_coverage(
        "What academic programs are available?",
        [
            doc(
                "Hostel facilities and room information.",
                "/hostel_accommodation/general_information.docx",
            ),
        ],
    )

    assert result["status"] == "insufficient"