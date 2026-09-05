"""
Source-scope reranking tests.

These tests verify that broad program-list questions prefer program
sources without hardcoding institution-specific program names.
"""

from langchain_core.documents import Document

from backend.retriever import (
    detect_query_scope,
    score_source_scope,
    score_document_relevance,
)


def test_program_query_scope_is_detected():
    assert detect_query_scope(
        "What programs are offered at IIT Jodhpur?"
    ) == "programs"


def test_non_program_question_has_no_program_scope_bonus():
    assert score_source_scope(
        "What are the hostel fees?",
        "/corpus/programs/mtech/programs.docx",
    ) == 0.0


def test_dedicated_program_source_beats_unrelated_source_scope():
    query = "What programs are offered at IIT Jodhpur?"

    program_score = score_source_scope(
        query,
        "/corpus/departments/example/programs.docx",
    )

    unrelated_score = score_source_scope(
        query,
        "/corpus/hostel_accommodation/general_information.docx",
    )

    assert program_score > unrelated_score


def test_program_scope_is_soft_not_absolute():
    query = "What programs are offered at IIT Jodhpur?"

    program_document = Document(
        page_content=(
            "Bachelor of Technology programs in engineering."
        ),
        metadata={
            "source": "/corpus/departments/example/programs.docx"
        },
    )

    general_document = Document(
        page_content=(
            "IIT Jodhpur offers academic programs across several areas."
        ),
        metadata={
            "source": "/corpus/institute_overview/overview.docx"
        },
    )

    program_score = score_document_relevance(
        query=query,
        document=program_document,
        original_rank=10,
    )

    general_score = score_document_relevance(
        query=query,
        document=general_document,
        original_rank=1,
    )

    # A strong early retrieval result is allowed to remain competitive.
    assert general_score > 0
    assert program_score > 0
