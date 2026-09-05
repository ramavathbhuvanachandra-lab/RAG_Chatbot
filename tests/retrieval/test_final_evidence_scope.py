"""
Phase 4 — Final Evidence Scope Filter Tests
"""

from langchain_core.documents import Document

from backend.final_evidence_scope import (
    filter_final_evidence_scope,
    filter_final_evidence_scope_with_stats,
)


def doc(
    source: str,
    content: str,
):
    return Document(
        page_content=content,
        metadata={
            "source": source,
        },
    )


QUERY = (
    "Can someone with a four-year bachelor's degree "
    "apply for a regular Ph.D.?"
)


def test_direct_admission_evidence_is_kept():

    documents = [
        doc(
            "admission.docx",
            (
                "The applicant must have a bachelor's degree "
                "of minimum four-year duration with at least "
                "70% marks or 7.0/10 CPI or CGPA."
            ),
        ),
    ]

    filtered = filter_final_evidence_scope(
        query=QUERY,
        documents=documents,
    )

    assert len(filtered) == 1
    assert filtered[0].metadata["source"] == "admission.docx"


def test_financial_assistance_evidence_is_removed():

    documents = [
        doc(
            "financial.docx",
            (
                "A candidate having a B.Tech / B.S. "
                "(four-year program) with GATE/NET(JRF)/NBHM "
                "can have financial assistance."
            ),
        ),
    ]

    filtered = filter_final_evidence_scope(
        query=QUERY,
        documents=documents,
    )

    assert filtered == []


def test_school_specific_evidence_is_removed():

    documents = [
        doc(
            "saide.docx",
            (
                "Applicants to the School of AI and Data Science "
                "must have a master's degree with at least 60% "
                "marks or 6.0/10 CGPA."
            ),
        ),
    ]

    filtered = filter_final_evidence_scope(
        query=QUERY,
        documents=documents,
    )

    assert filtered == []


def test_part_time_evidence_is_removed():

    documents = [
        doc(
            "part_time.docx",
            (
                "For sponsored, external, or part-time Ph.D. "
                "admission, the applicant must have two years "
                "of work experience."
            ),
        ),
    ]

    filtered = filter_final_evidence_scope(
        query=QUERY,
        documents=documents,
    )

    assert filtered == []


def test_mixed_evidence_keeps_only_compatible_documents():

    documents = [
        doc(
            "financial.docx",
            (
                "A B.Tech / B.S. holder with GATE may receive "
                "financial assistance."
            ),
        ),
        doc(
            "admission.docx",
            (
                "The applicant must have a bachelor's degree "
                "of minimum four-year duration with at least "
                "70% marks."
            ),
        ),
        doc(
            "saide.docx",
            (
                "Applicants to the School of AI and Data Science "
                "must have a master's degree with 60% marks."
            ),
        ),
    ]

    filtered = filter_final_evidence_scope(
        query=QUERY,
        documents=documents,
    )

    assert len(filtered) == 1
    assert (
        filtered[0].metadata["source"]
        == "admission.docx"
    )


def test_order_is_preserved():

    documents = [
        doc(
            "admission_1.docx",
            "The applicant must have a four-year bachelor's degree.",
        ),
        doc(
            "admission_2.docx",
            "At least 70% marks are required.",
        ),
    ]

    filtered = filter_final_evidence_scope(
        query=QUERY,
        documents=documents,
    )

    assert [
        document.metadata["source"]
        for document in filtered
    ] == [
        "admission_1.docx",
        "admission_2.docx",
    ]


def test_stats_are_correct():

    documents = [
        doc(
            "admission.docx",
            "The applicant must have a four-year bachelor's degree.",
        ),
        doc(
            "financial.docx",
            "A B.Tech holder may receive financial assistance.",
        ),
    ]

    result = filter_final_evidence_scope_with_stats(
        query=QUERY,
        documents=documents,
    )

    assert result["input_count"] == 2
    assert result["kept_count"] == 1
    assert result["removed_count"] == 1
    assert len(result["documents"]) == 1
