"""
Phase 4 — Claim-Level Context Filter Tests
"""

from langchain_core.documents import Document

from backend.claim_context_filter import (
    split_evidence_units,
    filter_document_claim_units,
    filter_claim_context,
    filter_claim_context_with_stats,
)


QUERY = (
    "Can someone with a four-year bachelor's degree "
    "apply for regular Ph.D. admission?"
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


def test_split_evidence_units():

    units = split_evidence_units(
        (
            "The applicant must have a four-year bachelor's "
            "degree. "
            "Financial assistance is provided according to "
            "applicable norms."
        )
    )

    assert len(units) >= 2


def test_keep_direct_admission_unit():

    document = doc(
        "admission.docx",
        (
            "The applicant must have a bachelor's degree "
            "of minimum four-year duration with at least "
            "70% marks."
        ),
    )

    result = filter_document_claim_units(
        query=QUERY,
        document=document,
    )

    assert result is not None
    assert "four-year" in result.page_content.lower()


def test_remove_financial_assistance_unit():

    document = doc(
        "financial.docx",
        (
            "A candidate having a B.Tech / B.S. "
            "(four-year program) with GATE/NET(JRF)/NBHM "
            "can have financial assistance."
        ),
    )

    result = filter_document_claim_units(
        query=QUERY,
        document=document,
    )

    assert result is None


def test_remove_school_specific_unit():

    document = doc(
        "saide.docx",
        (
            "Applicants to the School of AI and Data Science "
            "must have a master's degree with at least "
            "60% marks."
        ),
    )

    result = filter_document_claim_units(
        query=QUERY,
        document=document,
    )

    assert result is None


def test_mixed_chunk_is_cleaned():

    document = doc(
        "mixed.docx",
        (
            "The applicant must have a bachelor's degree "
            "of minimum four-year duration with at least "
            "70% marks. "
            "A B.Tech / B.S. holder may receive financial "
            "assistance under applicable norms. "
            "For sponsored or part-time admission, two years "
            "of work experience may be required."
        ),
    )

    result = filter_document_claim_units(
        query=QUERY,
        document=document,
    )

    assert result is not None

    text = result.page_content.lower()

    assert "four-year" in text
    assert "financial assistance" not in text
    assert "sponsored" not in text
    assert "part-time" not in text


def test_metadata_is_preserved():

    document = doc(
        "admission.docx",
        (
            "The applicant must have a four-year "
            "bachelor's degree."
        ),
    )

    result = filter_document_claim_units(
        query=QUERY,
        document=document,
    )

    assert result is not None
    assert (
        result.metadata["source"]
        == "admission.docx"
    )


def test_collection_filter_preserves_order():

    documents = [
        doc(
            "first.docx",
            (
                "The applicant must have a four-year "
                "bachelor's degree."
            ),
        ),
        doc(
            "second.docx",
            (
                "A B.Tech/B.S. holder may receive "
                "financial assistance."
            ),
        ),
        doc(
            "third.docx",
            (
                "At least 70% marks are required for "
                "the bachelor's route."
            ),
        ),
    ]

    result = filter_claim_context(
        query=QUERY,
        documents=documents,
    )

    assert [
        document.metadata["source"]
        for document in result
    ] == [
        "first.docx",
        "third.docx",
    ]


def test_stats():

    documents = [
        doc(
            "admission.docx",
            (
                "The applicant must have a four-year "
                "bachelor's degree."
            ),
        ),
        doc(
            "financial.docx",
            (
                "A B.Tech/B.S. holder may receive "
                "financial assistance."
            ),
        ),
    ]

    result = filter_claim_context_with_stats(
        query=QUERY,
        documents=documents,
    )

    assert result["input_documents"] == 2
    assert result["output_documents"] == 1
    assert result["removed_documents"] == 1
