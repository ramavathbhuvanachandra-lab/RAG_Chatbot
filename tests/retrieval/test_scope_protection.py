from langchain_core.documents import Document

from backend.retriever import (
    detect_programs,
    detect_topics,
    score_document_relevance,
)


def test_program_detection_btech():
    assert "btech" in detect_programs(
        "What are the B.Tech admission requirements?"
    )


def test_program_detection_msc():
    assert "msc" in detect_programs(
        "What are the M.Sc. admission requirements?"
    )


def test_topic_detection_admission():
    assert "admission" in detect_topics(
        "What are the eligibility criteria for admission?"
    )


def test_topic_detection_hostel():
    assert "hostel" in detect_topics(
        "What facilities are available in the hostel?"
    )


def test_matching_program_is_preferred():
    query = "What are the B.Tech admission requirements?"

    btech_doc = Document(
        page_content=(
            "B.Tech admission requirements include "
            "the required academic qualifications."
        ),
        metadata={
            "source": "btech_admission.docx",
        },
    )

    msc_doc = Document(
        page_content=(
            "M.Sc. admission requirements include "
            "the required academic qualifications."
        ),
        metadata={
            "source": "msc_admission.docx",
        },
    )

    btech_score = score_document_relevance(
        query=query,
        document=btech_doc,
        original_rank=1,
    )

    msc_score = score_document_relevance(
        query=query,
        document=msc_doc,
        original_rank=2,
    )

    assert btech_score > msc_score


def test_generic_program_content_is_not_hard_filtered():
    query = "What are the M.Sc. admission requirements?"

    generic_doc = Document(
        page_content=(
            "Postgraduate admission requires the prescribed "
            "academic qualification and eligibility criteria."
        ),
        metadata={
            "source": "postgraduate_admission.docx",
        },
    )

    score = score_document_relevance(
        query=query,
        document=generic_doc,
        original_rank=2,
    )

    assert isinstance(score, float)