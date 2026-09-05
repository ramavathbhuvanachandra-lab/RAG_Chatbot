"""
Phase 4/5 — Evidence Sufficiency Tests

Purpose
-------
Verify that the evidence layer can distinguish:

1. Strong supporting evidence.
2. No evidence.
3. Clearly unrelated evidence.

The evidence layer is intentionally separate from the retriever.
"""

from langchain_core.documents import Document

from backend.evidence import (
    assess_evidence_sufficiency,
)


# =========================================================
# Strong Evidence
# =========================================================

def test_strong_evidence_is_supported():
    """
    Evidence directly addresses the requested question.
    """

    question = (
        "What research areas are available in Electrical Engineering?"
    )

    documents = [
        Document(
            page_content=(
                "The Department of Electrical Engineering "
                "research areas include MIMO communications, "
                "control systems, signal processing, VLSI, "
                "and cyber-physical systems."
            ),
            metadata={
                "source": "electrical_research.docx",
            },
        )
    ]

    result = assess_evidence_sufficiency(
        query=question,
        documents=documents,
    )

    assert result["status"] == "supported"
    assert result["score"] > 0
    assert result["relevant_documents"] >= 1


# =========================================================
# Empty Evidence
# =========================================================

def test_empty_evidence_is_insufficient():
    """
    No retrieved evidence must never be considered sufficient.
    """

    question = "What are the hostel fees?"

    result = assess_evidence_sufficiency(
        query=question,
        documents=[],
    )

    assert result["status"] == "insufficient"
    assert result["score"] == 0.0
    assert result["relevant_documents"] == 0


# =========================================================
# Unrelated Evidence
# =========================================================

def test_unrelated_evidence_is_insufficient():
    """
    Evidence from an unrelated topic must not be treated as
    sufficient merely because it is college-related.
    """

    question = (
        "What are the eligibility requirements for regular "
        "Ph.D. admission?"
    )

    documents = [
        Document(
            page_content=(
                "The Department of Electrical Engineering "
                "offers B.Tech. and M.Tech. programs."
            ),
            metadata={
                "source": "electrical_overview.docx",
            },
        )
    ]

    result = assess_evidence_sufficiency(
        query=question,
        documents=documents,
    )

    assert result["status"] == "insufficient"
    assert result["relevant_documents"] == 0


# =========================================================
# Program-Aligned Evidence
# =========================================================

def test_program_aligned_evidence_is_supported():
    """
    Evidence matching the requested academic program and
    admission topic should receive sufficient support.
    """

    question = (
        "What are the eligibility requirements for "
        "regular Ph.D. admission?"
    )

    documents = [
        Document(
            page_content=(
                "Admission to the Ph.D. program requires "
                "a master's degree in engineering, pharmacy, "
                "agricultural science, science, humanities, "
                "social sciences, or management with the "
                "prescribed marks or CGPA."
            ),
            metadata={
                "source": "phd_admissions.docx",
            },
        )
    ]

    result = assess_evidence_sufficiency(
        query=question,
        documents=documents,
    )

    assert result["status"] == "supported"
    assert result["relevant_documents"] >= 1


# =========================================================
# Topic-Aligned Evidence
# =========================================================

def test_topic_aligned_evidence_is_supported():
    """
    Evidence matching the requested institutional topic should
    be recognized even when no academic program is specified.
    """

    question = (
        "What research areas are available in Electrical Engineering?"
    )

    documents = [
        Document(
            page_content=(
                "The Department of Electrical Engineering "
                "pursues research in power systems, control, "
                "signal processing, communications, VLSI, "
                "and cyber-physical systems."
            ),
            metadata={
                "source": "electrical_research.docx",
            },
        )
    ]

    result = assess_evidence_sufficiency(
        query=question,
        documents=documents,
    )

    assert result["status"] == "supported"
    assert result["relevant_documents"] >= 1