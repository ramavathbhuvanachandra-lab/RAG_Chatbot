"""
Phase 4 — Evidence Scope Tests

Purpose
-------
Ensure that evidence relevant to a broad topic is not automatically
treated as evidence for the specific claim being asked.

Example:

    Admission eligibility
        !=
    Financial assistance eligibility
"""

from langchain_core.documents import Document

from backend.evidence import (
    assess_evidence_sufficiency,
)


def test_phd_admission_question_prefers_direct_admission_evidence():

    documents = [
        Document(
            page_content=(
                "A candidate having a B.Tech / B.S. "
                "(four-year program) with GATE/NET(JRF)/NBHM "
                "can have financial assistance as per MHRD norms."
            )
        ),
        Document(
            page_content=(
                "The applicant must have a bachelor’s degree "
                "of minimum four-year duration in engineering "
                "or science or medicine or pharmacy or agricultural "
                "science or veterinary science or equivalent with "
                "at least 70% marks or at least 7.0/10 CPI or CGPA "
                "for GEN/OBC."
            )
        ),
    ]

    result = assess_evidence_sufficiency(
        query=(
            "Can someone with a four-year bachelor's degree "
            "apply for a regular Ph.D.?"
        ),
        documents=documents,
    )

    assert result["status"] == "supported"

    # The direct admission document must provide the strongest
    # support for the admission claim.
    assert result["score"] >= 0.55


def test_financial_assistance_evidence_alone_is_not_admission_evidence():

    documents = [
        Document(
            page_content=(
                "A candidate having a B.Tech / B.S. "
                "(four-year program) with GATE/NET(JRF)/NBHM "
                "can have financial assistance as per MHRD norms."
            )
        ),
    ]

    result = assess_evidence_sufficiency(
        query=(
            "Can someone with a four-year bachelor's degree "
            "apply for a regular Ph.D.?"
        ),
        documents=documents,
    )

    assert result["status"] == "insufficient"


def test_school_specific_phd_eligibility_is_not_automatically_institute_wide():

    documents = [
        Document(
            page_content=(
                "Applicants to the School of AI and Data Science "
                "must have a master’s degree in engineering, science, "
                "humanities, social sciences, or equivalent with at "
                "least 60% marks or 6.0/10 CGPA."
            )
        ),
    ]

    result = assess_evidence_sufficiency(
        query=(
            "Can someone with a four-year bachelor's degree "
            "apply for a regular Ph.D.?"
        ),
        documents=documents,
    )

    assert result["status"] == "insufficient"