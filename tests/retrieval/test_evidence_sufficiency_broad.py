"""
Evidence Sufficiency — Broad Question Tests

Purpose
-------
Ensure that broad list questions can be supported by multiple
moderately relevant documents instead of requiring one very strong
document.

The logic must remain deterministic and college-agnostic.
"""

from langchain_core.documents import Document

from backend.evidence import (
    assess_evidence_sufficiency,
)


# =========================================================
# Broad list support
# =========================================================

def test_broad_list_question_can_be_supported_by_multiple_documents():

    result = assess_evidence_sufficiency(
        query="What programs are offered at IIT Jodhpur?",
        documents=[
            Document(
                page_content=(
                    "IIT Jodhpur offers undergraduate "
                    "and postgraduate academic programs."
                )
            ),
            Document(
                page_content=(
                    "The institute offers B.Tech programs "
                    "in several engineering disciplines."
                )
            ),
            Document(
                page_content=(
                    "The institute also offers M.Tech and "
                    "Ph.D. programs."
                )
            ),
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# Specific question remains conservative
# =========================================================

def test_specific_question_still_requires_direct_support():

    result = assess_evidence_sufficiency(
        query="What is the hostel fee?",
        documents=[
            Document(
                page_content=(
                    "IIT Jodhpur provides hostel accommodation."
                )
            ),
            Document(
                page_content=(
                    "Hostels have Wi-Fi, common rooms, "
                    "study rooms, and gyms."
                )
            ),
        ],
    )

    assert result["status"] == "insufficient"


# =========================================================
# Empty evidence
# =========================================================

def test_empty_evidence_is_insufficient():

    result = assess_evidence_sufficiency(
        query="What programs are offered at IIT Jodhpur?",
        documents=[],
    )

    assert result["status"] == "insufficient"