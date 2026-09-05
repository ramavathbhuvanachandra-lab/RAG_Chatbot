"""
Phase 4 — Broad-List Completeness Tests

These tests verify that list-style questions distinguish:
    complete-enough evidence
    partial evidence
    insufficient evidence

The tests are intentionally corpus-agnostic.
"""

from langchain_core.documents import Document

from backend.evidence_coverage import (
    assess_evidence_coverage,
)


def doc(text: str):
    return Document(
        page_content=text
    )


# =========================================================
# Strong list evidence
# =========================================================

def test_broad_list_with_multiple_coherent_documents_can_be_supported():

    result = assess_evidence_coverage(
        query="What programs are offered at the institute?",
        documents=[
            doc(
                "The institute offers undergraduate B.Tech programs "
                "in Electrical Engineering, Computer Science, and "
                "Mechanical Engineering."
            ),
            doc(
                "The institute offers postgraduate M.Tech programs "
                "and Ph.D. programs across several academic areas."
            ),
            doc(
                "Academic programs include B.Tech, M.Tech, M.Sc., "
                "and Ph.D. programs."
            ),
        ],
    )

    assert result["question_type"] == "list"
    assert result["status"] == "supported"


# =========================================================
# Partial list evidence
# =========================================================

def test_single_relevant_document_is_not_automatically_complete():

    result = assess_evidence_coverage(
        query="What programs are offered at the institute?",
        documents=[
            doc(
                "The institute offers B.Tech in Electrical Engineering "
                "and B.Tech in Computer Science."
            ),
        ],
    )

    assert result["question_type"] == "list"
    assert result["status"] == "partially_supported"


def test_partial_documents_can_support_partial_list():

    result = assess_evidence_coverage(
        query="What departments are available?",
        documents=[
            doc(
                "The Department of Electrical Engineering offers "
                "teaching and research programs."
            ),
            doc(
                "The Department of Computer Science conducts "
                "teaching and research."
            ),
        ],
    )

    assert result["question_type"] == "list"
    assert result["status"] == "partially_supported"


# =========================================================
# Insufficient
# =========================================================

def test_unrelated_documents_are_insufficient_for_list():

    result = assess_evidence_coverage(
        query="What programs are offered at the institute?",
        documents=[
            doc(
                "The hostel provides Wi-Fi, dining, gyms, and study rooms."
            ),
            doc(
                "The institute has a health centre and security services."
            ),
        ],
    )

    assert result["question_type"] == "list"
    assert result["status"] == "insufficient"


# =========================================================
# Broadness should not be inferred from character count alone
# =========================================================

def test_long_unrelated_document_does_not_create_list_support():

    result = assess_evidence_coverage(
        query="What programs are offered at the institute?",
        documents=[
            doc(
                "The institute has extensive campus infrastructure. "
                + ("Facilities and services are available throughout campus. " * 30)
            ),
        ],
    )

    assert result["status"] == "insufficient"


# =========================================================
# One huge document containing a coherent inventory
# =========================================================

def test_single_coherent_inventory_document_can_be_supported():

    result = assess_evidence_coverage(
        query="What departments are available?",
        documents=[
            doc(
                "Academic Departments: "
                "Electrical Engineering; "
                "Computer Science and Engineering; "
                "Mechanical Engineering; "
                "Mathematics; "
                "Physics; "
                "Chemistry; "
                "Management."
            ),
        ],
    )

    assert result["question_type"] == "list"
    assert result["status"] in {
        "supported",
        "partially_supported",
    }


# =========================================================
# List question with clearly narrow evidence
# =========================================================

def test_narrow_research_evidence_is_not_complete_institute_list():

    result = assess_evidence_coverage(
        query="What research areas are available at the institute?",
        documents=[
            doc(
                "Electrical Engineering research includes "
                "control systems and power engineering."
            ),
        ],
    )

    assert result["status"] == "partially_supported"


# =========================================================
# Program-specific list
# =========================================================

def test_program_scoped_list_can_be_supported_by_coherent_evidence():

    result = assess_evidence_coverage(
        query=(
            "What research areas are available in "
            "Electrical Engineering?"
        ),
        documents=[
            doc(
                "Electrical Engineering research areas include "
                "control systems, power systems, communication systems, "
                "signal processing, and VLSI."
            ),
            doc(
                "Electrical Engineering faculty work in control, "
                "power, communications, RF, and embedded systems."
            ),
        ],
    )

    assert result["question_type"] == "list"
    assert result["status"] == "supported"


# =========================================================
# Duplicate evidence should not fake breadth
# =========================================================

def test_duplicate_documents_do_not_create_completeness():

    text = (
        "B.Tech is offered in Electrical Engineering "
        "and Computer Science."
    )

    result = assess_evidence_coverage(
        query="What programs are offered?",
        documents=[
            doc(text),
            doc(text),
            doc(text),
        ],
    )

    assert result["status"] != "supported"


# =========================================================
# Mixed relevant and irrelevant evidence
# =========================================================

def test_irrelevant_documents_do_not_count_toward_list_breadth():

    result = assess_evidence_coverage(
        query="What programs are offered?",
        documents=[
            doc(
                "B.Tech programs are offered in Electrical Engineering "
                "and Computer Science."
            ),
            doc(
                "Hostel rooms have Wi-Fi and dining facilities."
            ),
            doc(
                "The health centre provides medical services."
            ),
        ],
    )

    assert result["status"] == "partially_supported"


# =========================================================
# No documents
# =========================================================

def test_empty_documents_are_insufficient():

    result = assess_evidence_coverage(
        query="What programs are offered?",
        documents=[],
    )

    assert result["status"] == "insufficient"
