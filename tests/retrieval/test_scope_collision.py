"""
Phase 3 — Scope / Entity Collision Regression

Purpose
-------
Verify that program/topic/entity signals help ranking when the
corpus contains semantically related but potentially conflicting
documents.

Important:
- These tests do not hardcode folder paths.
- These tests do not require exact top-1 results.
- Scope/entity signals are expected to be soft preferences.
"""

from langchain_core.documents import Document

from backend.retriever import (
    score_document_relevance,
    rerank_documents,
)


# =========================================================
# Program Scope
# =========================================================

def test_msc_document_beats_mtech_document_for_msc_query():
    query = (
        "What are the M.Sc. eligibility requirements?"
    )

    msc_doc = Document(
        page_content=(
            "M.Sc. admission eligibility requires the "
            "prescribed academic qualification."
        ),
        metadata={
            "source": "msc_admissions.docx",
        },
    )

    mtech_doc = Document(
        page_content=(
            "M.Tech admission eligibility requires a "
            "valid GATE score or equivalent qualification."
        ),
        metadata={
            "source": "mtech_admissions.docx",
        },
    )

    msc_score = score_document_relevance(
        query=query,
        document=msc_doc,
        original_rank=2,
    )

    mtech_score = score_document_relevance(
        query=query,
        document=mtech_doc,
        original_rank=1,
    )

    assert msc_score > mtech_score


# =========================================================
# Department Entity
# =========================================================

def test_electrical_engineering_beats_electronics_for_electrical_query():
    query = (
        "What research areas are available in Electrical Engineering?"
    )

    electrical_doc = Document(
        page_content=(
            "The Department of Electrical Engineering "
            "works on signal processing, embedded systems, "
            "VLSI, communications, and sustainable energy."
        ),
        metadata={
            "source": "electrical_engineering_research.docx",
        },
    )

    electronics_doc = Document(
        page_content=(
            "The Department of Electronics Engineering "
            "works on semiconductor devices, circuits, "
            "microelectronics, and communication systems."
        ),
        metadata={
            "source": "electronics_engineering_research.docx",
        },
    )

    electrical_score = score_document_relevance(
        query=query,
        document=electrical_doc,
        original_rank=2,
    )

    electronics_score = score_document_relevance(
        query=query,
        document=electronics_doc,
        original_rank=1,
    )

    assert electrical_score > electronics_score


# =========================================================
# Hostel vs Generic Finance
# =========================================================

def test_hostel_entity_beats_unrelated_program_fee_document():
    query = "What are the hostel fees?"

    hostel_doc = Document(
        page_content=(
            "Hostel accommodation fees are charged as part "
            "of the student residential facilities."
        ),
        metadata={
            "source": "hostel_accommodation.docx",
        },
    )

    program_fee_doc = Document(
        page_content=(
            "The academic program tuition fee is charged "
            "per semester together with the applicable fees."
        ),
        metadata={
            "source": "management_program_fees.docx",
        },
    )

    hostel_score = score_document_relevance(
        query=query,
        document=hostel_doc,
        original_rank=2,
    )

    program_fee_score = score_document_relevance(
        query=query,
        document=program_fee_doc,
        original_rank=1,
    )

    assert hostel_score > program_fee_score


# =========================================================
# Generic Evidence Must Remain Possible
# =========================================================

def test_generic_postgraduate_evidence_remains_eligible():
    query = (
        "What are the M.Sc. eligibility requirements?"
    )

    generic_doc = Document(
        page_content=(
            "Postgraduate admission requires the prescribed "
            "academic qualification and eligibility criteria."
        ),
        metadata={
            "source": "general_admissions.docx",
        },
    )

    score = score_document_relevance(
        query=query,
        document=generic_doc,
        original_rank=3,
    )

    assert isinstance(score, float)


# =========================================================
# Reranking Contract
# =========================================================

def test_reranking_preserves_relevant_scope_candidates():
    query = (
        "What are the M.Sc. admission requirements?"
    )

    documents = [
        Document(
            page_content=(
                "M.Tech admission requirements include "
                "a valid GATE score."
            ),
            metadata={
                "source": "mtech_admissions.docx",
            },
        ),
        Document(
            page_content=(
                "M.Sc. admission requirements include "
                "the prescribed academic qualification."
            ),
            metadata={
                "source": "msc_admissions.docx",
            },
        ),
        Document(
            page_content=(
                "Postgraduate admission requires the "
                "prescribed academic qualification."
            ),
            metadata={
                "source": "general_admissions.docx",
            },
        ),
    ]

    ranked = rerank_documents(
        query=query,
        documents=documents,
        top_k=3,
    )

    sources = [
        document.metadata["source"]
        for document in ranked
    ]

    assert (
        sources.index("msc_admissions.docx")
        <
        sources.index("mtech_admissions.docx")
    )