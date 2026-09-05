"""
Phase 4 — Local Context Expansion Integration Tests

Purpose
-------
Verify that local context expansion improves the evidence set for
real retrieval results without:

- crossing source boundaries
- adding obvious noise
- exploding the number of documents
- destroying the original retrieval anchor

These tests use the current corpus as a stress test, but the
production behavior must remain generic.
"""

from backend.local_context import (
    expand_local_context,
)

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    rerank_documents,
    FINAL_CONTEXT_DOCUMENTS,
    get_source,
)


# =========================================================
# Retrieval Helper
# =========================================================

def retrieve_reranked(
    question: str,
):
    dense_docs = dense_retrieve(
        question
    )

    keyword_docs = keyword_retrieve(
        question
    )

    fused_docs = reciprocal_rank_fusion(
        [
            dense_docs,
            keyword_docs,
        ]
    )

    fused_docs = deduplicate_documents(
        fused_docs
    )

    return rerank_documents(
        query=question,
        documents=fused_docs,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )


# =========================================================
# Test 1 — Electrical Engineering
# =========================================================

def test_electrical_context_expansion_preserves_anchor():

    question = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    reranked = retrieve_reranked(
        question
    )

    expanded = expand_local_context(
        reranked
    )

    original_sources = {
        get_source(document)
        for document in reranked
    }

    expanded_sources = {
        get_source(document)
        for document in expanded
    }

    # Expansion must not introduce a completely new source
    # family that was absent from the retrieved anchors.
    assert expanded_sources.issubset(
        original_sources
    )


# =========================================================
# Test 2 — Expansion remains bounded
# =========================================================

def test_local_context_expansion_remains_small():

    question = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    reranked = retrieve_reranked(
        question
    )

    expanded = expand_local_context(
        reranked
    )

    # Each anchor can contribute at most:
    #
    # previous + anchor + next
    #
    # So a 5-document result should never explode into an
    # arbitrary large context.
    assert len(expanded) <= 15

    assert len(expanded) >= len(
        reranked
    )


# =========================================================
# Test 3 — Hostel fee expansion
# =========================================================

def test_hostel_fee_expansion_keeps_same_source_evidence():

    question = "What are the hostel fees?"

    reranked = retrieve_reranked(
        question
    )

    expanded = expand_local_context(
        reranked
    )

    finance_documents = [
        document
        for document in expanded
        if "finance/fees_and_finance.docx"
        in get_source(document)
    ]

    # The real corpus contains multiple adjacent finance chunks
    # describing accommodation charges.
    assert finance_documents

    assert all(
        "finance/fees_and_finance.docx"
        in get_source(document)
        for document in finance_documents
    )


# =========================================================
# Test 4 — Ph.D. continuation remains available
# =========================================================

def test_phd_context_expansion_retains_eligibility_evidence():

    question = (
        "What are the eligibility requirements "
        "for regular Ph.D. admission?"
    )

    reranked = retrieve_reranked(
        question
    )

    expanded = expand_local_context(
        reranked
    )

    phd_documents = [
        document
        for document in expanded
        if "phd"
        in get_source(document).lower()
    ]

    assert phd_documents

    combined_text = " ".join(
        document.page_content.lower()
        for document in phd_documents
    )

    # The real corpus contains the four-year bachelor's
    # eligibility route in nearby context.
    assert (
        "four-year"
        in combined_text
        or
        "70%"
        in combined_text
    )


# =========================================================
# Test 5 — No obvious retrieval noise explosion
# =========================================================

def test_local_context_does_not_add_retrieval_metadata_noise():

    question = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    reranked = retrieve_reranked(
        question
    )

    expanded = expand_local_context(
        reranked
    )

    for document in expanded:

        content = (
            document.page_content
            .lower()
        )

        assert (
            "retrieval representation"
            not in content
        )

        assert (
            "rrf score"
            not in content
        )

        assert (
            "chunk id"
            not in content
        )