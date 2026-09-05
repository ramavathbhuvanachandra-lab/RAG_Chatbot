"""
Local Context Pipeline Tests

Validate that local context expansion happens only after an initial
relevance pass and that each anchor remains a single evidence unit.
"""

from backend.local_context import expand_local_context

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    rerank_documents,
    FINAL_CONTEXT_DOCUMENTS,
)


def initial_rerank(question):
    dense = dense_retrieve(
        question
    )

    bm25 = keyword_retrieve(
        question
    )

    fused = reciprocal_rank_fusion(
        [dense, bm25]
    )

    fused = deduplicate_documents(
        fused
    )

    return rerank_documents(
        query=question,
        documents=fused,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )


def test_local_context_is_applied_only_to_initial_top_documents():

    question = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    initial_docs = initial_rerank(
        question
    )

    expanded = expand_local_context(
        initial_docs
    )

    assert len(initial_docs) <= FINAL_CONTEXT_DOCUMENTS

    assert (
        len(expanded)
        <= FINAL_CONTEXT_DOCUMENTS * 3
    )


def test_initial_rerank_is_small_before_expansion():

    question = (
        "What are the eligibility requirements "
        "for regular Ph.D. admission?"
    )

    initial_docs = initial_rerank(
        question
    )

    assert len(initial_docs) <= 5