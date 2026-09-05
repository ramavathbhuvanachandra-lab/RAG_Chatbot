"""
Phase 8A.2 — Reranker Score Trace

Diagnostic only.

Purpose:
    Show why candidate documents move up/down during reranking.
"""

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    score_document_relevance,
)


QUESTION = (
    "What research areas are related to robotics at IIT Jodhpur?"
)


def main():

    print("=" * 100)
    print("PHASE 8 — RERANKER SCORE TRACE")
    print("=" * 100)

    dense_docs = dense_retrieve(
        QUESTION
    )

    bm25_docs = keyword_retrieve(
        QUESTION
    )

    fused_docs = reciprocal_rank_fusion(
        [
            dense_docs,
            bm25_docs,
        ],
        weights=[
            0.7,
            0.3,
        ],
    )

    docs = deduplicate_documents(
        fused_docs
    )

    print(
        f"\nCANDIDATES AFTER DEDUP: {len(docs)}"
    )

    for rank, document in enumerate(
        docs,
        start=1,
    ):

        source = (
            document.metadata.get(
                "source",
                "",
            )
            if document.metadata
            else ""
        )

        score = score_document_relevance(
            QUESTION,
            document,
            rank,
        )

        print("\n" + "-" * 100)

        print(
            f"RANK: {rank}"
        )

        print(
            f"SCORE: {score}"
        )

        print(
            f"SOURCE: {source}"
        )

        print(
            "TEXT:"
        )

        print(
            document.page_content[:1200]
        )


if __name__ == "__main__":
    main()