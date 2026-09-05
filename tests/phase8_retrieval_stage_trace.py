"""
Phase 8 — Retrieval Stage Trace

Diagnose one retrieval failure through each retrieval stage.

No production behavior is modified.
"""

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    rerank_documents,
)


QUESTION = (
    "What research areas are related to robotics at IIT Jodhpur?"
)


def show_documents(
    label,
    documents,
):
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)

    print(
        f"COUNT: {len(documents)}"
    )

    for index, document in enumerate(
        documents,
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

        print(
            f"\n[{index}] SOURCE: {source}"
        )

        print(
            "TEXT:"
        )

        print(
            document.page_content[:1200]
        )


def main():

    print("=" * 100)
    print("PHASE 8 — ROBOTICS RETRIEVAL STAGE TRACE")
    print("=" * 100)

    print(
        f"\nQUERY:\n{QUESTION}"
    )

    # -----------------------------------------------------
    # Dense retrieval
    # -----------------------------------------------------

    dense_docs = dense_retrieve(
        QUESTION
    )

    show_documents(
        "DENSE TOP RESULTS",
        dense_docs,
    )

    # -----------------------------------------------------
    # BM25 retrieval
    # -----------------------------------------------------

    bm25_docs = keyword_retrieve(
        QUESTION
    )

    show_documents(
        "BM25 TOP RESULTS",
        bm25_docs,
    )

    # -----------------------------------------------------
    # Weighted RRF
    # -----------------------------------------------------

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

    show_documents(
        "RRF RESULTS",
        fused_docs,
    )

    # -----------------------------------------------------
    # Deduplication
    # -----------------------------------------------------

    deduped_docs = deduplicate_documents(
        fused_docs
    )

    show_documents(
        "AFTER DEDUPLICATION",
        deduped_docs,
    )

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------

    reranked_docs = rerank_documents(
        QUESTION,
        deduped_docs,
    )

    show_documents(
        "AFTER RERANKING",
        reranked_docs,
    )


if __name__ == "__main__":
    main()