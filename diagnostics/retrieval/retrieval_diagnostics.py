"""
IIT Jodhpur Chatbot V1 - Retrieval Diagnostics

Pipeline inspected:

    Question
        ↓
    Rewrite
        ↓
    Multi-query
        ↓
    Dense + BM25
        ↓
    RRF
        ↓
    Intent-aware reranking
        ↓
    Final evidence

This script does NOT modify the production pipeline.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.nodes import (
    rewrite_query,
    generate_multi_query,
)

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    rerank_documents,
    get_document_id,
    RRF_K,
    FINAL_CONTEXT_DOCUMENTS,
)

from backend.state import GraphState


DEFAULT_QUERY = (
    "What are the eligibility requirements for B.Tech admission "
    "at IIT Jodhpur?"
)

TOP_RRF_RESULTS = 15


def source_name(document):
    source = document.metadata.get("source", "")
    source = str(source).replace("\\", "/")

    marker = "iitj_rag_v1_docs_production/"

    if marker in source:
        return source.split(marker, 1)[1]

    return source


def preview(document, length=500):
    content = document.page_content.strip()

    if len(content) > length:
        return content[:length] + "..."

    return content


def main():

    query = (
        " ".join(sys.argv[1:]).strip()
        if len(sys.argv) > 1
        else DEFAULT_QUERY
    )

    print("\n" + "=" * 100)
    print("IITJ V1 — RETRIEVAL + RERANKING DIAGNOSTICS")
    print("=" * 100)

    print("\nOriginal question:")
    print(query)

    # ========================================================
    # STEP 1 — Rewrite
    # ========================================================

    state: GraphState = {
        "question": query,
        "chat_history": [],
    }

    print("\n" + "-" * 100)
    print("STEP 1 — QUERY REWRITE")
    print("-" * 100)

    start = time.perf_counter()

    rewritten_state = rewrite_query(state)
    state.update(rewritten_state)

    rewrite_time = time.perf_counter() - start

    print(f"Time: {rewrite_time:.2f}s")
    print(f"Rewritten: {state['rewritten_question']}")

    # ========================================================
    # STEP 2 — Multi-query
    # ========================================================

    print("\n" + "-" * 100)
    print("STEP 2 — GENERATED QUERIES")
    print("-" * 100)

    start = time.perf_counter()

    generated_state = generate_multi_query(state)
    state.update(generated_state)

    multi_query_time = time.perf_counter() - start

    generated_queries = state["generated_queries"]

    print(f"Time: {multi_query_time:.2f}s")
    print(f"Generated queries: {len(generated_queries)}")

    for index, generated_query in enumerate(
        generated_queries,
        start=1,
    ):
        print(f"{index}. {generated_query}")

    # ========================================================
    # STEP 3 — Dense + BM25
    # ========================================================

    print("\n" + "-" * 100)
    print("STEP 3 — RETRIEVAL")
    print("-" * 100)

    retrieval_results = []

    for index, generated_query in enumerate(
        generated_queries,
        start=1,
    ):

        print(f"\nQuery {index}: {generated_query}")

        start = time.perf_counter()
        dense_docs = dense_retrieve(generated_query)
        dense_time = time.perf_counter() - start

        start = time.perf_counter()
        bm25_docs = keyword_retrieve(generated_query)
        bm25_time = time.perf_counter() - start

        retrieval_results.append(dense_docs)
        retrieval_results.append(bm25_docs)

        print(
            f"  Dense: {len(dense_docs)} docs "
            f"({dense_time:.2f}s)"
        )

        print(
            f"  BM25:  {len(bm25_docs)} docs "
            f"({bm25_time:.2f}s)"
        )

    # ========================================================
    # STEP 4 — RRF
    # ========================================================

    print("\n" + "-" * 100)
    print("STEP 4 — RRF RANKING")
    print("-" * 100)

    start = time.perf_counter()

    fused_docs = reciprocal_rank_fusion(
        retrieval_results
    )

    rrf_time = time.perf_counter() - start

    print(f"Time: {rrf_time:.4f}s")
    print(f"Unique fused candidates: {len(fused_docs)}")

    for rank, document in enumerate(
        fused_docs[:TOP_RRF_RESULTS],
        start=1,
    ):
        print(
            f"\nRRF #{rank}"
            f" | Source: {source_name(document)}"
        )

        print(
            f"Preview:\n{preview(document)}"
        )

    # ========================================================
    # STEP 5 — Intent-aware reranking
    # ========================================================

    print("\n" + "-" * 100)
    print("STEP 5 — INTENT-AWARE RERANKING")
    print("-" * 100)

    start = time.perf_counter()

    reranked_docs = rerank_documents(
        query=query,
        documents=fused_docs,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )

    rerank_time = time.perf_counter() - start

    print(f"Time: {rerank_time:.4f}s")
    print(
        f"Reranked documents: {len(reranked_docs)}"
    )

    for rank, document in enumerate(
        reranked_docs,
        start=1,
    ):
        print(
            f"\nRERANKED #{rank}"
            f" | Source: {source_name(document)}"
        )

        print(
            f"Preview:\n{preview(document)}"
        )

    # ========================================================
    # STEP 6 — Final context
    # ========================================================

    print("\n" + "-" * 100)
    print("STEP 6 — FINAL EVIDENCE")
    print("-" * 100)

    final_docs = reranked_docs[:FINAL_CONTEXT_DOCUMENTS]

    for rank, document in enumerate(
        final_docs,
        start=1,
    ):
        print(
            f"\nEvidence #{rank}"
        )

        print(
            f"Source: {source_name(document)}"
        )

        print(
            f"Content:\n{preview(document, 800)}"
        )

    # ========================================================
    # STEP 7 — Summary
    # ========================================================

    print("\n" + "-" * 100)
    print("STEP 7 — SUMMARY")
    print("-" * 100)

    print(f"Rewrite time:      {rewrite_time:.2f}s")
    print(f"Multi-query time:  {multi_query_time:.2f}s")
    print(f"RRF time:          {rrf_time:.4f}s")
    print(f"Rerank time:       {rerank_time:.4f}s")
    print(f"RRF candidates:    {len(fused_docs)}")
    print(f"Final evidence:    {len(final_docs)}")

    print("\nFINAL SOURCES:")

    for index, document in enumerate(
        final_docs,
        start=1,
    ):
        print(
            f"{index}. {source_name(document)}"
        )

    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()