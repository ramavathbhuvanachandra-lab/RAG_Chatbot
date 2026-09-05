import argparse
import sys
import time
from pathlib import Path


# =========================================================
# Project Path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Imports
# =========================================================

from backend.nodes import (
    rewrite_query,
    generate_multi_query,
)

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
)


# =========================================================
# Configuration
# =========================================================

TOP_K = 5


# =========================================================
# Questions
# =========================================================

QUESTIONS = {
    "ee_research":
        "What research areas are available in Electrical Engineering?",

    "btech_fees":
        "What are the fees for B.Tech students?",

    "hostel_facilities":
        "What are the hostel facilities for students?",
}


# =========================================================
# Retrieval Helper
# =========================================================

def retrieve_query(query):
    """
    Run the current production retrieval foundation.

    Dense + BM25 + weighted RRF.
    """

    dense_docs = dense_retrieve(
        query
    )

    bm25_docs = keyword_retrieve(
        query
    )

    fused_docs = reciprocal_rank_fusion(
        [
            dense_docs,
            bm25_docs,
        ]
    )

    return fused_docs[:TOP_K]


# =========================================================
# Source
# =========================================================

def source(document):
    return str(
        document.metadata.get(
            "source",
            "UNKNOWN",
        )
    ).replace(
        "\\",
        "/",
    )


# =========================================================
# Chunk Display
# =========================================================

def show_evidence(
    label,
    documents,
):
    print()
    print(
        "=" * 100
    )

    print(label)

    print(
        "=" * 100
    )

    for rank, document in enumerate(
        documents,
        start=1,
    ):

        print()
        print(
            f"#{rank} | {source(document)}"
        )

        text = document.page_content.strip()

        if len(text) > 600:
            text = (
                text[:600]
                + " ..."
            )

        print(text)


# =========================================================
# Main Test
# =========================================================

def run_question(question):
    print()
    print(
        "=" * 110
    )

    print(
        "IITJ V1 — PHASE 5 MULTI-QUERY AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        f"\nOriginal question:\n{question}"
    )

    # =====================================================
    # A — ORIGINAL QUERY
    # =====================================================

    start = time.perf_counter()

    original_docs = retrieve_query(
        question
    )

    original_time = (
        time.perf_counter()
        - start
    )

    show_evidence(
        "A — ORIGINAL QUERY RETRIEVAL",
        original_docs,
    )

    # =====================================================
    # B — REWRITE
    # =====================================================

    state = {
        "question": question,
        "chat_history": [],
    }

    start = time.perf_counter()

    rewritten_state = rewrite_query(
        state
    )

    rewrite_time = (
        time.perf_counter()
        - start
    )

    rewritten_question = (
        rewritten_state[
            "rewritten_question"
        ]
    )

    print()
    print(
        "=" * 100
    )

    print(
        "B — QUERY REWRITE"
    )

    print(
        "=" * 100
    )

    print(
        f"Time: {rewrite_time:.3f}s"
    )

    print(
        f"Rewritten:\n{rewritten_question}"
    )

    # =====================================================
    # C — MULTI QUERY
    # =====================================================

    multi_state = {
        "rewritten_question":
            rewritten_question,
    }

    start = time.perf_counter()

    generated_state = (
        generate_multi_query(
            multi_state
        )
    )

    multi_time = (
        time.perf_counter()
        - start
    )

    generated_queries = (
        generated_state[
            "generated_queries"
        ]
    )

    print()
    print(
        "=" * 100
    )

    print(
        "C — GENERATED QUERIES"
    )

    print(
        "=" * 100
    )

    print(
        f"Time: {multi_time:.3f}s"
    )

    for index, query in enumerate(
        generated_queries,
        start=1,
    ):

        print(
            f"{index}. {query}"
        )

    # =====================================================
    # D — SUPPORTING QUERY RETRIEVAL
    # =====================================================

    supporting_results = []

    for index, query in enumerate(
        generated_queries,
        start=1,
    ):

        if index == 1:
            continue

        supporting_results.extend(
            retrieve_query(
                query
            )
        )

    # =====================================================
    # E — ANCHOR-FIRST MULTI-QUERY
    # =====================================================

    combined_candidates = (
        list(original_docs)
        + supporting_results
    )

    # Deduplicate by document ID.
    seen = set()

    final_docs = []

    for document in combined_candidates:

        document_key = (
            document.page_content.strip()
        )

        if document_key in seen:
            continue

        seen.add(
            document_key
        )

        final_docs.append(
            document
        )

        if len(final_docs) >= TOP_K:
            break

    show_evidence(
        "D — ANCHOR-FIRST MULTI-QUERY EVIDENCE",
        final_docs,
    )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print(
        "=" * 100
    )

    print(
        "PHASE 5 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Original retrieval time: "
        f"{original_time:.3f}s"
    )

    print(
        f"Rewrite time: "
        f"{rewrite_time:.3f}s"
    )

    print(
        f"Multi-query generation time: "
        f"{multi_time:.3f}s"
    )

    print(
        f"Generated queries: "
        f"{len(generated_queries)}"
    )

    print()
    print(
        "Decision questions:"
    )

    print(
        "1. Does rewrite preserve the original intent?"
    )

    print(
        "2. Is query #1 essentially the original question?"
    )

    print(
        "3. Do Q2/Q3 add genuinely useful evidence?"
    )

    print(
        "4. Do Q2/Q3 introduce unrelated evidence?"
    )

    print(
        "5. Does multi-query recover evidence the original missed?"
    )

    print(
        "6. Is the improvement worth the extra LLM latency?"
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        required=True,
        choices=QUESTIONS.keys(),
    )

    args = parser.parse_args()

    run_question(
        QUESTIONS[
            args.id
        ]
    )


if __name__ == "__main__":
    main()