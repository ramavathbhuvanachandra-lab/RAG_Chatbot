"""
IIT Jodhpur V1 — Multi-Query A/B Retrieval Test

Purpose
-------
Compare:

A:
    Original question
    -> Dense + BM25
    -> RRF
    -> deterministic reranking

B:
    Original question
    -> Rewrite
    -> Multi-query generation
    -> Dense + BM25
    -> RRF
    -> deterministic reranking

The test measures whether multi-query:

    - improves retrieval
    - improves confidence
    - discovers useful new evidence
    - loses evidence found by the original query
    - introduces unrelated evidence
    - adds significant latency

This test does NOT generate the final chatbot answer.
"""


import argparse
import json
import sys
import time
from pathlib import Path


# =========================================================
# Project Path
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# =========================================================
# Project Imports
# =========================================================

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
)

from tests.run_regression import (
    calculate_confidence,
    expected_term_coverage,
    query_term_coverage,
    source_diversity,
    detect_noise,
)


# =========================================================
# Configuration
# =========================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "tests"
    / "regression_questions.json"
)

FINAL_K = 5


# =========================================================
# Question Loading
# =========================================================

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# =========================================================
# Source Helper
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
# Result Summary
# =========================================================

def summarize(
    item,
    final_docs,
):

    confidence = calculate_confidence(
        item,
        final_docs,
    )

    expected = expected_term_coverage(
        item,
        final_docs,
    )

    query_coverage = query_term_coverage(
        item["question"],
        final_docs,
    )

    diversity = source_diversity(
        final_docs
    )

    noise = detect_noise(
        item["question"],
        final_docs,
    )

    return {
        "confidence":
            confidence,
        "expected":
            expected,
        "query_coverage":
            query_coverage,
        "diversity":
            diversity,
        "noise":
            noise,
    }


# =========================================================
# Retrieval A
# =========================================================

def run_original_query(
    question,
):

    start = time.perf_counter()

    dense_docs = dense_retrieve(
        question
    )

    bm25_docs = keyword_retrieve(
        question
    )

    fused_docs = reciprocal_rank_fusion(
        [
            dense_docs,
            bm25_docs,
        ]
    )

    final_docs = rerank_documents(
        query=question,
        documents=fused_docs,
        top_k=FINAL_K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "dense":
            dense_docs,
        "bm25":
            bm25_docs,
        "fused":
            fused_docs,
        "final":
            final_docs,
        "time":
            elapsed,
    }


# =========================================================
# Retrieval B
# =========================================================

def run_multi_query(
    question,
):

    total_start = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Rewrite
    # -----------------------------------------------------

    state = {
        "question": question,
        "chat_history": [],
    }

    rewrite_start = (
        time.perf_counter()
    )

    rewritten_state = rewrite_query(
        state
    )

    rewritten_question = (
        rewritten_state[
            "rewritten_question"
        ]
    )

    rewrite_time = (
        time.perf_counter()
        - rewrite_start
    )

    # -----------------------------------------------------
    # Multi-query generation
    # -----------------------------------------------------

    multi_start = (
        time.perf_counter()
    )

    generated_state = (
        generate_multi_query(
            {
                **state,
                **rewritten_state,
            }
        )
    )

    generated_queries = (
        generated_state[
            "generated_queries"
        ]
    )

    multi_time = (
        time.perf_counter()
        - multi_start
    )

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    retrieval_start = (
        time.perf_counter()
    )

    retrieval_lists = []

    per_query_results = []

    for query in generated_queries:

        dense_docs = dense_retrieve(
            query
        )

        bm25_docs = keyword_retrieve(
            query
        )

        retrieval_lists.append(
            dense_docs
        )

        retrieval_lists.append(
            bm25_docs
        )

        per_query_results.append(
            {
                "query":
                    query,
                "dense":
                    dense_docs,
                "bm25":
                    bm25_docs,
            }
        )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------

    rrf_start = (
        time.perf_counter()
    )

    fused_docs = reciprocal_rank_fusion(
        retrieval_lists
    )

    rrf_time = (
        time.perf_counter()
        - rrf_start
    )

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------

    rerank_start = (
        time.perf_counter()
    )

    final_docs = rerank_documents(
        query=question,
        documents=fused_docs,
        top_k=FINAL_K,
    )

    rerank_time = (
        time.perf_counter()
        - rerank_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    return {
        "rewritten_question":
            rewritten_question,

        "generated_queries":
            generated_queries,

        "per_query_results":
            per_query_results,

        "fused":
            fused_docs,

        "final":
            final_docs,

        "rewrite_time":
            rewrite_time,

        "multi_query_time":
            multi_time,

        "retrieval_time":
            retrieval_time,

        "rrf_time":
            rrf_time,

        "rerank_time":
            rerank_time,

        "total_time":
            total_time,
    }


# =========================================================
# Evidence Overlap
# =========================================================

def compare_evidence(
    original_docs,
    multi_docs,
):

    original_ids = {
        get_document_id(
            document
        )
        for document in original_docs
    }

    multi_ids = {
        get_document_id(
            document
        )
        for document in multi_docs
    }

    overlap = (
        original_ids
        & multi_ids
    )

    lost = (
        original_ids
        - multi_ids
    )

    new = (
        multi_ids
        - original_ids
    )

    return {
        "overlap":
            overlap,
        "lost":
            lost,
        "new":
            new,
    }


# =========================================================
# Print Documents
# =========================================================

def print_documents(
    title,
    documents,
):

    print(
        f"\n{title}"
    )

    if not documents:

        print(
            "  NONE"
        )

        return

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print(
            f"  {index}. "
            f"{source(document)}"
        )


# =========================================================
# Single Comparison
# =========================================================

def run_comparison(
    item,
):

    question = item[
        "question"
    ]

    print()
    print("=" * 100)
    print(
        f"QUESTION: {question}"
    )
    print("=" * 100)

    # =====================================================
    # A — Original query
    # =====================================================

    print(
        "\nA — ORIGINAL QUERY RETRIEVAL"
    )

    result_a = run_original_query(
        question
    )

    summary_a = summarize(
        item,
        result_a["final"],
    )

    print(
        f"Time: "
        f"{result_a['time']:.3f}s"
    )

    print(
        f"Confidence: "
        f"{summary_a['confidence']['score']:.2f}"
    )

    print(
        f"Status: "
        f"{summary_a['confidence']['status']}"
    )

    print(
        f"Expected coverage: "
        f"{summary_a['expected']['ratio']:.2f}"
    )

    print(
        f"Query coverage: "
        f"{summary_a['query_coverage']['ratio']:.2f}"
    )

    print_documents(
        "A — FINAL EVIDENCE",
        result_a["final"],
    )

    # =====================================================
    # B — Multi-query
    # =====================================================

    print(
        "\nB — REWRITE + MULTI-QUERY RETRIEVAL"
    )

    result_b = run_multi_query(
        question
    )

    summary_b = summarize(
        item,
        result_b["final"],
    )

    print(
        f"Rewrite time: "
        f"{result_b['rewrite_time']:.3f}s"
    )

    print(
        f"Multi-query time: "
        f"{result_b['multi_query_time']:.3f}s"
    )

    print(
        f"Retrieval time: "
        f"{result_b['retrieval_time']:.3f}s"
    )

    print(
        f"RRF time: "
        f"{result_b['rrf_time']:.4f}s"
    )

    print(
        f"Rerank time: "
        f"{result_b['rerank_time']:.4f}s"
    )

    print(
        f"Total time: "
        f"{result_b['total_time']:.3f}s"
    )

    print(
        f"Rewritten question: "
        f"{result_b['rewritten_question']}"
    )

    print(
        "\nGenerated queries:"
    )

    for index, query in enumerate(
        result_b["generated_queries"],
        start=1,
    ):

        print(
            f"  {index}. {query}"
        )

    print(
        f"\nConfidence: "
        f"{summary_b['confidence']['score']:.2f}"
    )

    print(
        f"Status: "
        f"{summary_b['confidence']['status']}"
    )

    print(
        f"Expected coverage: "
        f"{summary_b['expected']['ratio']:.2f}"
    )

    print(
        f"Query coverage: "
        f"{summary_b['query_coverage']['ratio']:.2f}"
    )

    print_documents(
        "B — FINAL EVIDENCE",
        result_b["final"],
    )

    # =====================================================
    # Comparison
    # =====================================================

    comparison = compare_evidence(
        result_a["final"],
        result_b["final"],
    )

    print(
        "\nC — A/B COMPARISON"
    )

    confidence_delta = (
        summary_b["confidence"]["score"]
        -
        summary_a["confidence"]["score"]
    )

    time_delta = (
        result_b["total_time"]
        -
        result_a["time"]
    )

    print(
        f"Confidence delta: "
        f"{confidence_delta:+.2f}"
    )

    print(
        f"Latency delta: "
        f"{time_delta:+.3f}s"
    )

    print(
        f"Evidence overlap: "
        f"{len(comparison['overlap'])}"
    )

    print(
        f"Evidence lost from A: "
        f"{len(comparison['lost'])}"
    )

    print(
        f"New evidence introduced by B: "
        f"{len(comparison['new'])}"
    )

    # -----------------------------------------------------
    # Lost evidence
    # -----------------------------------------------------

    if comparison["lost"]:

        print(
            "\nPotentially lost A evidence:"
        )

        for document in result_a[
            "final"
        ]:

            document_id = (
                get_document_id(
                    document
                )
            )

            if (
                document_id
                in comparison["lost"]
            ):

                print(
                    f"  - "
                    f"{source(document)}"
                )

    # -----------------------------------------------------
    # New evidence
    # -----------------------------------------------------

    if comparison["new"]:

        print(
            "\nNew B evidence:"
        )

        for document in result_b[
            "final"
        ]:

            document_id = (
                get_document_id(
                    document
                )
            )

            if (
                document_id
                in comparison["new"]
            ):

                print(
                    f"  + "
                    f"{source(document)}"
                )

    # -----------------------------------------------------
    # Decision signal
    # -----------------------------------------------------

    print(
        "\nD — INTERPRETATION"
    )

    if (
        confidence_delta >= 0.10
        and time_delta < 3.0
    ):

        print(
            "  MULTI-QUERY LOOKS USEFUL"
        )

    elif (
        confidence_delta <= -0.10
    ):

        print(
            "  MULTI-QUERY MAY BE HURTING RETRIEVAL"
        )

    elif (
        len(comparison["lost"]) > 0
        and confidence_delta <= 0
    ):

        print(
            "  WARNING: MULTI-QUERY LOST ORIGINAL EVIDENCE"
        )

    else:

        print(
            "  RESULT IS INCONCLUSIVE"
        )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        help="Run one regression question.",
    )

    parser.add_argument(
        "--index",
        type=int,
        help="Run one question by 1-based index.",
    )

    args = parser.parse_args()

    questions = load_questions()

    if args.id:

        questions = [
            item
            for item in questions
            if item["id"]
            == args.id
        ]

        if not questions:

            print(
                f"No question found: {args.id}"
            )

            sys.exit(1)

    elif args.index:

        if (
            args.index < 1
            or args.index > len(
                questions
            )
        ):

            print(
                "Invalid question index."
            )

            sys.exit(1)

        questions = [
            questions[
                args.index - 1
            ]
        ]

    for item in questions:

        # Unknown tests are still included so we can check
        # whether multi-query invents unrelated retrieval.
        run_comparison(
            item
        )


if __name__ == "__main__":
    main()