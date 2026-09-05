
"""
IITJ V1 — Retrieval Ladder Diagnostic

Purpose
-------
Identify exactly where a known-good corpus fact disappears.

This script does NOT change production code. It runs:
    1. resolved question
    2. planner query set
    3. Dense + BM25 per query
    4. weighted RRF
    5. deduplication
    6. initial reranking
    7. final evidence pipeline
    8. final answer context

For the robotics test, the corpus is known to contain:
    "Adaptive control & robotics"

The diagnostic reports whether that signal survives each stage.
"""

from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.graph import create_graph
from backend import retriever
from backend.nodes import (
    hybrid_retrieve,
    fuse_retrieved_documents,
    initial_rerank_documents,
    expand_retrieved_context,
    final_rerank_documents,
    assess_evidence_node,
    assess_evidence_coverage_node,
    compress_context,
)
from backend.situation_retrieval_integration import (
    prepare_situation_retrieval_node,
)
from backend.final_evidence_scope import (
    filter_final_evidence_scope,
)
from backend.claim_context_filter import (
    filter_claim_context,
)


QUESTION = (
    "What research areas are related to robotics at IIT Jodhpur?"
)

TARGET_PATTERNS = (
    "robotics",
    "adaptive control & robotics",
    "mechatronics",
    "cyber-physical",
)


def source(document):
    return str(
        document.metadata.get("source", "")
    ).replace("\\", "/")


def snippet(document, limit=260):
    text = str(document.page_content).replace(
        "\n",
        " ",
    ).strip()

    if len(text) > limit:
        return text[:limit] + "..."

    return text


def contains_target(document):
    text = (
        str(document.page_content)
        + " "
        + source(document)
    ).lower()

    return any(
        pattern in text
        for pattern in TARGET_PATTERNS
    )


def print_documents(label, documents, limit=15):
    print()
    print("=" * 110)
    print(label)
    print("=" * 110)

    for index, document in enumerate(
        list(documents)[:limit],
        start=1,
    ):
        target = (
            " <<< TARGET"
            if contains_target(document)
            else ""
        )

        print(
            f"[{index:02d}] {source(document)}{target}"
        )
        print(
            f"     {snippet(document)}"
        )


def print_target_presence(label, documents):
    matches = [
        document
        for document in documents
        if contains_target(document)
    ]

    print(
        f"{label}: "
        f"{len(matches)} target documents "
        f"/ {len(documents)} total"
    )

    for document in matches[:10]:
        print(
            f"  TARGET -> {source(document)}"
        )
        print(
            f"             {snippet(document)}"
        )


def main():
    print("=" * 110)
    print("IITJ V1 — FINAL RETRIEVAL LADDER DIAGNOSTIC")
    print("=" * 110)
    print(f"QUESTION: {QUESTION}")

    graph = create_graph()

    # ---------------------------------------------------------
    # A. Conversation resolution / retrieval preparation
    # ---------------------------------------------------------
    state = {
        "question": QUESTION,
        "chat_history": [],
        "resolved_question": QUESTION,
    }

    try:
        prepared = prepare_situation_retrieval_node(
            state
        )
        state.update(prepared)
    except Exception as exc:
        print(
            f"\n[prepare_situation_retrieval] ERROR: {exc}"
        )

    print(
        f"\nRESOLVED QUESTION:\n"
        f"{state.get('resolved_question', QUESTION)}"
    )

    print(
        "\nGENERATED QUERIES:"
    )

    for query in (
        state.get("generated_queries", [])
        or state.get("retrieval_queries", [])
        or []
    ):
        print(f"  - {query}")

    # ---------------------------------------------------------
    # B. Direct Dense / BM25 visibility
    # ---------------------------------------------------------
    queries = [
        state.get(
            "resolved_question",
            QUESTION,
        )
    ]

    for query in (
        state.get("generated_queries", [])
        or []
    ):
        if query.casefold() in {
            item.casefold()
            for item in queries
        }:
            continue
        queries.append(query)

        if len(queries) >= 3:
            break

    for query_index, query in enumerate(
        queries,
        start=1,
    ):
        dense = retriever.dense_retrieve(query)
        bm25 = retriever.keyword_retrieve(query)

        print_documents(
            f"DENSE Q{query_index}: {query}",
            dense,
            limit=10,
        )

        print_target_presence(
            f"DENSE Q{query_index}",
            dense,
        )

        print_documents(
            f"BM25 Q{query_index}: {query}",
            bm25,
            limit=10,
        )

        print_target_presence(
            f"BM25 Q{query_index}",
            bm25,
        )

    # ---------------------------------------------------------
    # C. Production nodes, one stage at a time
    # ---------------------------------------------------------
    state = hybrid_retrieve(
        state
    )

    flat_candidates = []

    for result_list in state.get(
        "retrieval_results",
        [],
    ):
        flat_candidates.extend(
            result_list
        )

    print_target_presence(
        "HYBRID RAW CANDIDATES",
        flat_candidates,
    )

    print(
        "\nRETRIEVAL QUERIES USED:"
    )
    for query in state.get(
        "retrieval_queries",
        [],
    ):
        print(
            f"  - {query}"
        )

    print(
        "\nRETRIEVAL WEIGHTS:"
    )
    print(
        state.get(
            "retrieval_weights"
        )
    )

    state = fuse_retrieved_documents(
        state
    )

    fused = state.get(
        "fused_docs",
        [],
    )

    print_target_presence(
        "AFTER RRF + DEDUP",
        fused,
    )

    print_documents(
        "TOP 30 AFTER RRF + DEDUP",
        fused,
        limit=30,
    )

    state = initial_rerank_documents(
        state
    )

    initial = state.get(
        "initial_reranked_docs",
        [],
    )

    print_target_presence(
        "AFTER INITIAL RERANK",
        initial,
    )

    print_documents(
        "INITIAL RERANK",
        initial,
        limit=20,
    )

    state = expand_retrieved_context(
        state
    )

    expanded = state.get(
        "expanded_docs",
        [],
    )

    print_target_presence(
        "AFTER LOCAL CONTEXT",
        expanded,
    )

    state = final_rerank_documents(
        state
    )

    reranked = state.get(
        "reranked_docs",
        [],
    )

    print_target_presence(
        "AFTER FINAL RERANK / SCOPE FILTER",
        reranked,
    )

    print_documents(
        "FINAL RERANKED DOCUMENTS",
        reranked,
        limit=20,
    )

    state = assess_evidence_node(
        state
    )

    state = assess_evidence_coverage_node(
        state
    )

    print()
    print(
        "EVIDENCE STATUS:",
        state.get("evidence_status"),
    )
    print(
        "EVIDENCE SCORE:",
        state.get("evidence_score"),
    )
    print(
        "COVERAGE STATUS:",
        state.get("evidence_coverage_status"),
    )
    print(
        "QUESTION TYPE:",
        state.get("evidence_question_type"),
    )
    print(
        "STRONG DOCUMENTS:",
        state.get("evidence_strong_documents"),
    )
    print(
        "PARTIAL DOCUMENTS:",
        state.get("evidence_partial_documents"),
    )

    state = compress_context(
        state
    )

    compressed = state.get(
        "compressed_docs",
        [],
    )

    print_target_presence(
        "FINAL COMPRESSED CONTEXT",
        compressed,
    )

    print_documents(
        "FINAL COMPRESSED DOCUMENTS",
        compressed,
        limit=20,
    )

    final_context = retriever.format_context(
        compressed
    )

    print()
    print("=" * 110)
    print("FINAL CONTEXT TARGET CHECK")
    print("=" * 110)
    print(
        f"contains 'robotics': "
        f"{'robotics' in final_context.lower()}"
    )
    print(
        f"contains 'adaptive control': "
        f"{'adaptive control' in final_context.lower()}"
    )
    print(
        f"contains 'cyber-physical': "
        f"{'cyber-physical' in final_context.lower()}"
    )
    print(
        f"context chars: {len(final_context)}"
    )


if __name__ == "__main__":
    main()
