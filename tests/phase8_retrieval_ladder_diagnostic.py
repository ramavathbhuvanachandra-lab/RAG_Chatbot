"""
IITJ V1 — Current Retrieval Ladder Diagnostic

Purpose
-------
Identify exactly where a known-good corpus fact disappears.

This diagnostic follows the current canonical query pipeline:

    1. question understanding
    2. retrieval query planning
    3. Dense + BM25 retrieval
    4. weighted RRF
    5. deduplication
    6. initial reranking
    7. local context expansion
    8. final reranking / scope handling
    9. evidence sufficiency
    10. evidence coverage
    11. context compression
    12. final context inspection

This file is diagnostic-only.
It does NOT modify production code.

Known IITJ corpus target for this diagnostic:
    "Adaptive control & robotics"
"""

from pathlib import Path
import sys


# =========================================================
# Project Root
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =========================================================
# Current Canonical Query Pipeline
# =========================================================

from backend.core.query_pipeline import (
    process_query,
)

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


# =========================================================
# Diagnostic Question
# =========================================================

QUESTION = (
    "What research areas are related to robotics at IIT Jodhpur?"
)


TARGET_PATTERNS = (
    "robotics",
    "adaptive control & robotics",
    "mechatronics",
    "cyber-physical",
)


# =========================================================
# Helpers
# =========================================================

def source(document):
    return str(
        document.metadata.get(
            "source",
            "",
        )
    ).replace(
        "\\",
        "/",
    )


def snippet(
    document,
    limit=260,
):
    text = str(
        document.page_content
    ).replace(
        "\n",
        " ",
    ).strip()

    if len(text) > limit:
        return (
            text[:limit]
            + "..."
        )

    return text


def contains_target(document):
    text = (
        str(
            document.page_content
        )
        + " "
        + source(document)
    ).lower()

    return any(
        pattern in text
        for pattern in TARGET_PATTERNS
    )


def print_documents(
    label,
    documents,
    limit=15,
):
    print()
    print("=" * 110)
    print(label)
    print("=" * 110)

    for index, document in enumerate(
        list(documents or [])[:limit],
        start=1,
    ):
        target_marker = (
            " <<< TARGET"
            if contains_target(document)
            else ""
        )

        print(
            f"[{index:02d}] "
            f"{source(document)}"
            f"{target_marker}"
        )

        print(
            f"     {snippet(document)}"
        )


def print_target_presence(
    label,
    documents,
):
    documents = list(
        documents or []
    )

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
            f"  TARGET -> "
            f"{source(document)}"
        )

        print(
            f"             "
            f"{snippet(document)}"
        )


# =========================================================
# Main Diagnostic
# =========================================================

def main():

    print("=" * 110)
    print(
        "IITJ V1 — CURRENT RETRIEVAL LADDER DIAGNOSTIC"
    )
    print("=" * 110)

    print(
        f"QUESTION:\n{QUESTION}"
    )

    # =====================================================
    # A. Canonical Query Understanding + Planning
    # =====================================================

    print()
    print("=" * 110)
    print(
        "A. QUERY UNDERSTANDING + PLANNING"
    )
    print("=" * 110)

    query_result = process_query(
        QUESTION
    )

    frame = query_result.frame
    plan = query_result.plan

    print(
        "\nORIGINAL QUESTION:"
    )

    print(
        frame.original_query
    )

    print(
        "\nNORMALIZED QUESTION:"
    )

    print(
        frame.normalized_query
    )

    print(
        "\nTARGET:"
    )

    print(
        frame.target
    )

    print(
        "\nREQUEST TYPE:"
    )

    print(
        frame.request_type
    )

    print(
        "\nCONFIDENCE:"
    )

    print(
        frame.confidence
    )

    print(
        "\nRETRIEVAL QUERIES:"
    )

    for query in (
        plan.queries
        or []
    ):
        print(
            f"  - {query}"
        )

    # =====================================================
    # B. Build Production-Like State
    # =====================================================

    state = {
        "question": QUESTION,
        "chat_history": [],
        "resolved_question": QUESTION,
        "generated_queries": list(
            plan.queries or []
        ),
    }

    # =====================================================
    # C. Direct Dense / BM25 Visibility
    # =====================================================

    print()
    print("=" * 110)
    print(
        "C. DIRECT DENSE / BM25 VISIBILITY"
    )
    print("=" * 110)

    queries = list(
        plan.queries or []
    )

    if not queries:
        queries = [
            QUESTION
        ]

    for query_index, query in enumerate(
        queries[:3],
        start=1,
    ):

        dense_documents = (
            retriever.dense_retrieve(
                query
            )
        )

        keyword_documents = (
            retriever.keyword_retrieve(
                query
            )
        )

        print_documents(
            (
                f"DENSE Q{query_index}: "
                f"{query}"
            ),
            dense_documents,
            limit=10,
        )

        print_target_presence(
            f"DENSE Q{query_index}",
            dense_documents,
        )

        print_documents(
            (
                f"BM25 Q{query_index}: "
                f"{query}"
            ),
            keyword_documents,
            limit=10,
        )

        print_target_presence(
            f"BM25 Q{query_index}",
            keyword_documents,
        )

    # =====================================================
    # D. Hybrid Retrieval
    # =====================================================

    print()
    print("=" * 110)
    print(
        "D. HYBRID RETRIEVAL"
    )
    print("=" * 110)

    state.update(
        hybrid_retrieve(
            state
        )
    )

    flat_candidates = []

    for result_list in state.get(
        "retrieval_results",
        [],
    ):
        flat_candidates.extend(
            result_list or []
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
            "retrieval_weights",
            [],
        )
    )

    # =====================================================
    # E. Weighted RRF + Deduplication
    # =====================================================

    print()
    print("=" * 110)
    print(
        "E. RRF + DEDUPLICATION"
    )
    print("=" * 110)

    state.update(
        fuse_retrieved_documents(
            state
        )
    )

    fused_documents = state.get(
        "fused_docs",
        [],
    )

    print_target_presence(
        "AFTER RRF + DEDUP",
        fused_documents,
    )

    print_documents(
        "TOP 30 AFTER RRF + DEDUP",
        fused_documents,
        limit=30,
    )

    # =====================================================
    # F. Initial Reranking
    # =====================================================

    print()
    print("=" * 110)
    print(
        "F. INITIAL RERANK"
    )
    print("=" * 110)

    state.update(
        initial_rerank_documents(
            state
        )
    )

    initial_documents = state.get(
        "initial_reranked_docs",
        [],
    )

    print_target_presence(
        "AFTER INITIAL RERANK",
        initial_documents,
    )

    print_documents(
        "INITIAL RERANKED DOCUMENTS",
        initial_documents,
        limit=20,
    )

    # =====================================================
    # G. Local Context Expansion
    # =====================================================

    print()
    print("=" * 110)
    print(
        "G. LOCAL CONTEXT EXPANSION"
    )
    print("=" * 110)

    state.update(
        expand_retrieved_context(
            state
        )
    )

    expanded_documents = state.get(
        "expanded_docs",
        [],
    )

    print_target_presence(
        "AFTER LOCAL CONTEXT",
        expanded_documents,
    )

    print_documents(
        "EXPANDED DOCUMENTS",
        expanded_documents,
        limit=20,
    )

    # =====================================================
    # H. Final Reranking / Scope
    # =====================================================

    print()
    print("=" * 110)
    print(
        "H. FINAL RERANK / SCOPE"
    )
    print("=" * 110)

    state.update(
        final_rerank_documents(
            state
        )
    )

    reranked_documents = state.get(
        "reranked_docs",
        [],
    )

    print_target_presence(
        "AFTER FINAL RERANK / SCOPE",
        reranked_documents,
    )

    print_documents(
        "FINAL RERANKED DOCUMENTS",
        reranked_documents,
        limit=20,
    )

    # =====================================================
    # I. Evidence Sufficiency
    # =====================================================

    print()
    print("=" * 110)
    print(
        "I. EVIDENCE SUFFICIENCY"
    )
    print("=" * 110)

    state.update(
        assess_evidence_node(
            state
        )
    )

    print(
        "EVIDENCE STATUS:",
        state.get(
            "evidence_status"
        ),
    )

    print(
        "EVIDENCE SCORE:",
        state.get(
            "evidence_score"
        ),
    )

    print(
        "QUESTION TYPE:",
        state.get(
            "evidence_question_type"
        ),
    )

    print(
        "STRONG DOCUMENTS:",
        state.get(
            "evidence_strong_documents"
        ),
    )

    print(
        "PARTIAL DOCUMENTS:",
        state.get(
            "evidence_partial_documents"
        ),
    )

    # =====================================================
    # J. Evidence Coverage
    # =====================================================

    print()
    print("=" * 110)
    print(
        "J. EVIDENCE COVERAGE"
    )
    print("=" * 110)

    state.update(
        assess_evidence_coverage_node(
            state
        )
    )

    print(
        "COVERAGE STATUS:",
        state.get(
            "evidence_coverage_status"
        ),
    )

    # =====================================================
    # K. Context Compression
    # =====================================================

    print()
    print("=" * 110)
    print(
        "K. CONTEXT COMPRESSION"
    )
    print("=" * 110)

    state.update(
        compress_context(
            state
        )
    )

    compressed_documents = state.get(
        "compressed_docs",
        [],
    )

    print_target_presence(
        "FINAL COMPRESSED CONTEXT",
        compressed_documents,
    )

    print_documents(
        "FINAL COMPRESSED DOCUMENTS",
        compressed_documents,
        limit=20,
    )

    # =====================================================
    # L. Final Context Check
    # =====================================================

    final_context = retriever.format_context(
        compressed_documents
    )

    final_context_lower = (
        final_context.lower()
    )

    print()
    print("=" * 110)
    print(
        "L. FINAL CONTEXT TARGET CHECK"
    )
    print("=" * 110)

    print(
        "contains 'robotics':",
        "robotics" in final_context_lower,
    )

    print(
        "contains 'adaptive control':",
        "adaptive control" in final_context_lower,
    )

    print(
        "contains 'cyber-physical':",
        "cyber-physical" in final_context_lower,
    )

    print(
        "context chars:",
        len(final_context),
    )

    # =====================================================
    # M. Final Summary
    # =====================================================

    print()
    print("=" * 110)
    print(
        "M. DIAGNOSTIC SUMMARY"
    )
    print("=" * 110)

    print(
        "Query confidence:",
        frame.confidence,
    )

    print(
        "Generated query count:",
        len(
            plan.queries or []
        ),
    )

    print(
        "Hybrid candidate count:",
        len(
            flat_candidates
        ),
    )

    print(
        "Fused document count:",
        len(
            fused_documents
        ),
    )

    print(
        "Initial reranked count:",
        len(
            initial_documents
        ),
    )

    print(
        "Expanded document count:",
        len(
            expanded_documents
        ),
    )

    print(
        "Final reranked count:",
        len(
            reranked_documents
        ),
    )

    print(
        "Compressed document count:",
        len(
            compressed_documents
        ),
    )

    print(
        "Evidence status:",
        state.get(
            "evidence_status"
        ),
    )

    print(
        "Coverage status:",
        state.get(
            "evidence_coverage_status"
        ),
    )

    print()
    print("=" * 110)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()