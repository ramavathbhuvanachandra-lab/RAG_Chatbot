from __future__ import annotations

import time
from pathlib import Path

from backend.core.query_understanding import understand_query
from backend.core.semantic_query_gate import decide_semantic_query

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    rerank_documents,
)


OUTPUT_PATH = Path(
    "test_outputs/phase2_semantic_gap_ab.txt"
)


TEST_CASES = [
    {
        "id": "Q1",
        "type": "semantic paraphrase",
        "query": (
            "Main reserved category ka student hoon, "
            "kya meri tuition maaf hai?"
        ),
        "expected_terms": [
            "tuition",
            "sc",
            "st",
            "exempt",
        ],
    },
    {
        "id": "Q2",
        "type": "semantic paraphrase",
        "query": (
            "Mehmaan ke liye ek room, ek din, bed ka samaan "
            "bhi chahiye — kitna padega?"
        ),
        "expected_terms": [
            "visitor",
            "single",
            "bedding",
        ],
    },
    {
        "id": "Q3",
        "type": "semantic paraphrase",
        "query": (
            "M.Tech mein jo subject reh gaya hai, "
            "use vacation term mein dobara lene ka process kya hai?"
        ),
        "expected_terms": [
            "m.tech",
            "summer",
            "backlog",
        ],
    },
    {
        "id": "Q4",
        "type": "semantic paraphrase",
        "query": (
            "Institute mein admission lene ke alag-alag modes "
            "kaun se hote hain?"
        ),
        "expected_terms": [
            "admission",
            "full-time",
            "part-time",
        ],
    },
    {
        "id": "Q5",
        "type": "unknown semantic trap",
        "query": (
            "Bhai woh free wala sachar kya scene hai, "
            "aur kitna paisa milta hai?"
        ),
        "expected_terms": [
            "free sachar",
        ],
    },
]


def source(document):
    metadata = getattr(
        document,
        "metadata",
        {},
    ) or {}

    return str(
        metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("path")
        or "UNKNOWN"
    )


def preview(document, limit=300):
    text = str(
        getattr(
            document,
            "page_content",
            "",
        ) or ""
    )

    text = " ".join(
        text.split()
    )

    return (
        text[:limit] + "..."
        if len(text) > limit
        else text
    )


def retrieve_baseline(query):
    dense = dense_retrieve(query)
    bm25 = keyword_retrieve(query)

    fused = reciprocal_rank_fusion(
        [dense, bm25],
        weights=[0.70, 0.30],
    )

    fused = deduplicate_documents(
        fused
    )

    return rerank_documents(
        query=query,
        documents=fused,
        top_k=5,
    )


def retrieve_semantic(
    query,
    semantic_query,
):
    if not semantic_query:
        return retrieve_baseline(
            query
        )

    dense_original = dense_retrieve(
        query
    )

    bm25_original = keyword_retrieve(
        query
    )

    dense_semantic = dense_retrieve(
        semantic_query
    )

    bm25_semantic = keyword_retrieve(
        semantic_query
    )

    fused = reciprocal_rank_fusion(
        [
            dense_original,
            bm25_original,
            dense_semantic,
            bm25_semantic,
        ],
        weights=[
            0.70,
            0.30,
            0.35,
            0.15,
        ],
    )

    fused = deduplicate_documents(
        fused
    )

    return rerank_documents(
        query=query,
        documents=fused,
        top_k=5,
        query_variants=[
            semantic_query
        ],
    )


def term_hits(
    documents,
    expected_terms,
):
    text = " ".join(
        str(
            getattr(
                document,
                "page_content",
                "",
            ) or ""
        )
        for document in documents
    ).lower()

    return [
        term
        for term in expected_terms
        if term.lower() in text
    ]


def run_case(case):
    query = case["query"]

    print()
    print("=" * 100)
    print(
        case["id"],
        "-",
        case["type"],
    )
    print(
        "USER QUERY:",
        query,
    )
    print("=" * 100)

    understanding_start = time.perf_counter()

    understanding = understand_query(
        query
    )

    understanding_time = (
        time.perf_counter()
        - understanding_start
    )

    decision = decide_semantic_query(
        understanding
    )

    semantic_query = (
        decision.semantic_query
        if decision.use_semantic_query
        else None
    )

    print("\nUNDERSTANDING")
    print(
        "search_query:",
        understanding.search_query,
    )
    print(
        "intent:",
        understanding.intent,
    )
    print(
        "confidence:",
        understanding.confidence,
    )
    print(
        "latency_sec:",
        round(
            understanding_time,
            3,
        ),
    )

    print("\nGATE")
    print(
        "use_semantic:",
        decision.use_semantic_query,
    )
    print(
        "reason:",
        decision.reason,
    )

    baseline_start = time.perf_counter()

    baseline_docs = retrieve_baseline(
        query
    )

    baseline_time = (
        time.perf_counter()
        - baseline_start
    )

    semantic_start = time.perf_counter()

    semantic_docs = retrieve_semantic(
        query,
        semantic_query,
    )

    semantic_time = (
        time.perf_counter()
        - semantic_start
    )

    print("\nBASELINE")
    print(
        "latency_sec:",
        round(
            baseline_time,
            3,
        ),
    )
    print(
        "term_hits:",
        term_hits(
            baseline_docs,
            case["expected_terms"],
        ),
    )

    for i, document in enumerate(
        baseline_docs,
        start=1,
    ):
        print(
            f"\n#{i} {source(document)}"
        )
        print(
            preview(document)
        )

    print("\nSEMANTIC")
    print(
        "latency_sec:",
        round(
            semantic_time,
            3,
        ),
    )
    print(
        "term_hits:",
        term_hits(
            semantic_docs,
            case["expected_terms"],
        ),
    )

    for i, document in enumerate(
        semantic_docs,
        start=1,
    ):
        print(
            f"\n#{i} {source(document)}"
        )
        print(
            preview(document)
        )


def main():
    started = time.perf_counter()

    for case in TEST_CASES:
        run_case(case)

    total = (
        time.perf_counter()
        - started
    )

    print()
    print("=" * 100)
    print(
        "TOTAL RUNTIME:",
        round(total, 3),
        "seconds",
    )
    print(
        "OUTPUT:",
        OUTPUT_PATH,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
