from __future__ import annotations

import contextlib
import io
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
    "test_outputs/phase2_semantic_retrieval_ab_gated.txt"
)


TEST_CASES = [
    {
        "id": "Q1",
        "type": "normal",
        "query": "What are the tuition fees for SC/ST students?",
        "cues": ["tuition", "sc", "st"],
    },
    {
        "id": "Q2",
        "type": "qualifier-heavy",
        "query": (
            "Yaar visitor ke liye single occupancy chahiye, "
            "bedding ke saath, sirf 1 day ka kitna charge hai?"
        ),
        "cues": ["visitor", "single", "bedding", "charge"],
    },
    {
        "id": "Q3",
        "type": "hard natural / Hinglish",
        "query": (
            "Bhai M.Tech summer course mein backlog wala "
            "registration kaise kare?"
        ),
        "cues": ["mtech", "summer", "registration", "backlog"],
    },
    {
        "id": "Q4",
        "type": "multi-condition / ambiguous",
        "query": (
            "Yaar admission ke different categories kaun kaun se hain?"
        ),
        "cues": ["admission", "categories"],
    },
    {
        "id": "Q5",
        "type": "adversarial breaker",
        "query": (
            "Abey yaar free sachar kya hai aur iska exact benefit "
            "kitna milta hai?"
        ),
        "cues": ["free", "sachar"],
    },
]


def document_source(document) -> str:
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


def document_preview(
    document,
    limit: int = 320,
) -> str:
    text = str(
        getattr(
            document,
            "page_content",
            "",
        ) or ""
    )

    text = text.replace(
        "\n",
        " ",
    ).strip()

    if len(text) > limit:
        text = text[:limit] + "..."

    return text


def cue_hits(
    documents,
    cues,
):
    combined = " ".join(
        str(
            getattr(
                doc,
                "page_content",
                "",
            ) or ""
        )
        for doc in documents
    ).lower()

    return [
        cue
        for cue in cues
        if cue.lower() in combined
    ]


def retrieve_baseline(
    query: str,
):
    dense = dense_retrieve(
        query
    )

    bm25 = keyword_retrieve(
        query
    )

    fused = reciprocal_rank_fusion(
        [
            dense,
            bm25,
        ],
        weights=[
            0.70,
            0.30,
        ],
    )

    fused = deduplicate_documents(
        fused
    )

    return rerank_documents(
        query=query,
        documents=fused,
        top_k=5,
    )


def retrieve_gated_semantic(
    original_query: str,
    semantic_query: str | None,
):
    # ---------------------------------------------------------
    # Safety rule:
    # rejected semantic query means EXACTLY the baseline path.
    # ---------------------------------------------------------
    if not semantic_query:
        return retrieve_baseline(
            original_query
        )

    dense_original = dense_retrieve(
        original_query
    )

    bm25_original = keyword_retrieve(
        original_query
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
        query=original_query,
        documents=fused,
        top_k=5,
        query_variants=[
            semantic_query
        ],
    )


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

    # ---------------------------------------------------------
    # Semantic understanding + gate
    # ---------------------------------------------------------
    understanding_started = time.perf_counter()

    understanding = understand_query(
        query
    )

    understanding_latency = (
        time.perf_counter()
        - understanding_started
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
        "search_query :",
        understanding.search_query,
    )
    print(
        "intent       :",
        understanding.intent,
    )
    print(
        "is_list      :",
        understanding.is_list_question,
    )
    print(
        "confidence   :",
        understanding.confidence,
    )
    print(
        "latency_sec  :",
        round(
            understanding_latency,
            3,
        ),
    )

    print("\nGATE")
    print(
        "use_semantic :",
        decision.use_semantic_query,
    )
    print(
        "reason       :",
        decision.reason,
    )
    print(
        "retrieval_q  :",
        semantic_query
        or query,
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------
    baseline_started = time.perf_counter()

    baseline_docs = retrieve_baseline(
        query
    )

    baseline_latency = (
        time.perf_counter()
        - baseline_started
    )

    # ---------------------------------------------------------
    # Gated semantic retrieval
    # ---------------------------------------------------------
    semantic_started = time.perf_counter()

    semantic_docs = retrieve_gated_semantic(
        query,
        semantic_query,
    )

    semantic_latency = (
        time.perf_counter()
        - semantic_started
    )

    baseline_sources = [
        document_source(doc)
        for doc in baseline_docs
    ]

    semantic_sources = [
        document_source(doc)
        for doc in semantic_docs
    ]

    print("\nBASELINE")
    print(
        "latency_sec:",
        round(
            baseline_latency,
            3,
        ),
    )
    print(
        "cue_hits:",
        cue_hits(
            baseline_docs,
            case["cues"],
        ),
    )

    for index, doc in enumerate(
        baseline_docs,
        start=1,
    ):
        print(
            f"\n  #{index} "
            f"{document_source(doc)}"
        )
        print(
            "    ",
            document_preview(doc),
        )

    print("\nGATED SEMANTIC")
    print(
        "latency_sec:",
        round(
            semantic_latency,
            3,
        ),
    )
    print(
        "cue_hits:",
        cue_hits(
            semantic_docs,
            case["cues"],
        ),
    )

    for index, doc in enumerate(
        semantic_docs,
        start=1,
    ):
        print(
            f"\n  #{index} "
            f"{document_source(doc)}"
        )
        print(
            "    ",
            document_preview(doc),
        )

    print("\nCOMPARISON")
    print(
        "same_top1:",
        baseline_sources[:1]
        == semantic_sources[:1],
    )
    print(
        "same_sources:",
        baseline_sources
        == semantic_sources,
    )


def main():
    started = time.perf_counter()

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        for case in TEST_CASES:
            run_case(case)

    output = buffer.getvalue()

    print(output)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        output,
        encoding="utf-8",
    )

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


if __name__ == "__main__":
    main()
