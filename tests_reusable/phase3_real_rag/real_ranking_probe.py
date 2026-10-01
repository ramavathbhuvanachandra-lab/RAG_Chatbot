"""
Phase 3 — Real Ranking A/B Probe
================================

Fast real-corpus comparison between:

    current backend.retriever.rerank_documents()
    vs
    backend.core.retrieval.ranking.rank_candidates()

No answer LLM.
No LangGraph.
No production code modification.

Run from repository root after copying this file:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/real_ranking_probe.py

Use one question:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/real_ranking_probe.py \
        --question "What is the admission process for IIT Jodhpur?"

Use the critical set:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/real_ranking_probe.py \
        --critical
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from backend.core.retrieval_contracts import RetrievalCandidate
from backend.core.rrf import fuse_ranked_lists
from backend.core.retrieval.ranking import rank_candidates


CRITICAL = (
    "What is the admission process for IIT Jodhpur?",
    "What documents are required during admission?",
    "How can I contact the admissions office?",
    "What is the hostel fee for students?",
    "What minor programs are available at IIT Jodhpur?",
    "What are the research areas offered by the Electrical Engineering department?",
    "What food and dining facilities are available on campus?",
    "What emergency medical facilities are available at IIT Jodhpur?",
    "hostel mein students ko kya rules follow karne hote hain?",
    "Mtech regstration kaise hota hai?",
)


def source(document) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return str(metadata.get("source", "") or "")
    return ""


def preview(document, limit: int = 260) -> str:
    text = " ".join(
        str(getattr(document, "page_content", "") or "").split()
    )
    return text if len(text) <= limit else text[:limit] + "…"


def legacy_rank(
    query: str,
    documents,
):
    import backend.retriever as legacy

    return list(
        legacy.rerank_documents(
            query=query,
            documents=list(documents),
            top_k=5,
        )
    )


def new_rank(
    query: str,
    documents,
):
    candidates = [
        RetrievalCandidate.from_document(
            document,
            source=source(document) or "unknown",
        )
        for document in documents
    ]

    ranked = rank_candidates(
        query,
        candidates,
        top_k=5,
    )

    return [
        candidate.document
        for candidate in ranked
    ]


def run(query: str) -> None:
    import backend.retriever as legacy

    print("\n" + "=" * 110)
    print(f"QUERY: {query}")
    print("=" * 110)

    dense = list(
        legacy.dense_retrieve(query)
    )
    bm25 = list(
        legacy.keyword_retrieve(query)
    )

    fused_candidates = list(
        fuse_ranked_lists(
            (dense, bm25),
            primary_query=query,
        )
    )

    fused_documents = [
        candidate.document
        for candidate in fused_candidates
    ]

    old = legacy_rank(
        query,
        fused_documents,
    )

    new = new_rank(
        query,
        fused_documents,
    )

    print(
        f"\nRaw Dense={len(dense)} "
        f"BM25={len(bm25)} "
        f"Fused={len(fused_documents)}"
    )

    print("\nLEGACY TOP 5")
    for index, document in enumerate(old, start=1):
        print(
            f"[{index}] {source(document)}"
        )
        print(
            f"    {preview(document)}"
        )

    print("\nNEW CORE RANKING TOP 5")
    for index, document in enumerate(new, start=1):
        print(
            f"[{index}] {source(document)}"
        )
        print(
            f"    {preview(document)}"
        )

    old_sources = [
        source(document)
        for document in old
    ]
    new_sources = [
        source(document)
        for document in new
    ]

    if old_sources == new_sources:
        comparison = "UNCHANGED"
    else:
        comparison = "CHANGED"

    print(
        f"\nA/B ranking: {comparison}"
    )

    if old and new:
        print(
            f"Legacy top-1: {old_sources[0]}"
        )
        print(
            f"New top-1:    {new_sources[0]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare legacy and new ranking on the real corpus."
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--question",
        help="One real query.",
    )

    group.add_argument(
        "--critical",
        action="store_true",
        help="Run the critical ranking regression set.",
    )

    args = parser.parse_args()

    queries = (
        (args.question,)
        if args.question
        else CRITICAL
    )

    for query in queries:
        run(query)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())