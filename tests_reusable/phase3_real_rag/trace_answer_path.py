"""
Phase 3 — Single-Question RAG Stage Trace
=========================================

Purpose
-------
Trace ONE real user question through the existing production graph while
DISABLING the final answer LLM call.

This isolates:

    conversation resolution
        ↓
    situation/query preparation
        ↓
    hybrid retrieval
        ↓
    RRF
        ↓
    initial reranking
        ↓
    local context
        ↓
    evidence groups
        ↓
    final reranking
        ↓
    evidence sufficiency
        ↓
    evidence coverage
        ↓
    final compressed context

The final answer LLM is replaced by a fake chain so the script is much faster
than a full QA run and, more importantly, exposes the exact context that would
have been sent to the answer model.

Run from repo root:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/trace_answer_path.py

Specific question:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/trace_answer_path.py \
        --question "What is the admission process for IIT Jodhpur?"

Critical regression set:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/trace_answer_path.py \
        --preset critical

The script NEVER modifies production files.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CRITICAL_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "admission-process",
        "What is the admission process for IIT Jodhpur?",
    ),
    (
        "hostel-fees",
        "What is the hostel fee for students?",
    ),
    (
        "minor-programs",
        "What minor programs are available at IIT Jodhpur?",
    ),
)


def _source(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return str(
            metadata.get("source", "")
            or ""
        )
    return ""


def _text(document: Any) -> str:
    return " ".join(
        str(
            getattr(document, "page_content", "")
            or ""
        ).split()
    )


def _preview(document: Any, limit: int = 360) -> str:
    value = _text(document)
    if len(value) > limit:
        return value[:limit] + "…"
    return value


def _print_documents(
    title: str,
    documents: Any,
    *,
    limit: int = 10,
) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)

    docs = list(
        documents or []
    )

    print(f"count={len(docs)}")

    for index, document in enumerate(
        docs[:limit],
        start=1,
    ):
        print(
            f"\n[{index}] source={_source(document)}"
        )
        print(
            f"    {_preview(document)}"
        )


def _print_candidate_stage(
    title: str,
    candidates: Any,
    *,
    limit: int = 10,
) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)

    items = list(
        candidates or []
    )

    print(f"count={len(items)}")

    for index, candidate in enumerate(
        items[:limit],
        start=1,
    ):
        document = getattr(
            candidate,
            "document",
            None,
        )

        provenance = getattr(
            candidate,
            "provenance",
            None,
        )

        alignment = getattr(
            candidate,
            "alignment",
            None,
        )

        quality = getattr(
            candidate,
            "quality",
            None,
        )

        print(
            f"\n[{index}] source={getattr(candidate, 'source', '')}"
        )
        print(
            f"    document_id={getattr(candidate, 'document_id', '')}"
        )

        if provenance is not None:
            print(
                "    provenance="
                f"rrf={getattr(provenance, 'rrf_score', None)} "
                f"dense_rank={getattr(provenance, 'dense_rank', None)} "
                f"bm25_rank={getattr(provenance, 'bm25_rank', None)} "
                f"primary_query={getattr(provenance, 'primary_query', '')!r}"
            )

        if alignment is not None:
            print(
                "    alignment="
                f"semantic={getattr(alignment, 'semantic_match', None)} "
                f"coverage={getattr(alignment, 'coverage', None)} "
                f"scope={getattr(alignment, 'scope_match', None)} "
                f"conflicts={getattr(alignment, 'conflicts', None)}"
            )

        if quality is not None:
            print(
                "    quality="
                f"score={getattr(quality, 'score', None)} "
                f"level={getattr(quality, 'level', None)}"
            )

        if document is not None:
            print(
                f"    {_preview(document)}"
            )


def _print_groups(
    groups: Any,
    *,
    limit: int = 10,
) -> None:
    print("\n" + "-" * 100)
    print("EVIDENCE GROUPS")
    print("-" * 100)

    items = list(
        groups or []
    )

    print(f"count={len(items)}")

    for index, group in enumerate(
        items[:limit],
        start=1,
    ):
        print(
            f"\n[{index}]"
        )

        # EvidenceGroup is a project object; keep this intentionally
        # reflection-based so the trace remains compatible as the object
        # evolves.
        if hasattr(group, "to_dict"):
            try:
                payload = group.to_dict()
                print(
                    f"    {payload}"
                )
                continue
            except Exception:
                pass

        for field in (
            "source",
            "score",
            "final_score",
            "group_score",
            "document_count",
            "documents",
            "evidence_documents",
        ):
            if hasattr(group, field):
                value = getattr(group, field)
                if field in {
                    "documents",
                    "evidence_documents",
                }:
                    print(
                        f"    {field}: {len(list(value or []))}"
                    )
                else:
                    print(
                        f"    {field}: {value!r}"
                    )


class _FakeResponse:
    content = ""


class _FakeAnswerChain:
    """
    Capture what generate_answer would send to the final answer model.

    Returning an empty response triggers the normal deterministic fallback,
    but all retrieval/evidence state remains available for inspection.
    """

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}

    def invoke(self, payload: dict[str, Any]) -> _FakeResponse:
        self.payload = dict(
            payload
        )
        return _FakeResponse()


def _build_graph_with_disabled_answer_llm() -> tuple[Any, _FakeAnswerChain]:
    nodes = importlib.import_module(
        "backend.nodes"
    )
    graph_module = importlib.import_module(
        "backend.graph"
    )

    fake = _FakeAnswerChain()

    original_chain = getattr(
        nodes,
        "answer_chain",
    )

    nodes.answer_chain = fake

    try:
        graph = graph_module.create_graph()
    except Exception:
        nodes.answer_chain = original_chain
        raise

    # Return the original chain so the caller can restore it after invoke().
    setattr(
        fake,
        "_original_chain",
        original_chain,
    )

    return graph, fake


def _run_one(
    case_id: str,
    question: str,
) -> None:
    print("\n" + "=" * 110)
    print(f"TRACE: {case_id}")
    print("=" * 110)
    print(f"Question: {question}")

    nodes = importlib.import_module(
        "backend.nodes"
    )
    graph_module = importlib.import_module(
        "backend.graph"
    )

    fake = _FakeAnswerChain()

    original_chain = nodes.answer_chain
    nodes.answer_chain = fake

    try:
        graph = graph_module.create_graph()

        result = graph.invoke(
            {
                "question": question,
                "chat_history": [],
            }
        )
    finally:
        nodes.answer_chain = original_chain

    print("\n" + "-" * 100)
    print("QUERY / CONVERSATION STATE")
    print("-" * 100)

    for key in (
        "resolved_question",
        "conversation_mode",
        "active_topic",
        "active_entity",
        "generated_queries",
        "retrieval_queries",
        "retrieval_weights",
        "is_multi_intent",
        "intent_count",
    ):
        print(
            f"{key}: {result.get(key)!r}"
        )

    retrieval_results = result.get(
        "retrieval_results",
        [],
    )

    if retrieval_results:
        for index, documents in enumerate(
            retrieval_results,
            start=1,
        ):
            _print_documents(
                f"RAW RETRIEVAL LIST {index}",
                documents,
                limit=5,
            )

    _print_documents(
        "FUSED DOCS",
        result.get(
            "fused_docs",
            [],
        ),
        limit=10,
    )

    _print_documents(
        "INITIAL RERANKED DOCS",
        result.get(
            "initial_reranked_docs",
            [],
        ),
        limit=10,
    )

    _print_documents(
        "EXPANDED DOCS",
        result.get(
            "expanded_docs",
            [],
        ),
        limit=10,
    )

    _print_groups(
        result.get(
            "evidence_groups",
            [],
        ),
        limit=10,
    )

    _print_documents(
        "FINAL RERANKED DOCS",
        result.get(
            "reranked_docs",
            [],
        ),
        limit=10,
    )

    _print_documents(
        "COMPRESSED / ANSWER DOCS",
        result.get(
            "compressed_docs",
            [],
        ),
        limit=10,
    )

    print("\n" + "-" * 100)
    print("EVIDENCE STATE")
    print("-" * 100)

    for key in (
        "evidence_status",
        "evidence_score",
        "relevant_evidence_documents",
        "evidence_coverage_status",
        "evidence_question_type",
        "evidence_strong_documents",
        "evidence_partial_documents",
        "evidence_combined_characters",
    ):
        print(
            f"{key}: {result.get(key)!r}"
        )

    print("\n" + "-" * 100)
    print("FINAL ANSWER-MODEL INPUT (CAPTURED WITHOUT LLM CALL)")
    print("-" * 100)

    for key, value in fake.payload.items():
        if key == "context":
            context = str(value or "")
            print(
                f"\n{key} (chars={len(context)}):\n{context}"
            )
        else:
            print(
                f"\n{key}:\n{value}"
            )

    print("\n" + "=" * 110)
    print(
        "TRACE COMPLETE — final answer LLM was NOT called."
    )
    print("=" * 110)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace one real RAG question without the answer LLM."
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Run one explicit question.",
    )
    parser.add_argument(
        "--preset",
        choices=("critical",),
        default=None,
        help="Run the critical retrieval regression questions.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.question:
        _run_one(
            "custom",
            args.question,
        )
        return 0

    if args.preset == "critical":
        for case_id, question in CRITICAL_QUESTIONS:
            _run_one(
                case_id,
                question,
            )
        return 0

    _run_one(
        "admission-process",
        "What is the admission process for IIT Jodhpur?",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
