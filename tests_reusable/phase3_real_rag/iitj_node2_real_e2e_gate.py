"""Real IITJ end-to-end E7/E8 gate and diagnostic report.

This gate runs the NEW CoreNodes path on the actual IITJ corpus:

    question
      -> query understanding
      -> bounded hybrid retrieval
      -> RRF
      -> candidate verification/ranking
      -> evidence grouping/scope
      -> evidence sufficiency/coverage
      -> claim audit
      -> E6 evidence package
      -> E7 answer generation (REAL answer model)
      -> E7.2 answer grounding
      -> E8 final output guard

The gate captures the exact user payload sent to the answer model.  The
payload is diagnostic output only; internal source metadata must not appear
inside the model-facing prompt.

It does NOT change production code.
"""

from __future__ import annotations

import argparse
from dataclasses import replace, is_dataclass, asdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.nodes import CoreNodes, default_dependencies
from backend.core.query.understanding import understand_query


CORPUS = ROOT / "data" / "data_iitj" / "iitj_rag_v1_docs_production"
REPORT_DIR = ROOT / "tests_reusable" / "phase3_real_rag" / "reports"
REPORT_JSON = REPORT_DIR / "node2_real_e2e_report.json"
REPORT_SUMMARY = REPORT_DIR / "node2_real_e2e_summary.txt"

DEFAULT_QUESTIONS = (
    "What are the M.Tech eligibility requirements?",
    "What are the admission routes for M.Sc.?",
    "What is the tuition fee?",
    "What are the hostel fees for students?",
    "How do I apply for admission?",
    "What are the eligibility criteria for admission?",
    "What academic programs are available?",
    "What departments are there?",
    "What research opportunities are available?",
    "What research areas are available?",
    "What facilities are available?",
    "What hostel facilities are available?",
    "What are the admission requirements for B.Tech?",
    "What are the admission requirements for Ph.D.?",
    "What is the procedure for applying?",
    "What are the fees for admission?",
    "Where is the institute located?",
    "Can I apply if I do not meet the stated eligibility requirements?",
    "Tell me about the institute's academic programs and research opportunities.",
    "What are the application deadlines?",
)

INTERNAL_LEAK_PATTERNS = (
    re.compile(r"\bDocument\s+#?\s*\d+\b", re.I),
    re.compile(r"\bchunk(?:\s+id|_id)\b", re.I),
    re.compile(r"\bretrieval\s+(?:rank|score)\b", re.I),
    re.compile(r"\bRRF\s+score\b", re.I),
    re.compile(r"\bsource\s+(?:path|file|filename)\b", re.I),
    re.compile(r"\bbackend/core/", re.I),
    re.compile(r"\b(?:data|storage)/[^\s]+\.(?:docx|pdf|json|txt)\b", re.I),
)


ANSWER_OUTPUT_LEAK_PATTERNS = (
    re.compile(r"\b(?:evidence|document|chunk)\s*\d+\b", re.I),
    re.compile(r"\b(?:the user is asking|the user asks|let me|first[, ]+i|i need to|looking at the evidence|looking through the evidence|the key is|the problem is|hmm|step by step|my reasoning|analysis:)\b", re.I),
)




def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _status(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, Mapping):
        raw = value.get("status")
    else:
        raw = getattr(value, "status", None)
    return _clean(raw).casefold() or default


def _answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("answer", "text", "content"):
            if key in value:
                return _clean(value[key])
    answer = getattr(value, "answer", None)
    if answer is not None:
        return _clean(answer)
    content = getattr(value, "content", None)
    if content is not None:
        return _clean(content)
    return _clean(value)


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(v) for v in value]
    if is_dataclass(value):
        return _serializable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _serializable(value.to_dict())
        except Exception:
            pass
    return _clean(value)


def _source(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        return _clean(
            metadata.get("source")
            or metadata.get("source_path")
            or metadata.get("path")
            or metadata.get("url")
        )
    return ""


def _package_items(package: Any) -> list[dict[str, Any]]:
    items = getattr(package, "items", ()) or ()
    result: list[dict[str, Any]] = []
    for item in items:
        result.append(
            {
                "evidence_id": _clean(getattr(item, "evidence_id", "")),
                "source": _clean(getattr(item, "source", "")),
                "text": _clean(getattr(item, "text", "")),
                "supported_claim_ids": list(
                    getattr(item, "supported_claim_ids", ()) or ()
                ),
                "partial_claim_ids": list(
                    getattr(item, "partial_claim_ids", ()) or ()
                ),
            }
        )
    return result


class TracingAnswerModel:
    """Wrap the REAL configured answer model and capture its exact payload."""

    def __init__(self, real_model: Any) -> None:
        self.real_model = real_model
        self.calls = 0
        self.payloads: list[Any] = []

    def invoke(self, payload: Any) -> Any:
        self.calls += 1
        self.payloads.append(payload)
        return self.real_model.invoke(payload)


def _real_model() -> Any:
    try:
        from backend.runtime.llm import answer_llm
        return answer_llm
    except (ImportError, AttributeError):
        from backend.llm import answer_llm
        return answer_llm


def _make_nodes() -> tuple[CoreNodes, TracingAnswerModel]:
    deps = default_dependencies()
    tracer = TracingAnswerModel(_real_model())
    deps = replace(deps, answer_model_provider=lambda _state: tracer)
    return CoreNodes(deps), tracer


def _assert_corpus() -> tuple[int, int]:
    assert CORPUS.is_dir(), f"REAL IITJ CORPUS NOT FOUND: {CORPUS}"
    docs = sorted(CORPUS.rglob("*.docx"))
    assert len(docs) >= 10, f"REAL IITJ CORPUS LOOKS INVALID: {len(docs)} DOCX files"
    return len(docs), len(docs)


def run_case(nodes: CoreNodes, tracer: TracingAnswerModel, question: str) -> dict[str, Any]:
    start = time.perf_counter()
    calls_before = tracer.calls

    qualified = nodes.retrieve_and_qualify_node(
        {
            "question": question,
            "resolved_question": question,
            "chat_history": (),
        }
    )

    retrieval_elapsed = time.perf_counter()

    answer_state = dict(qualified)
    answer_state["question"] = question
    answer_state["resolved_question"] = _clean(
        qualified.get("resolved_question") or question
    )
    answer = nodes.answer_node(answer_state)

    end = time.perf_counter()

    query = qualified.get("query")
    package = qualified.get("evidence_package")
    generation = answer.get("answer_generation")
    grounding = answer.get("answer_grounding")
    guard = answer.get("answer_guard")

    exact_user_payload = ""
    exact_system_payload = ""
    if tracer.payloads:
        payload = tracer.payloads[-1]
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            exact_system_payload = str(payload[0].get("content", ""))
            exact_user_payload = str(payload[1].get("content", ""))

    internal_leaks = [
        pattern.pattern
        for pattern in INTERNAL_LEAK_PATTERNS
        if pattern.search(exact_user_payload)
    ]

    package_ready = bool(getattr(package, "ready_for_generation", False))
    evidence_status = _status(qualified.get("evidence_assessment"), "")
    coverage_status = _status(qualified.get("coverage_assessment"), "")
    grounding_status = _status(grounding, "not_run")
    guard_status = _status(guard, "")
    final_answer = _answer_text(answer.get("answer"))
    generated_answer = _answer_text(generation)
    final_answer_leaks = [
        pattern.pattern
        for pattern in ANSWER_OUTPUT_LEAK_PATTERNS
        if pattern.search(final_answer)
    ]
    model_calls = int(answer.get("model_calls", 0) or 0)
    case_model_calls = tracer.calls - calls_before

    expected_generation = (
        package_ready
        and evidence_status not in {"insufficient", "conflicted"}
    )

    checks: list[tuple[str, bool]] = [
        ("answer_non_empty", bool(final_answer)),
        ("model_calls_max_one", model_calls in (0, 1)),
        ("tracer_calls_consistent", case_model_calls == model_calls),
    ]

    if expected_generation:
        checks.append(("generation_expected", model_calls == 1))
        checks.append(("generated_answer_non_empty", bool(generated_answer)))
        checks.append(("current_question_in_prompt", question in exact_user_payload))
        checks.append(("no_internal_prompt_leak", not internal_leaks))
        checks.append(("no_internal_answer_leak", not final_answer_leaks))
    else:
        checks.append(("fail_closed_no_generation", model_calls == 0))

    if grounding_status in {"review", "conflict", "unsafe"}:
        checks.append(("grounding_review_fails_closed", bool(answer.get("answer_guard_fallback_used"))))

    failed = [name for name, ok in checks if not ok]
    status = "FAIL" if failed else (
        "REVIEW" if grounding_status in {"review", "conflict", "unsafe"}
        else "PASS"
    )

    package_context = _clean(getattr(package, "context", ""))
    sources = sorted({item["source"] for item in _package_items(package) if item["source"]})

    return {
        "question": question,
        "status": status,
        "elapsed_ms": round((end - start) * 1000, 1),
        "qualification_elapsed_ms": round((retrieval_elapsed - start) * 1000, 1),
        "answer_elapsed_ms": round((end - retrieval_elapsed) * 1000, 1),
        "query": {
            "request_type": _clean(getattr(query, "request_type", "")),
            "target": _clean(getattr(getattr(query, "target", None), "text", "")),
            "is_list": bool(getattr(getattr(query, "list_intent", None), "is_list", False))
            if getattr(query, "list_intent", None) is not None
            else bool(getattr(query, "is_list", False)),
            "search_query": _clean(getattr(query, "search_query", "")),
            "confidence": float(getattr(query, "confidence", 0.0) or 0.0),
        },
        "retrieval": {
            "queries": list(qualified.get("retrieval_queries", ()) or ()),
            "dense_count": sum(
                len(x or ())
                for x in (qualified.get("retrieval_results", ()) or ())
                [0::3]
            ),
            "bm25_count": sum(
                len(x or ())
                for x in (qualified.get("retrieval_results", ()) or ())
                [1::3]
            ),
            "lexical_recall_count": int(qualified.get("lexical_recall_count", 0) or 0),
            "lexical_recall_queries": list(qualified.get("lexical_recall_queries", ()) or ()),
            "fused_count": len(qualified.get("fused_candidates", ()) or ()),
            "verified_count": len(qualified.get("verified_candidates", ()) or ()),
            "ranked_count": len(qualified.get("ranked_candidates", ()) or ()),
            "evidence_document_count": len(qualified.get("evidence_documents", ()) or ()),
            "evidence_candidate_count": len(qualified.get("evidence_candidates", ()) or ()),
            "lexical_fallback_used": bool(qualified.get("lexical_fallback_used", False)),
            "lexical_fallback_candidates": int(qualified.get("lexical_fallback_candidates", 0) or 0),
            "lexical_fallback_trace": list(qualified.get("lexical_fallback_trace", ()) or ()),
            "ranked_trace": list(qualified.get("retrieval_trace", ()) or ()),
        },
        "evidence": {
            "status": evidence_status,
            "coverage_status": coverage_status,
            "package_status": _status(package, "empty"),
            "package_ready": package_ready,
            "context_characters": len(package_context),
            "source_count": int(getattr(package, "source_count", 0) or 0),
            "selected_count": int(getattr(package, "selected_count", 0) or 0),
            "sources": sources,
            "exact_context_sent_to_answer_model": package_context,
            "package_items": _package_items(package),
        },
        "answer_generation": {
            "generated": bool(getattr(generation, "generated", False)),
            "model_calls": model_calls,
            "actual_answer_model_calls": case_model_calls,
            "reason": _clean(getattr(generation, "reason", "")),
            "evidence_status": _clean(getattr(generation, "evidence_status", "")),
            "raw_generated_answer": generated_answer,
            "answer_mode": _clean(getattr(generation, "answer_mode", "")),
        },
        "grounding": {
            "status": grounding_status,
            "details": _serializable(grounding),
        },
        "guard": {
            "status": guard_status,
            "fallback_used": bool(answer.get("answer_guard_fallback_used", False)),
            "reason": _clean(answer.get("answer_guard_reason")),
            "details": _serializable(guard),
        },
        "final_answer": final_answer,
        "model_prompt": {
            "system": exact_system_payload,
            "user": exact_user_payload,
            "internal_leak_patterns_found": internal_leaks,
            "final_answer_leak_patterns_found": final_answer_leaks,
        },
        "checks": {name: ok for name, ok in checks},
        "failed_checks": failed,
    }


def write_reports(results: list[dict[str, Any]], corpus_files: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "NODE 2 IITJ REAL END-TO-END E7/E8 REPORT",
        "===========================================",
        f"corpus_docx_files={corpus_files}",
        f"cases={len(results)}",
        f"PASS={sum(r['status'] == 'PASS' for r in results)}",
        f"REVIEW={sum(r['status'] == 'REVIEW' for r in results)}",
        f"FAIL={sum(r['status'] == 'FAIL' for r in results)}",
        "",
    ]

    for index, result in enumerate(results, start=1):
        summary_lines.extend(
            [
                f"[{index:02d}] {result['status']} | {result['question']}",
                f"     request_type={result['query']['request_type']} | target={result['query']['target']} | list={result['query']['is_list']}",
                f"     retrieval={result['retrieval']['dense_count']}d/{result['retrieval']['bm25_count']}bm25/{result['retrieval']['lexical_recall_count']}lex -> fused={result['retrieval']['fused_count']} -> verified={result['retrieval']['verified_count']} -> ranked={result['retrieval']['ranked_count']}",
                f"     evidence={result['evidence']['status']} | coverage={result['evidence']['coverage_status']} | package_ready={result['evidence']['package_ready']} | context_chars={result['evidence']['context_characters']}",
                f"     generation_calls={result['answer_generation']['model_calls']} | grounding={result['grounding']['status']} | guard={result['guard']['status']} | fallback={result['guard']['fallback_used']}",
                f"     elapsed_ms={result['elapsed_ms']}",
            ]
        )
        if result["failed_checks"]:
            summary_lines.append(f"     FAILED_CHECKS={result['failed_checks']}")
        summary_lines.append("")

    REPORT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps(
            {
                "corpus": str(CORPUS),
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--question", type=str, default=None)
    args = parser.parse_args()

    corpus_files, _ = _assert_corpus()
    nodes, tracer = _make_nodes()

    questions = (args.question,) if args.question else DEFAULT_QUESTIONS[: max(1, min(args.limit, len(DEFAULT_QUESTIONS)))]
    results: list[dict[str, Any]] = []

    for question in questions:
        print(f"RUN: {question}")
        result = run_case(nodes, tracer, question)
        results.append(result)
        print(
            f"  {result['status']} | "
            f"evidence={result['evidence']['status']} | "
            f"coverage={result['evidence']['coverage_status']} | "
            f"calls={result['answer_generation']['model_calls']} | "
            f"grounding={result['grounding']['status']} | "
            f"guard={result['guard']['status']}"
        )
        tracer.payloads.clear()

    write_reports(results, corpus_files)

    failures = [r for r in results if r["status"] == "FAIL"]
    print() 
    print("NODE 2 IITJ REAL END-TO-END GATE")
    print(
        f"cases={len(results)} | "
        f"PASS={sum(r['status'] == 'PASS' for r in results)} | "
        f"REVIEW={sum(r['status'] == 'REVIEW' for r in results)} | "
        f"FAIL={len(failures)}"
    )
    print(f"JSON_REPORT={REPORT_JSON}")
    print(f"SUMMARY_REPORT={REPORT_SUMMARY}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())