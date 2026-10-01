"""
Deep real-RAG diagnostic for the current reusable IITJ core.

Run from repository root:
    PYTHONPATH=. python tests_reusable/final_rag_core/deep_rag_diagnostic.py

This script DOES NOT modify production code.
It traces one real pipeline pass across:
    understanding
    -> Dense / BM25 / lexical recall
    -> RRF
    -> verification
    -> ranking
    -> evidence grouping/scope
    -> coverage
    -> claims
    -> evidence package
    -> answer model
    -> grounding
    -> final guard

The output is diagnostic, not a quality score.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

from backend.core.nodes import CoreNodes, default_dependencies
from backend.core.query.understanding import understand_query_frame


QUESTIONS = (
    "What are the M.Tech eligibility requirements?",
    "What is the tuition fee?",
    "How do I apply for admission?",
    "What are the research areas in Electrical Engineering?",
    "How do I reach the library?",
)

REPORT_DIR = Path("tests_reusable/final_rag_core/reports")
REPORT_PATH = REPORT_DIR / "deep_rag_diagnostic.json"


class TracingAnswerModel:
    def __init__(self, real_model: Any) -> None:
        self.real_model = real_model
        self.calls = 0
        self.payloads: list[Any] = []
        self.outputs: list[str] = []

    def invoke(self, payload: Any) -> Any:
        self.calls += 1
        self.payloads.append(payload)
        response = self.real_model.invoke(payload)
        text = getattr(response, "content", response)
        self.outputs.append(str(text or "").strip())
        return response


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _text(document: Any) -> str:
    return _clean(getattr(document, "page_content", ""))


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


def _preview(document: Any, n: int = 320) -> str:
    return _text(document)[:n]


def _status(value: Any, default: str = "") -> str:
    if value is None:
        return default
    raw = value.get("status") if isinstance(value, Mapping) else getattr(value, "status", None)
    return _clean(raw).casefold() or default


def _answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("answer", "text", "content"):
            if key in value:
                return _clean(value[key])
    for attr in ("answer", "content", "text"):
        raw = getattr(value, attr, None)
        if raw is not None:
            return _clean(raw)
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


def _query_dump(query: Any) -> dict[str, Any]:
    if query is None:
        return {}
    if hasattr(query, "to_dict") and callable(query.to_dict):
        try:
            return _serializable(query.to_dict())
        except Exception:
            pass
    if hasattr(query, "model_dump") and callable(query.model_dump):
        try:
            return _serializable(query.model_dump())
        except Exception:
            pass
    if is_dataclass(query):
        return _serializable(asdict(query))
    return {
        "request_type": _clean(getattr(query, "request_type", "")),
        "search_query": _clean(getattr(query, "search_query", "")),
        "target": _clean(getattr(getattr(query, "target", None), "text", "")),
        "confidence": float(getattr(query, "confidence", 0.0) or 0.0),
        "entities": _serializable(getattr(query, "entities", ())),
        "concepts": _serializable(getattr(query, "concepts", ())),
        "qualifiers": _serializable(getattr(query, "qualifiers", ())),
        "constraints": _serializable(getattr(query, "constraints", ())),
        "numeric_requirements": _serializable(getattr(query, "numeric_requirements", ())),
        "temporal_constraints": _serializable(getattr(query, "temporal_constraints", ())),
        "list_intent": _serializable(getattr(query, "list_intent", None)),
    }


def _print_channel(title: str, docs: list[Any] | tuple[Any, ...]) -> None:
    print(f"\n{title} ({len(docs)} hits)")
    for i, doc in enumerate(docs[:5], start=1):
        print(f"  {i:02d}. { _source(doc) }")
        print(f"      {_preview(doc)}")


def _print_fused(candidates: list[Any] | tuple[Any, ...]) -> None:
    print(f"\nRRF FUSED ({len(candidates)} unique candidates)")
    for i, candidate in enumerate(candidates[:12], start=1):
        provenance = getattr(candidate, "provenance", None)
        rrf = getattr(provenance, "rrf_score", 0.0) if provenance else 0.0
        print(f"  {i:02d}. rrf={rrf:.6f} | {_source(candidate.document)}")
        print(f"      {_preview(candidate.document)}")


def _print_verification(
    fused_candidates: list[Any] | tuple[Any, ...],
    verification_trace: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> None:
    decision_by_id = {
        str(row.get("candidate_id")): row
        for row in verification_trace
        if isinstance(row, Mapping)
    }

    print("\nVERIFICATION TOP RESULTS")
    for i, candidate in enumerate(fused_candidates[:20], start=1):
        row = decision_by_id.get(str(getattr(candidate, "document_id", "")), {})
        print(
            f"  {i:02d}. status={row.get('status')} "
            f"accepted={row.get('accepted')} "
            f"score={row.get('score')} "
            f"target={row.get('target_grounded')} "
            f"attr={row.get('attribute_grounded')} "
            f"scope={row.get('scope_compatible')} "
            f"semantic={row.get('semantic_compatible')} "
            f"conflict={row.get('conflict_detected')}"
        )
        print(f"      {_source(candidate.document)}")
        print(f"      reasons={row.get('reasons')}")


def _print_ranked(candidates: list[Any] | tuple[Any, ...]) -> None:
    print(f"\nFINAL RANKING ({len(candidates)} candidates)")
    for i, candidate in enumerate(candidates[:12], start=1):
        alignment = getattr(candidate, "alignment", None)
        print(
            f"  {i:02d}. score={getattr(candidate, 'final_score', 0.0):.6f} "
            f"program={getattr(alignment, 'program_match', 0.0):.2f} "
            f"entity={getattr(alignment, 'entity_match', 0.0):.2f} "
            f"topic={getattr(alignment, 'topic_match', 0.0):.2f} "
            f"attr={getattr(alignment, 'attribute_match', 0.0):.2f} "
            f"semantic={getattr(alignment, 'semantic_match', 0.0):.2f}"
        )
        print(f"      {_source(candidate.document)}")
        print(f"      {_preview(candidate.document)}")


def run_case(nodes: CoreNodes, tracer: TracingAnswerModel, question: str) -> dict[str, Any]:
    print("\n" + "=" * 110)
    print(f"QUESTION: {question}")
    started = time.perf_counter()

    state: dict[str, Any] = {
        "question": question,
        "resolved_question": question,
        "chat_history": (),
    }

    query_state = nodes.understand_query_node(state)
    state.update(query_state)
    query = state.get("query")
    print("\nQUERY UNDERSTANDING")
    print(json.dumps(_query_dump(query), indent=2, ensure_ascii=False))

    retrieval_state = nodes.hybrid_retrieve_node(state)
    state.update(retrieval_state)
    retrieval_results = tuple(state.get("retrieval_results", ()))
    queries = tuple(state.get("retrieval_queries", ()))

    print("\nRETRIEVAL QUERIES")
    for i, q in enumerate(queries, start=1):
        print(f"  {i}. {q}")

    per_query = []
    for i, q in enumerate(queries):
        dense = list(retrieval_results[i * 3]) if i * 3 < len(retrieval_results) else []
        bm25 = list(retrieval_results[i * 3 + 1]) if i * 3 + 1 < len(retrieval_results) else []
        lexical = list(retrieval_results[i * 3 + 2]) if i * 3 + 2 < len(retrieval_results) else []
        _print_channel(f"QUERY {i+1} DENSE: {q}", dense)
        _print_channel(f"QUERY {i+1} BM25: {q}", bm25)
        _print_channel(f"QUERY {i+1} LEXICAL: {q}", lexical)
        per_query.append(
            {
                "query": q,
                "dense": [{"source": _source(d), "preview": _preview(d)} for d in dense[:5]],
                "bm25": [{"source": _source(d), "preview": _preview(d)} for d in bm25[:5]],
                "lexical": [{"source": _source(d), "preview": _preview(d)} for d in lexical[:5]],
            }
        )

    fused_state = nodes.fuse_retrieved_documents_node(state)
    state.update(fused_state)
    fused_candidates = tuple(state.get("fused_candidates", ()))
    _print_fused(fused_candidates)

    verified_state = nodes.verify_and_rank_node(state)
    state.update(verified_state)
    verification_trace = tuple(state.get("verification_trace", ()))
    _print_verification(fused_candidates, verification_trace)

    ranked_candidates = tuple(state.get("ranked_candidates", ()))
    _print_ranked(ranked_candidates)

    evidence_state = nodes.evidence_context_node(state)
    state.update(evidence_state)

    evidence_docs = tuple(state.get("evidence_documents", ()))
    evidence_candidates = tuple(state.get("evidence_candidates", ()))
    print("\nEVIDENCE CONTEXT")
    for i, doc in enumerate(evidence_docs[:12], start=1):
        print(f"  {i:02d}. {_source(doc)}")
        print(f"      {_preview(doc, 500)}")

    assessment = nodes.assess_evidence_node(state)
    state.update(assessment)
    coverage = nodes.assess_coverage_node(state)
    state.update(coverage)

    print("\nEVIDENCE / COVERAGE")
    print(f"  evidence_status   = {_status(state.get('evidence_assessment'))}")
    print(f"  evidence_score    = {getattr(state.get('evidence_assessment'), 'score', None)}")
    print(f"  coverage_status   = {_status(state.get('coverage_assessment'))}")
    print(f"  coverage_score    = {getattr(state.get('coverage_assessment'), 'score', None)}")
    print(f"  coverage_type     = {state.get('evidence_question_type')}")

    claims_state = nodes.audit_claims_node(state)
    state.update(claims_state)
    claim_audit = state.get("claim_audit")
    print("\nCLAIMS")
    print(f"  claims={_serializable(state.get('claims', ())) }")
    print(f"  audit_status={_status(claim_audit)}")
    print(f"  supported={_serializable(getattr(claim_audit, 'supported_claim_ids', ())) }")
    print(f"  partial={_serializable(getattr(claim_audit, 'partial_claim_ids', ())) }")
    print(f"  unsupported={_serializable(getattr(claim_audit, 'unsupported_claim_ids', ())) }")
    print(f"  conflicting={_serializable(getattr(claim_audit, 'conflicting_claim_ids', ())) }")

    package_state = nodes.package_evidence_node(state)
    state.update(package_state)
    package = state.get("evidence_package")

    print("\nEVIDENCE PACKAGE")
    print(f"  status={_status(package)}")
    print(f"  ready_for_generation={bool(getattr(package, 'ready_for_generation', False))}")
    print(f"  source_count={getattr(package, 'source_count', 0)}")
    print(f"  selected_count={getattr(package, 'selected_count', 0)}")
    print("  ITEMS:")
    for i, item in enumerate(getattr(package, "items", ()) or (), start=1):
        print(f"    {i:02d}. source={getattr(item, 'source', '')}")
        print(f"        {getattr(item, 'text', '')[:700]}")
    print("\n  EXACT CONTEXT:")
    print(getattr(package, "context", ""))

    answer_state = dict(state)
    tracer_before = tracer.calls
    answer_result = nodes.answer_node(answer_state)
    tracer_calls = tracer.calls - tracer_before

    generation = answer_result.get("answer_generation")
    grounding = answer_result.get("answer_grounding")
    guard = answer_result.get("answer_guard")

    print("\nANSWER")
    print(f"  mode={answer_result.get('answer_mode')}")
    print(f"  model_calls={answer_result.get('model_calls')}")
    print(f"  actual_tracer_calls={tracer_calls}")
    print(f"  generation_reason={getattr(generation, 'reason', None)}")
    print(f"  RAW MODEL ANSWER:\n{_answer(generation)}")
    print(f"  GROUNDING={_status(grounding, 'not_run')}")
    print(f"  GUARD={_status(guard)}")
    print(f"  FINAL ANSWER:\n{_answer(answer_result.get('answer'))}")

    elapsed = time.perf_counter() - started

    return {
        "question": question,
        "query": _query_dump(query),
        "retrieval": {
            "queries": list(queries),
            "per_query": per_query,
            "dense_total": sum(
                len(retrieval_results[i * 3])
                for i in range(len(queries))
                if i * 3 < len(retrieval_results)
            ),
            "bm25_total": sum(
                len(retrieval_results[i * 3 + 1])
                for i in range(len(queries))
                if i * 3 + 1 < len(retrieval_results)
            ),
            "lexical_total": int(state.get("lexical_recall_count", 0) or 0),
            "fused_count": len(fused_candidates),
            "verification_trace": _serializable(verification_trace),
            "verified_count": len(state.get("verified_candidates", ())),
            "ranked": _serializable(ranked_candidates),
            "lexical_fallback_used": bool(state.get("lexical_fallback_used", False)),
            "lexical_fallback_trace": _serializable(state.get("lexical_fallback_trace", ())),
        },
        "evidence": {
            "documents": [{"source": _source(d), "preview": _preview(d, 700)} for d in evidence_docs[:12]],
            "candidate_count": len(evidence_candidates),
            "status": _status(state.get("evidence_assessment")),
            "coverage_status": _status(state.get("coverage_assessment")),
            "coverage": _serializable(state.get("coverage_assessment")),
            "claims": _serializable(state.get("claims", ())),
            "claim_audit": _serializable(claim_audit),
            "package": _serializable(package),
        },
        "answer": {
            "generation": _serializable(generation),
            "grounding": _serializable(grounding),
            "guard": _serializable(guard),
            "final_answer": _answer(answer_result.get("answer")),
            "mode": answer_result.get("answer_mode"),
            "model_calls": int(answer_result.get("model_calls", 0) or 0),
            "tracer_calls": tracer_calls,
        },
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    from backend.runtime.llm import answer_llm

    deps = default_dependencies()
    tracer = TracingAnswerModel(answer_llm)
    deps = replace(deps, answer_model_provider=lambda _state: tracer)
    nodes = CoreNodes(deps)

    results = []
    for question in QUESTIONS:
        try:
            results.append(run_case(nodes, tracer, question))
        except Exception as exc:
            results.append(
                {
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"\nERROR for {question}: {type(exc).__name__}: {exc}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"questions": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("DEEP RAG DIAGNOSTIC COMPLETE")
    print(f"REPORT={REPORT_PATH}")
    return 0 if not any("error" in row for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())