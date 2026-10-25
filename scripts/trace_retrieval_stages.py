"""Diagnostic tracer for Phase 1 of RAG Reliability Audit.

Traces the live canonical pipeline across all stages for:
1. Electrical Engineering research areas.
2. M.Tech tuition fee for regular Indian students.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest
from ai_platform.core.graph.nodes import CoreNodes, default_dependencies


def trace_query(question: str) -> dict[str, Any]:
    nodes = CoreNodes(default_dependencies())
    state: dict[str, Any] = {
        "question": question,
        "chat_history": (),
    }

    # 1. Conversation resolution
    state.update(nodes.resolve_conversation_node(state))
    resolved_q = state.get("resolved_question")

    # 2. Query understanding
    state.update(nodes.understand_query_node(state))
    plan = state.get("query_plan")
    frame = state.get("query_frame")
    retrieval_queries = state.get("retrieval_queries")

    # 3. Hybrid retrieval
    state.update(nodes.hybrid_retrieve_node(state))
    raw_results = state.get("retrieval_results", ())
    weights = state.get("retrieval_weights", ())

    # 4. Fusion
    state.update(nodes.fuse_retrieved_documents_node(state))
    fused = state.get("fused_candidates", ())

    # 5. Verification
    state.update(nodes.verify_and_rank_node(state))
    verified = state.get("verified_candidates", ())
    uncertain = state.get("uncertain_candidates", ())
    rejected = state.get("rejected_candidates", ())
    ranked = state.get("ranked_candidates", ())
    v_trace = state.get("verification_trace", ())

    # 6. Evidence boundary & packaging
    state.update(nodes.evidence_boundary_node(state))
    state.update(nodes.assess_evidence_node(state))
    state.update(nodes.assess_coverage_node(state))
    state.update(nodes.audit_claims_node(state))
    state.update(nodes.package_evidence_node(state))
    package = state.get("evidence_package")
    final_context = state.get("final_evidence_context", "")

    # 7. Answer generation & UI citations via assistant service
    service_resp = handle_chat_turn(ChatTurnRequest(question=question))

    # Compile structured stage trace
    fused_top15 = []
    for idx, cand in enumerate(fused[:15], start=1):
        doc = getattr(cand, "document", cand)
        prov = getattr(cand, "provenance", None)
        fused_top15.append({
            "fused_rank": idx,
            "document_id": getattr(cand, "document_id", ""),
            "source": getattr(cand, "source", "") or (getattr(doc, "metadata", {}).get("source", "") if hasattr(doc, "metadata") else ""),
            "rrf_score": getattr(prov, "rrf_score", 0.0) if prov else 0.0,
            "dense_rank": getattr(prov, "dense_rank", None) if prov else None,
            "bm25_rank": getattr(prov, "bm25_rank", None) if prov else None,
            "text_snippet": (getattr(doc, "page_content", "") or "")[:200].strip(),
        })

    v_decisions = {}
    for d in v_trace:
        if isinstance(d, dict):
            cid = d.get("candidate_id")
            v_decisions[cid] = {
                "status": d.get("status"),
                "accepted": d.get("accepted"),
                "reasons": d.get("reasons", []),
                "coverage": d.get("coverage"),
                "score": d.get("score"),
            }

    packaged_units = []
    if package is not None and hasattr(package, "items"):
        for item in package.items:
            packaged_units.append({
                "evidence_id": getattr(item, "evidence_id", ""),
                "source": getattr(item, "source", ""),
                "heading": getattr(item, "heading", ""),
                "text_snippet": (getattr(item, "text", "") or "")[:200].strip(),
            })

    return {
        "question": question,
        "resolved_question": resolved_q,
        "retrieval_queries": retrieval_queries,
        "target": getattr(frame, "target", None),
        "request_type": getattr(frame, "request_type", None),
        "raw_retrieval_lists_count": len(raw_results),
        "fused_candidates_count": len(fused),
        "fused_top15": fused_top15,
        "verified_count": len(verified),
        "uncertain_count": len(uncertain),
        "rejected_count": len(rejected),
        "ranked_count": len(ranked),
        "verification_decisions": v_decisions,
        "packaged_units_count": len(packaged_units),
        "packaged_units": packaged_units,
        "context_character_count": len(final_context),
        "context_estimated_tokens": len(final_context) // 4,
        "answer": service_resp.answer,
        "displayed_sources": service_resp.sources,
        "latency_seconds": service_resp.latency_seconds,
        "is_safe_refusal": service_resp.is_safe_refusal,
        "evidence_status": service_resp.evidence_status,
    }


def main():
    out_dir = Path("test_outputs/retrieval_reliability_diagnostics_20261018")
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = [
        "What research areas are available in Electrical Engineering?",
        "What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. programme at IIT Jodhpur?",
    ]

    results = []
    for q in queries:
        print(f"Tracing: {q}")
        t = trace_query(q)
        results.append(t)

    out_file = out_dir / "stage_traces.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved traces to {out_file}")


if __name__ == "__main__":
    main()
