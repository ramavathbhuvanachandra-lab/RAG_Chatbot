"""Phase 1 Deep Forensic Trace for Electrical Engineering Query."""

import time
import json
from pathlib import Path
from ai_platform.runtime.engine import get_inspect_nodes, get_app_graph, ask_with_diagnostics
from ai_platform.service.assistant_service import handle_chat_turn

def run_phase1_trace():
    q = "What research areas are available in Electrical Engineering?"
    print(f"=== PHASE 1 TRACE FOR: '{q}' ===")

    start_total = time.perf_counter()
    nodes = get_inspect_nodes()
    state = {
        "question": q,
        "chat_history": (),
    }

    timings = {}

    # Stage 1: Resolve conversation
    t0 = time.perf_counter()
    state.update(nodes.resolve_conversation_node(state))
    timings["resolve_conversation"] = time.perf_counter() - t0

    # Stage 2: Understand query
    t0 = time.perf_counter()
    state.update(nodes.understand_query_node(state))
    timings["understand_query"] = time.perf_counter() - t0

    # Stage 3: Multi-intent plan
    t0 = time.perf_counter()
    state.update(nodes.plan_multi_intent_node(state))
    timings["plan_multi_intent"] = time.perf_counter() - t0

    # Stage 4: Retrieve and qualify
    t0 = time.perf_counter()
    state.update(nodes.retrieve_and_qualify_node(state))
    timings["retrieve_and_qualify"] = time.perf_counter() - t0

    # Stage 5: Answer generation
    t0 = time.perf_counter()
    state.update(nodes.answer_node(state))
    timings["answer_node"] = time.perf_counter() - t0

    timings["total_pipeline"] = time.perf_counter() - start_total

    # Also run through handle_chat_turn to see what frontend receives
    t0 = time.perf_counter()
    chat_resp = handle_chat_turn(q, session_id="trace_phase1_ee", institution_id="iitj")
    timings["handle_chat_turn_call"] = time.perf_counter() - t0

    resolved_q = state.get("resolved_question") or q
    raw_retrieval_results = state.get("retrieval_results") or ()
    fused = state.get("fused_candidates") or ()
    verified = state.get("verified_candidates") or ()
    ranked = state.get("ranked_candidates") or ()
    package = state.get("evidence_package")
    evidence_context = getattr(package, "context", "") or ""
    answer = state.get("answer", "")
    sources = chat_resp.sources

    # Check for presence of 'knowledge_buffer_general' or 'Fallback' across stages
    presence = {
        "raw_retrieval": False,
        "fused_candidates": False,
        "verified_candidates": False,
        "ranked_candidates": False,
        "final_package_items": False,
        "final_evidence_context": False,
        "answer_text": False,
        "frontend_sources": False,
    }

    # Analyze raw retrieval
    raw_list = []
    for query_idx, res_list in enumerate(raw_retrieval_results):
        for rank, cand in enumerate(res_list):
            src = getattr(cand, "source", None) or getattr(getattr(cand, "document", None), "source", "")
            doc_id = getattr(cand, "document_id", "")
            title = getattr(cand, "heading", "")
            score = float(getattr(cand, "score", 0.0) or 0.0)
            meta = getattr(cand, "metadata", {}) or {}
            if "knowledge_buffer_general" in str(src):
                presence["raw_retrieval"] = True
            raw_list.append({
                "query_idx": query_idx,
                "rank": rank,
                "doc_id": doc_id,
                "source": src,
                "heading": title,
                "score": score,
                "metadata": meta,
            })

    # Analyze fused
    fused_list = []
    for rank, cand in enumerate(fused):
        src = getattr(cand, "source", None) or getattr(getattr(cand, "document", None), "source", "")
        if "knowledge_buffer_general" in str(src):
            presence["fused_candidates"] = True
        fused_list.append({
            "rank": rank,
            "source": src,
            "heading": getattr(cand, "heading", ""),
            "score": float(getattr(cand, "score", 0.0) or 0.0),
        })

    # Analyze verified
    verified_list = []
    for rank, cand in enumerate(verified):
        src = getattr(cand, "source", None) or getattr(getattr(cand, "document", None), "source", "")
        if "knowledge_buffer_general" in str(src):
            presence["verified_candidates"] = True
        verified_list.append({
            "rank": rank,
            "source": src,
            "heading": getattr(cand, "heading", ""),
            "score": float(getattr(cand, "score", 0.0) or 0.0),
        })

    # Analyze ranked
    ranked_list = []
    for rank, cand in enumerate(ranked):
        src = getattr(cand, "source", None) or getattr(getattr(cand, "document", None), "source", "")
        if "knowledge_buffer_general" in str(src):
            presence["ranked_candidates"] = True
        ranked_list.append({
            "rank": rank,
            "source": src,
            "heading": getattr(cand, "heading", ""),
            "score": float(getattr(cand, "score", 0.0) or 0.0),
        })

    # Analyze package items
    package_items = []
    if package is not None and hasattr(package, "items"):
        for item in package.items:
            src = getattr(item, "source", "")
            if "knowledge_buffer_general" in str(src):
                presence["final_package_items"] = True
            package_items.append({
                "evidence_id": getattr(item, "evidence_id", ""),
                "source": src,
                "heading": getattr(item, "heading", ""),
                "text_excerpt": getattr(item, "text", "")[:200],
            })

    if "knowledge_buffer_general" in evidence_context or "Technical English" in evidence_context:
        presence["final_evidence_context"] = True

    if "Technical English" in answer or "CFD" in answer or "hospital" in answer.casefold():
        presence["answer_text"] = True

    for s in sources:
        if "knowledge_buffer_general" in str(s.get("source", "")) or "Fallback" in str(s.get("title", "")):
            presence["frontend_sources"] = True

    report = {
        "original_query": q,
        "resolved_query": resolved_q,
        "presence_of_knowledge_buffer_general": presence,
        "timings": timings,
        "raw_candidates_count": len(raw_list),
        "fused_count": len(fused_list),
        "verified_count": len(verified_list),
        "ranked_count": len(ranked_list),
        "package_items_count": len(package_items),
        "frontend_sources_count": len(sources),
        "generator_answer": answer,
        "chat_turn_answer": chat_resp.answer,
        "sample_fused": fused_list[:10],
        "sample_verified": verified_list[:10],
        "sample_ranked": ranked_list[:10],
        "package_items": package_items[:10],
        "frontend_sources": sources[:10],
    }

    out_path = Path("test_outputs/phase1_ee_trace.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote trace to {out_path}")
    print("\n--- PRESENCE OF FALLBACK BUFFER ACROSS STAGES ---")
    for k, v in presence.items():
        print(f"  {k}: {v}")
    print("\n--- TIMINGS ---")
    for k, v in timings.items():
        print(f"  {k}: {v:.2f}s")
    print(f"\nFrontend Sources Count: {len(sources)}")
    print(f"Generator Answer Preview:\n{answer[:300]}")

if __name__ == "__main__":
    run_phase1_trace()
