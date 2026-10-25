#!/usr/bin/env python3
"""Diagnose Phase 1 failure paths: tuition fee and academic list queries."""

import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

QUERIES = [
    ("what is the tuition fee?", "Tuition Fee Inquiry"),
    ("what are the B.Tech programmes offered?", "B.Tech Programmes List"),
]

def diagnose_query(query: str, label: str):
    print("=" * 80)
    print(f"DIAGNOSING: [{label}] '{query}'")
    print("=" * 80)
    
    t0 = time.perf_counter()
    req = ChatTurnRequest(
        question=query,
        chat_history=(),
        session_id=f"diag_{int(time.time())}",
        institution_id="iitj",
    )
    resp = handle_chat_turn(req)
    total_elapsed = time.perf_counter() - t0
    
    diag = resp.diagnostics or {}
    raw_state = diag.get("raw_state") or diag
    
    print(f"Total Latency: {total_elapsed:.2f}s")
    print(f"Terminal Status: {diag.get('terminal_status')}")
    print(f"Resolved Question: {diag.get('resolved_question') or raw_state.get('resolved_question')}")
    print(f"Query Understanding Plan/Intent:")
    print(f"  Plan: {raw_state.get('plan')}")
    print(f"  Retrieval queries: {raw_state.get('retrieval_queries') or diag.get('retrieval_queries')}")
    
    dense_records = raw_state.get("dense_retrieval_records") or diag.get("dense_retrieval_records") or ()
    bm25_records = raw_state.get("bm25_retrieval_records") or diag.get("bm25_retrieval_records") or ()
    fused_candidates = raw_state.get("fused_candidates") or diag.get("fused_candidates") or ()
    verified = raw_state.get("verified_candidates") or diag.get("verified_candidates") or ()
    rejected = raw_state.get("rejected_candidates") or diag.get("rejected_candidates") or ()
    v_trace = raw_state.get("verification_trace") or diag.get("verification_trace") or ()
    
    print(f"\n--- RETRIEVAL CANDIDATES ---")
    print(f"Dense hits count: {len(dense_records)}")
    for i, r in enumerate(dense_records[:5]):
        src = r.get("source", "") if isinstance(r, dict) else getattr(r, "source", "")
        sc = r.get("score", "") if isinstance(r, dict) else getattr(r, "score", "")
        txt = r.get("text", "")[:100] if isinstance(r, dict) else getattr(r, "text", "")[:100]
        print(f"  Dense [{i}] score={sc} src={src}\n    preview: {txt}...")
        
    print(f"BM25 hits count: {len(bm25_records)}")
    for i, r in enumerate(bm25_records[:5]):
        src = r.get("source", "") if isinstance(r, dict) else getattr(r, "source", "")
        sc = r.get("score", "") if isinstance(r, dict) else getattr(r, "score", "")
        txt = r.get("text", "")[:100] if isinstance(r, dict) else getattr(r, "text", "")[:100]
        print(f"  BM25 [{i}] score={sc} src={src}\n    preview: {txt}...")
        
    print(f"\n--- FUSION & VERIFICATION ---")
    print(f"Fused count: {len(fused_candidates)}")
    print(f"Verified count: {len(verified)}")
    print(f"Rejected count: {len(rejected)}")
    print(f"Verification Trace (first 5):")
    for vt in v_trace[:5]:
        print(f"  {vt}")
        
    print(f"\n--- FINAL CONTEXT ---")
    ctx = str(raw_state.get("final_evidence_context") or diag.get("final_evidence_context") or "")
    print(f"Context Length (chars): {len(ctx)}")
    print("Context Snippet (first 600 chars):")
    print(ctx[:600])
    
    print(f"\n--- FINAL ANSWER ---")
    print(resp.answer)
    
    print(f"\n--- CITATIONS ({len(resp.sources)}) ---")
    for i, s in enumerate(resp.sources):
        print(f"  [{i+1}] {s.get('source')} | heading: {s.get('heading')}")

def main():
    for q, label in QUERIES:
        diagnose_query(q, label)

if __name__ == "__main__":
    main()
