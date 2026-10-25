#!/usr/bin/env python3
"""Diagnose the 6 Master Task Failure Cases."""

import sys
import time
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

TESTS = [
    {
        "case": "Case A: Complete B.Tech programme list",
        "query": "What B.Tech programmes does IIT Jodhpur offer?",
    },
    {
        "case": "Case B: Institute-wide departments",
        "query": "What departments does IIT Jodhpur have?",
    },
    {
        "case": "Case C1: Tuition fee per semester",
        "query": "What is the tuition fee per semester?",
    },
    {
        "case": "Case C2: Fee structure for B.Tech",
        "query": "What is the fee structure for B.Tech?",
    },
    {
        "case": "Case D: Main research areas",
        "query": "What are the main research areas at IIT Jodhpur?",
    },
    {
        "case": "Case E: Navigation vs RAG telemetry",
        "query": "Where can I buy milk and snacks on campus?",
    },
    {
        "case": "Case F: Follow-up continuity",
        "multi_turn": [
            "Tell me about the Drone and Anti-Drone Technologies programme.",
            "Was that programme proposed for a particular academic year?",
        ]
    }
]

def main():
    print("=" * 80)
    print("RUNNING DIAGNOSIS ON MASTER TASK FAILURE CASES")
    print("=" * 80)
    
    results = []
    
    for idx, test in enumerate(TESTS, 1):
        case_name = test["case"]
        print(f"\n--- [{idx}/{len(TESTS)}] {case_name} ---")
        
        session_id = f"diag_{idx}_{int(time.time())}"
        history = []
        
        if "multi_turn" in test:
            ans = ""
            resp = None
            for turn_q in test["multi_turn"]:
                req = ChatTurnRequest(
                    question=turn_q,
                    chat_history=tuple(history),
                    session_id=session_id,
                    institution_id="iitj",
                )
                resp = handle_chat_turn(req)
                ans = resp.answer
                history.append((turn_q, resp.answer))
                print(f"Turn query: {turn_q}")
                print(f"Turn answer: {ans}\n")
        else:
            q = test["query"]
            print(f"Query: {q}")
            req = ChatTurnRequest(
                question=q,
                chat_history=(),
                session_id=session_id,
                institution_id="iitj",
            )
            resp = handle_chat_turn(req)
            ans = resp.answer
            
        diag = resp.diagnostics or {}
        print(f"Resp.answer: {ans[:300]}...")
        print(f"Resp.formatted_response: {resp.formatted_response[:300]}...")
        print(f"Sources ({len(resp.sources)}): {[s.get('source') for s in resp.sources]}")
        print(f"Diagnostics keys: {list(diag.keys())}")
        if "route_selected" in diag:
            print(f"Route selected: {diag.get('route_selected')}")
        if "execution_route" in diag:
            print(f"Execution route: {diag.get('execution_route')}")
        if "terminal_status" in diag:
            print(f"Terminal status: {diag.get('terminal_status')}")
            
        results.append({
            "case": case_name,
            "answer": ans,
            "formatted_response": resp.formatted_response,
            "sources": resp.sources,
            "diagnostics_summary": {
                "route_selected": diag.get("route_selected"),
                "terminal_status": diag.get("terminal_status"),
                "total_candidates": len(diag.get("section_a_candidates", [])),
                "verified_count": diag.get("funnel", {}).get("verified_count", 0),
            }
        })
        
    out_file = WORKSPACE_ROOT / "test_outputs" / "master_cases_diagnosis.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved diagnosis to {out_file}")

if __name__ == "__main__":
    main()
