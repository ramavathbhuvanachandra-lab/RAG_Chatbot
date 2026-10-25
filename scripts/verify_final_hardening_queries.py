#!/usr/bin/env python3
"""Run Phase 1/4/5 verification queries directly through handle_chat_turn."""

import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

QUERIES = [
    # 1. Campus navigation & food queries
    ("veg food kaha milega?", "Veg Food (Hindi/Hinglish)"),
    ("Need milk and snacks. Where should I go?", "Milk & Snacks (Navigation)"),
    ("where's the old mess?", "Old Mess Navigation"),
    ("where can i buy milk", "Milk Shopping"),
    ("best food spots", "Food Spots (Food Court)"),
    
    # 2. Short fee queries & broad institutional schedule
    ("fees?", "Short fee inquiry"),
    ("fee structure", "Fee structure inquiry"),
    
    # 3. Academic & admission queries
    ("M.Tech eligibility?", "M.Tech Eligibility"),
    
    # 4. Partial data & missing data queries
    ("mess timing?", "Mess timing (Missing time gap)"),
    
    # 5. Multi-intent
    ("tell me both hostel and dining details", "Multi-intent hostel and dining"),
]

def main():
    print("=" * 80)
    print("RUNNING FINAL SYSTEMIC HARDENING PASS EVALUATION")
    print("=" * 80)
    
    session_id = f"hardening_eval_{int(time.time())}"
    history = []
    
    for query, label in QUERIES:
        print(f"\n--- [{label}] Query: '{query}' ---")
        t0 = time.perf_counter()
        req = ChatTurnRequest(
            question=query,
            chat_history=tuple(history),
            session_id=session_id,
            institution_id="iitj",
        )
        resp = handle_chat_turn(req)
        elapsed = time.perf_counter() - t0
        
        diag = resp.diagnostics or {}
        timings = diag.get("timings", {})
        turn_id = diag.get("turn_id", "N/A")
        terminal_status = diag.get("terminal_status", "N/A")
        recon_delta = timings.get("reconciliation_delta_ms", 0.0)
        
        print(f"Turn ID: {turn_id} | Status: {terminal_status} | Latency: {elapsed:.2f}s | Recon Delta: {recon_delta:.2f}ms")
        print(f"Answer snippet: {resp.answer[:180]}...")
        print(f"Sources count: {len(resp.sources)}")
        for i, src in enumerate(resp.sources[:3], 1):
            title = src.get("heading") or src.get("source") or "Unknown"
            print(f"  Source [{i}]: {title}")
            
        history.append((query, resp.answer))
        
    print("\n" + "=" * 80)
    print("EVALUATION FINISHED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
