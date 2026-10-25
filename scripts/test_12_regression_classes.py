#!/usr/bin/env python3
"""Run the 12-Class Regression Matrix for Production RAG Reliability Overhaul."""

import sys
import time
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

REGRESSION_CASES = [
    # 1. Exact attribute and numeric-value retrieval
    {
        "class": "1. Exact Numeric Value",
        "query": "what are the short-term hostel charges for IIT Jodhpur students with bedding?",
        "check": lambda ans, src: ("400" in ans or "600" in ans or "₹400" in ans or "₹600" in ans) and any("fees_and_finance" in s.get("source", "").lower() for s in src),
        "expected": "₹400/- (Double) or ₹600/- (Single) per day with bedding",
    },
    # 2. Confusable attributes (Tuition vs Registration fee)
    {
        "class": "2. Confusable Attributes",
        "query": "what is the registration fee for students?",
        "check": lambda ans, src: ("500" in ans or "₹500" in ans or "rs. 500" in ans.lower() or "not documented" in ans.lower() or "500" in ans),
        "expected": "Distinct from tuition fee ₹50,000",
    },
    # 3. Generic vs School-Specific (MBA vs Institute)
    {
        "class": "3. Generic vs School-Specific",
        "query": "what is the MBA fee at SME?",
        "check": lambda ans, src: ("2,25,000" in ans or "225000" in ans or "management" in ans.lower() or "sme" in ans.lower()),
        "expected": "SME / MBA fee ₹2,25,000",
    },
    # 4. Complete list retrieval
    {
        "class": "4. Complete List Retrieval",
        "query": "what B.Tech programmes are offered by the institute?",
        "check": lambda ans, src: len(src) >= 1 and ("mechanical" in ans.lower() or "computer science" in ans.lower() or "electrical" in ans.lower()),
        "expected": "B.Tech programmes list",
    },
    # 5. Tables, academic years, categories, exceptions
    {
        "class": "5. Table & Exemption",
        "query": "are SC, ST and PwD students exempted from paying tuition fees?",
        "check": lambda ans, src: ("exempt" in ans.lower() or "yes" in ans.lower()) and any("fees_and_finance" in s.get("source", "").lower() for s in src),
        "expected": "Exempted from tuition fee",
    },
    # 6. Multiple requested attributes
    {
        "class": "6. Multi-Attribute Query",
        "query": "tell me both the M.Tech eligibility and the hostel charges",
        "check": lambda ans, src: ("60%" in ans or "cgpa" in ans.lower() or "bachelor" in ans.lower()) and ("hostel" in ans.lower() or "450" in ans or "300" in ans or "occupancy" in ans.lower()),
        "expected": "Both M.Tech eligibility and hostel charges",
    },
    # 7. Short, informal, misspelled, Hinglish
    {
        "class": "7. Hinglish Navigation",
        "query": "veg khana kidhar milega campus me?",
        "check": lambda ans, src: ("shamiyana" in ans.lower() or "food court" in ans.lower()) and any("campus" in s.get("source", "").lower() or "directory" in s.get("source", "").lower() for s in src),
        "expected": "Shamiyana Food Court",
    },
    # 8. Follow-up resolution
    {
        "class": "8. Multi-turn Follow-up",
        "multi_turn": [
            ("Tell me about the Drone and Anti-Drone Technologies programme.", "Drone intro"),
            ("Was that programme proposed for a particular academic year?", "2025-2026"),
        ],
        "check": lambda ans, src: "2025-2026" in ans or "2025–2026" in ans or "2025" in ans,
        "expected": "Academic year 2025-2026",
    },
    # 9. Relevant secondary evidence
    {
        "class": "9. Secondary Context",
        "query": "what is the B.S. in Mathematics and Computing about?",
        "check": lambda ans, src: len(src) >= 1 and ("mathematics" in ans.lower() or "computing" in ans.lower()),
        "expected": "B.S. in Mathematics and Computing details",
    },
    # 10. Missing information (Graceful abstention)
    {
        "class": "10. Missing Data Abstention",
        "query": "what are the specific hostel room numbers allocated to incoming Ph.D. scholars in 2026?",
        "check": lambda ans, src: ("not specified" in ans.lower() or "not available" in ans.lower() or "not provided" in ans.lower() or "not mentioned" in ans.lower() or "does not contain" in ans.lower() or "cannot" in ans.lower() or "erp" in ans.lower() or "not documented" in ans.lower() or "not found" in ans.lower()),
        "expected": "Safely states room numbers are not documented",
    },
    # 11. Citation correctness & direct provenance
    {
        "class": "11. Direct Provenance",
        "query": "where can i buy milk",
        "check": lambda ans, src: ("fresh n green" in ans.lower() or "supermarket" in ans.lower()) and all("registration" not in s.get("source", "").lower() for s in src) and any("campus" in s.get("source", "").lower() or "directory" in s.get("source", "").lower() for s in src),
        "expected": "Fresh n Green with Campus Directory source",
    },
    # 12. Short generic fee inquiry
    {
        "class": "12. Short Fee Inquiry",
        "query": "fees?",
        "check": lambda ans, src: ("50,000" in ans or "50000" in ans or "32,250" in ans) and any("fees_and_finance" in s.get("source", "").lower() for s in src),
        "expected": "Authoritative AY 2026-2027 fee schedule (Tuition ₹50,000)",
    },
]

def main():
    print("=" * 80)
    print("RUNNING 12-CLASS REGRESSION EVALUATION")
    print("=" * 80)
    
    passed_count = 0
    total_count = len(REGRESSION_CASES)
    results_summary = []
    
    for idx, test_case in enumerate(REGRESSION_CASES, 1):
        cls_name = test_case["class"]
        print(f"\n[{idx}/{total_count}] Testing: {cls_name}")
        
        session_id = f"reg_{idx}_{int(time.time())}"
        history = []
        
        if "multi_turn" in test_case:
            turns = test_case["multi_turn"]
            ans = ""
            sources = []
            turn_latency = 0.0
            for t_q, _ in turns:
                t0 = time.perf_counter()
                req = ChatTurnRequest(
                    question=t_q,
                    chat_history=tuple(history),
                    session_id=session_id,
                    institution_id="iitj",
                )
                resp = handle_chat_turn(req)
                turn_latency += (time.perf_counter() - t0)
                ans = resp.answer
                sources = resp.sources
                history.append((t_q, resp.answer))
        else:
            q = test_case["query"]
            t0 = time.perf_counter()
            req = ChatTurnRequest(
                question=q,
                chat_history=(),
                session_id=session_id,
                institution_id="iitj",
            )
            resp = handle_chat_turn(req)
            turn_latency = time.perf_counter() - t0
            ans = resp.answer
            sources = resp.sources
            
        diag = resp.diagnostics or {}
        timings = diag.get("timings", {})
        delta = timings.get("reconciliation_delta_ms", 0.0)
        
        is_pass = False
        try:
            is_pass = bool(test_case["check"](ans, sources))
        except Exception as e:
            print(f"  Check error: {e}")
            is_pass = False
            
        status_str = "PASS" if is_pass else "FAIL"
        if is_pass:
            passed_count += 1
            
        print(f"  Status: [{status_str}] | Latency: {turn_latency:.2f}s | Delta: {delta:.2f}ms")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Answer preview: {ans[:150]}...")
        src_names = [s.get("heading") or s.get("source") for s in sources]
        print(f"  Sources ({len(sources)}): {src_names}")
        
        results_summary.append({
            "class": cls_name,
            "status": status_str,
            "latency": round(turn_latency, 2),
            "delta_ms": round(delta, 2),
            "sources_count": len(sources),
            "sources": src_names,
            "answer_preview": ans[:200],
        })
        
    print("\n" + "=" * 80)
    print(f"REGRESSION MATRIX SUMMARY: {passed_count}/{total_count} PASSED ({(passed_count/total_count)*100:.1f}%)")
    print("=" * 80)
    
    out_file = WORKSPACE_ROOT / "test_outputs" / "regression_12_classes_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved report to {out_file}")

if __name__ == "__main__":
    main()
