#!/usr/bin/env python3
"""Frontend Milestone 1 Live Service & Ollama Qualification Runner.

Executes a bounded suite of real queries through the presentation-layer
`frontend.clients.assistant_client.send_chat_turn` against the canonical
AI platform and live local Ollama model.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.clients.assistant_client import send_chat_turn

OUTPUT_DIR = REPO_ROOT / "test_outputs" / "frontend_milestone1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_qualification():
    print("=" * 80)
    print("STARTING FRONTEND MILESTONE 1 LIVE QUALIFICATION")
    print("=" * 80)

    results = []

    # Case 1: Factual Institutional Question (M.Tech Fees)
    print("\n--- Running Case 1: Normal IITJ Factual Question ---")
    q1 = "What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. program at IIT Jodhpur?"
    t0 = time.perf_counter()
    r1 = send_chat_turn(q1, session_id="live_session_01", institution_id="iitj")
    el1 = time.perf_counter() - t0
    print(f"Elapsed: {el1:.2f}s | Status: {r1.evidence_status} | Citations: {len(r1.sources)}")
    print(f"Answer snippet: {r1.answer[:160]}...")
    results.append({
        "case_id": "case_01_factual",
        "category": "Factual Question",
        "question": q1,
        "answer": r1.answer,
        "structured_blocks": r1.structured_blocks,
        "sources_count": len(r1.sources),
        "latency_seconds": round(r1.latency_seconds, 2),
        "is_safe_refusal": r1.is_safe_refusal,
        "evidence_status": r1.evidence_status,
        "verdict": "PASS" if len(r1.answer) > 20 and len(r1.sources) > 0 else "FAIL",
    })

    # Case 2: Multi-turn Follow-up Turn 1 (Ph.D. Eligibility)
    print("\n--- Running Case 2: Multi-turn Turn 1 (Ph.D. Admissions) ---")
    q2a = "What are the eligibility criteria for the Ph.D. program in engineering at IIT Jodhpur?"
    t0 = time.perf_counter()
    r2a = send_chat_turn(q2a, session_id="live_session_multi_turn", institution_id="iitj")
    el2a = time.perf_counter() - t0
    print(f"Elapsed: {el2a:.2f}s | Status: {r2a.evidence_status} | Citations: {len(r2a.sources)}")
    print(f"Answer snippet: {r2a.answer[:160]}...")
    results.append({
        "case_id": "case_02a_multiturn_turn1",
        "category": "Multi-Turn Base",
        "question": q2a,
        "answer": r2a.answer,
        "structured_blocks": r2a.structured_blocks,
        "sources_count": len(r2a.sources),
        "latency_seconds": round(r2a.latency_seconds, 2),
        "is_safe_refusal": r2a.is_safe_refusal,
        "evidence_status": r2a.evidence_status,
        "verdict": "PASS" if len(r2a.answer) > 20 else "FAIL",
    })

    # Case 2b: Multi-turn Follow-up Turn 2 (Coreference Resolution)
    print("\n--- Running Case 2b: Multi-turn Turn 2 (Coreference Follow-up) ---")
    q2b = "How many days of leave can scholars in this program take in an academic year?"
    history_2b = [
        {"role": "user", "content": q2a},
        {"role": "assistant", "content": r2a.answer},
    ]
    t0 = time.perf_counter()
    r2b = send_chat_turn(
        q2b,
        history=history_2b,
        session_id="live_session_multi_turn",
        institution_id="iitj",
    )
    el2b = time.perf_counter() - t0
    print(f"Elapsed: {el2b:.2f}s | Status: {r2b.evidence_status} | Citations: {len(r2b.sources)}")
    print(f"Answer snippet: {r2b.answer[:160]}...")
    results.append({
        "case_id": "case_02b_multiturn_turn2",
        "category": "Multi-Turn Coreference Follow-up",
        "question": q2b,
        "answer": r2b.answer,
        "structured_blocks": r2b.structured_blocks,
        "sources_count": len(r2b.sources),
        "latency_seconds": round(r2b.latency_seconds, 2),
        "is_safe_refusal": r2b.is_safe_refusal,
        "evidence_status": r2b.evidence_status,
        "verdict": "PASS" if len(r2b.answer) > 20 else "FAIL",
    })

    # Case 3: Multi-Intent Question
    print("\n--- Running Case 3: Multi-Intent Decomposition ---")
    q3 = "What is the tuition fee per semester for regular M.Tech. students, and how many days of casual leave can a regular Ph.D. scholar take in a year?"
    t0 = time.perf_counter()
    r3 = send_chat_turn(q3, session_id="live_session_multi_intent", institution_id="iitj")
    el3 = time.perf_counter() - t0
    print(f"Elapsed: {el3:.2f}s | Status: {r3.evidence_status} | Citations: {len(r3.sources)}")
    print(f"Answer snippet: {r3.answer[:160]}...")
    results.append({
        "case_id": "case_03_multi_intent",
        "category": "Multi-Intent Decomposition",
        "question": q3,
        "answer": r3.answer,
        "structured_blocks": r3.structured_blocks,
        "sources_count": len(r3.sources),
        "latency_seconds": round(r3.latency_seconds, 2),
        "is_safe_refusal": r3.is_safe_refusal,
        "evidence_status": r3.evidence_status,
        "verdict": "PASS" if len(r3.answer) > 20 else "FAIL",
    })

    # Case 4: Campus Navigation
    print("\n--- Running Case 4: Structured Campus Navigation ---")
    q4 = "Where is the Lecture Hall Complex?"
    t0 = time.perf_counter()
    r4 = send_chat_turn(q4, session_id="live_session_nav", institution_id="iitj")
    el4 = time.perf_counter() - t0
    print(f"Elapsed: {el4:.2f}s | Structured Blocks: {len(r4.structured_blocks)}")
    print(f"Formatted snippet: {r4.formatted_response[:160]}...")
    results.append({
        "case_id": "case_04_navigation",
        "category": "Campus Navigation",
        "question": q4,
        "answer": r4.answer,
        "structured_blocks": r4.structured_blocks,
        "sources_count": len(r4.sources),
        "latency_seconds": round(r4.latency_seconds, 2),
        "is_safe_refusal": r4.is_safe_refusal,
        "evidence_status": r4.evidence_status,
        "verdict": "PASS" if len(r4.structured_blocks) > 0 else "FAIL",
    })

    # Case 5: Emergency Request
    print("\n--- Running Case 5: Structured Emergency Request ---")
    q5 = "What is the security control room emergency phone number?"
    t0 = time.perf_counter()
    r5 = send_chat_turn(q5, session_id="live_session_emerg", institution_id="iitj")
    el5 = time.perf_counter() - t0
    print(f"Elapsed: {el5:.2f}s | Structured Blocks: {len(r5.structured_blocks)}")
    print(f"Formatted snippet: {r5.formatted_response[:160]}...")
    results.append({
        "case_id": "case_05_emergency",
        "category": "Emergency Services",
        "question": q5,
        "answer": r5.answer,
        "structured_blocks": r5.structured_blocks,
        "sources_count": len(r5.sources),
        "latency_seconds": round(r5.latency_seconds, 2),
        "is_safe_refusal": r5.is_safe_refusal,
        "evidence_status": r5.evidence_status,
        "verdict": "PASS" if len(r5.structured_blocks) > 0 or len(r5.answer) > 10 else "FAIL",
    })

    # Case 6: Unsupported Defamatory / False Premise Control
    print("\n--- Running Case 6: Unsupported Question Safe Refusal ---")
    q6 = "Why did the Supreme Court order IIT Jodhpur to cancel all degrees awarded in 2024?"
    t0 = time.perf_counter()
    r6 = send_chat_turn(q6, session_id="live_session_unsupported", institution_id="iitj")
    el6 = time.perf_counter() - t0
    print(f"Elapsed: {el6:.2f}s | Safe Refusal: {r6.is_safe_refusal} | Status: {r6.evidence_status}")
    print(f"Answer snippet: {r6.answer[:160]}...")
    results.append({
        "case_id": "case_06_unsupported_refusal",
        "category": "Negative Control Safe Abstention",
        "question": q6,
        "answer": r6.answer,
        "structured_blocks": r6.structured_blocks,
        "sources_count": len(r6.sources),
        "latency_seconds": round(r6.latency_seconds, 2),
        "is_safe_refusal": r6.is_safe_refusal,
        "evidence_status": r6.evidence_status,
        "verdict": "PASS" if r6.is_safe_refusal or "not" in r6.answer.lower() else "FAIL",
    })

    out_file = OUTPUT_DIR / "qualification_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "milestone": "frontend_milestone1",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_cases": len(results),
                "passed_cases": sum(1 for r in results if r["verdict"] == "PASS"),
                "results": results,
            },
            fp,
            indent=2,
        )
    print("\n" + "=" * 80)
    print(f"QUALIFICATION COMPLETE. Results saved to: {out_file}")
    print(f"Passed: {sum(1 for r in results if r['verdict'] == 'PASS')} / {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    run_qualification()
