"""Reproduction script for RAG Recovery Scenarios A through F.

Executes all 6 required institutional scenarios through canonical handle_chat_turn,
saving complete stage-by-stage diagnostics:
- Query understanding & plan
- Retrieved & fused candidates
- Verification decisions & reasons
- Ranked candidates & scoring
- Packaged evidence context sent to LLM
- Answer generation & safe refusal status
- Citations emitted
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

OUT_DIR = WORKSPACE_ROOT / "test_outputs" / "recovery_baseline_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    {
        "id": "scenario_a",
        "title": "Institute fee table (ITEP AY 2026-27)",
        "question": "What are the ITEP fees for academic year 2026-27, including tuition fee, admission fee, convocation fee, semester fee, refundable deposit, and experiential and professional development fee?",
        "expected_facts": [
            "Tuition: ₹50,000",
            "Admission fee: ₹3,800",
            "Convocation fee: ₹4,000",
            "Semester fee: ₹32,250",
            "Refundable deposit: ₹8,000",
            "EPDF: ₹2,000",
        ],
    },
    {
        "id": "scenario_b",
        "title": "Short-term accommodation rates",
        "question": "What are the applicable short-term accommodation rates for visitors or executive students at IIT Jodhpur, including single and double occupancy, with and without bedding?",
        "expected_facts": [
            "Without bedding: Double ₹450/-, Single ₹750/-",
            "With bedding: Double ₹600/-, Single ₹900/-",
            "Excluding GST",
        ],
    },
    {
        "id": "scenario_c",
        "title": "Tuition exemptions and semester fees",
        "question": "Which students are exempt from paying tuition fees at IIT Jodhpur, and are hostel fees and mess charges included in the semester fee?",
        "expected_facts": [
            "SC, ST, and PwD students are exempt from tuition fees",
            "Semester fee includes hostel fees",
            "Mess charges are not included in semester fee and must be paid separately as mess advance",
        ],
    },
    {
        "id": "scenario_d",
        "title": "Longer accommodation bookings",
        "question": "What are the charges for longer accommodation bookings at IIT Jodhpur, and what basic facilities are provided in the hostel?",
        "expected_facts": [
            "Exceeding 11 days but less than a month: Rs. 2175 (excl. GST)",
            "Duration exceeding a month: Rs. 2175 per month (excl. GST)",
            "Basic facilities: bed, basic furniture (subject to availability), air-conditioning, bedding (if opted)",
        ],
    },
    {
        "id": "scenario_e",
        "title": "Postgraduate admission cycle",
        "question": "When does the postgraduate admission cycle typically occur for Semester 1 and Semester 2 at IIT Jodhpur?",
        "expected_facts": [
            "Semester 1: April-May",
            "Semester 2: November-December",
        ],
    },
    {
        "id": "scenario_f",
        "title": "Security and Transport office responsibilities",
        "question": "What does the Institute's Security and Transport office handle at IIT Jodhpur?",
        "expected_facts": [
            "Campus security, 24/7 security patrols",
            "Surveillance systems",
            "Emergency response protocols",
            "Transportation services / campus transport facilities",
        ],
    },
]


def run_scenario(sc: dict) -> dict:
    sc_id = sc["id"]
    title = sc["title"]
    q = sc["question"]

    print(f"\n{'='*60}")
    print(f"RUNNING {sc_id.upper()}: {title}")
    print(f"Question: '{q}'")

    t0 = time.perf_counter()
    req = ChatTurnRequest(
        question=q,
        chat_history=(),
        session_id=f"session_recovery_{sc_id}",
        institution_id="iitj",
    )
    resp = handle_chat_turn(req)
    latency = time.perf_counter() - t0

    diag = resp.diagnostics or {}
    sec_a = diag.get("section_a_candidates", [])
    sec_b = diag.get("section_b_context", {})
    sec_c = diag.get("section_c_citations", [])
    funnel = diag.get("funnel", {})
    resolved_q = diag.get("resolved_question") or q
    query_plan = diag.get("query_plan", {})

    print(f"Latency: {latency:.2f}s | Safe Refusal: {resp.is_safe_refusal} | Status: {resp.evidence_status}")
    print(f"Candidates (Sec A): {len(sec_a)} | Verified: {funnel.get('verified_count', 0)}")
    print(f"Sources emitted (Sec C): {len(resp.sources)}")
    print(f"Answer:\n{resp.formatted_response}\n")

    return {
        "scenario_id": sc_id,
        "title": title,
        "question": q,
        "resolved_question": resolved_q,
        "latency_seconds": round(latency, 2),
        "is_safe_refusal": resp.is_safe_refusal,
        "evidence_status": resp.evidence_status,
        "answer": resp.answer,
        "formatted_response": resp.formatted_response,
        "sources": resp.sources,
        "expected_facts": sc["expected_facts"],
        "funnel": funnel,
        "query_plan": query_plan,
        "section_a_candidates": sec_a[:15],
        "section_b_context": sec_b,
        "section_c_citations": sec_c,
    }


def main():
    results = {}
    for sc in SCENARIOS:
        res = run_scenario(sc)
        results[sc["id"]] = res

    out_file = OUT_DIR / "recovery_baseline_traces.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[DONE] Saved all recovery baseline traces to {out_file}")


if __name__ == "__main__":
    main()
