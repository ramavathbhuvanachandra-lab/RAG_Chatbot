"""Reproduction and diagnostic baseline capture for Tests A through G.

Executes Tests A-G through canonical handle_chat_turn, dumping full traces,
diagnostics, candidates, final context, and generated answers.
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

OUT_DIR = WORKSPACE_ROOT / "test_outputs" / "baseline_reproduction_7tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_test_turn(test_id: str, desc: str, question: str, history: list[dict] | None = None) -> dict:
    print(f"\n==========================================")
    print(f"RUNNING {test_id}: {desc}")
    print(f"Question: '{question}'")
    if history:
        print(f"History: {len(history)} messages")

    t0 = time.perf_counter()
    req = ChatTurnRequest(
        question=question,
        chat_history=tuple(history or []),
        session_id=f"session_{test_id}",
        institution_id="iitj",
    )
    resp = handle_chat_turn(req)
    latency = time.perf_counter() - t0

    diag = resp.diagnostics or {}
    sec_a = diag.get("section_a_candidates", [])
    sec_b = diag.get("section_b_context", {})
    sec_c = diag.get("section_c_citations", [])
    funnel = diag.get("funnel", {})
    resolved_q = diag.get("resolved_question") or question

    print(f"Latency: {latency:.2f}s | Safe Refusal: {resp.is_safe_refusal}")
    print(f"Resolved Question: '{resolved_q}'")
    print(f"Candidates (Sec A): {len(sec_a)} | Verified: {funnel.get('verified_count', 0)}")
    print(f"Sources emitted (Sec C / student sources): {len(resp.sources)}")
    print(f"Answer:\n{resp.formatted_response}\n")

    return {
        "test_id": test_id,
        "description": desc,
        "question": question,
        "history": history or [],
        "resolved_question": resolved_q,
        "latency_seconds": round(latency, 2),
        "is_safe_refusal": resp.is_safe_refusal,
        "evidence_status": resp.evidence_status,
        "answer": resp.answer,
        "formatted_response": resp.formatted_response,
        "sources": resp.sources,
        "funnel": funnel,
        "section_a_candidates": sec_a[:15],
        "section_b_context": sec_b,
        "section_c_citations": sec_c,
    }


def main():
    results = {}

    # Test A: M.Tech tuition
    res_a = run_test_turn(
        "test_a",
        "M.Tech tuition fee per semester",
        "What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. programme at IIT Jodhpur?",
        history=None,
    )
    results["test_a"] = res_a

    # Test B: Sponsored M.Tech follow-up (in same conversation as Test A)
    hist_b = [
        {"role": "user", "content": res_a["question"]},
        {"role": "assistant", "content": res_a["formatted_response"]},
    ]
    res_b = run_test_turn(
        "test_b",
        "Sponsored M.Tech follow-up",
        "What about sponsored students?",
        history=hist_b,
    )
    results["test_b"] = res_b

    # Test G: Hostel and mess (following M.Tech fee in Test A)
    hist_g = [
        {"role": "user", "content": res_a["question"]},
        {"role": "assistant", "content": res_a["formatted_response"]},
    ]
    res_g = run_test_turn(
        "test_g",
        "Hostel and mess charges follow-up",
        "Does that include hostel and mess charges?",
        history=hist_g,
    )
    results["test_g"] = res_g

    # Test C: Hostel eligibility
    res_c = run_test_turn(
        "test_c",
        "Hostel eligibility criteria",
        "What are the eligibility criteria for Hostel?",
        history=None,
    )
    results["test_c"] = res_c

    # Test D: Electrical Engineering research
    res_d = run_test_turn(
        "test_d",
        "Electrical Engineering research areas",
        "What research areas are available in Electrical Engineering?",
        history=None,
    )
    results["test_d"] = res_d

    # Test E: Conversation intent after clarification
    # Turn 1
    res_e1 = run_test_turn(
        "test_e_turn1",
        "Ambiguous programme eligibility",
        "What are the eligibility criteria for this programme?",
        history=None,
    )
    results["test_e_turn1"] = res_e1

    # Turn 2
    hist_e2 = [
        {"role": "user", "content": res_e1["question"]},
        {"role": "assistant", "content": res_e1["formatted_response"]},
    ]
    res_e2 = run_test_turn(
        "test_e_turn2",
        "Clarification response: B.Tech.",
        "B.Tech.",
        history=hist_e2,
    )
    results["test_e_turn2"] = res_e2

    # Test F: B.Tech fee
    res_f = run_test_turn(
        "test_f",
        "B.Tech fee structure",
        "What is the fee structure for B.Tech.?",
        history=None,
    )
    results["test_f"] = res_f

    out_file = OUT_DIR / "baseline_7tests_trace.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved all baseline test traces to {out_file}")


if __name__ == "__main__":
    main()
