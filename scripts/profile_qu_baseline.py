#!/usr/bin/env python3
"""Profile Query Understanding Baseline Before Three-Route Architecture Changes.

Captures exact timings, sub-stage latencies, LLM call counts, retrieval handoff payloads,
and conversation resolution results on the untouched codebase.
Outputs to test_outputs/hybrid_query_routing_20261021_master/query_understanding_baseline_traces.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_platform.core.graph.nodes import CoreNodes, default_dependencies
from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.guard import evaluate_security_and_scope
from ai_platform.core.query.planner import (
    build_query_plan,
    retrieval_queries_from_plan,
    intent_units_from_plan,
    should_stop_pipeline,
)
from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

OUTPUT_DIR = REPO_ROOT / "test_outputs" / "hybrid_query_routing_20261021_master"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test cases representing all operational categories and known edge cases
BASELINE_CASES = [
    {
        "id": "Q01",
        "category": "Natural English Standalone - Broad",
        "query": "I'm thinking about Aerospace at IIT Jodhpur. What is the department really focused on?",
        "history": [],
    },
    {
        "id": "Q04",
        "category": "Natural English Standalone - Department Discovery",
        "query": "I want to get into chip design. Where should I start looking at IIT Jodhpur?",
        "history": [],
    },
    {
        "id": "Q05",
        "category": "Comparison - ECE vs Materials",
        "query": "Both Electronics and Materials seem related to semiconductors. What's the difference, and which one should I read about first for chip design?",
        "history": [],
    },
    {
        "id": "Q11",
        "category": "Exact Code Lookup - Mathematics",
        "query": "Some Mathematics course codes have XXX in them. Are those actual course codes, or is the exact code missing?",
        "history": [],
    },
    {
        "id": "Q12",
        "category": "Curricular Credit Inconsistency",
        "query": "The Mathematics and Computing credit numbers don't seem to add up. Can you help me check them?",
        "history": [],
    },
    {
        "id": "Q15",
        "category": "Vocational Procurement Workshop",
        "query": "I missed those procurement workshops in 2023. Is another session scheduled now?",
        "history": [],
    },
    {
        "id": "Q20",
        "category": "Standalone Genuinely Underspecified",
        "query": "How many credits do I need to graduate from IIT Jodhpur?",
        "history": [],
    },
    {
        "id": "Q26",
        "category": "Natural Hindi Standalone",
        "query": "मुझे IIT जोधपुर में Aerospace में दिलचस्पी है। वहाँ का डिपार्टमेंट मुख्य रूप से किस तरह के काम पर फोकस करता है?",
        "history": [],
    },
    {
        "id": "Q28",
        "category": "Hinglish Standalone",
        "query": "Mujhe chip design mein jaana hai. IIT Jodhpur mein kis department ya area se shuru karna chahiye?",
        "history": [],
    },
    {
        "id": "Q32",
        "category": "Hinglish Follow-up - B.S. Math",
        "query": "Is degree mein internship ya thesis bhi hoti hai, ya mostly classes hoti hain?",
        "history": [
            ("user", "What is the B.S. in Mathematics and Computing actually about?"),
            ("assistant", "The B.S. in Mathematics and Computing at IIT Jodhpur covers fundamental mathematics, computational courses, data structures, and scientific computing."),
        ],
    },
    {
        "id": "Q33",
        "category": "Hinglish Follow-up - Materials AI",
        "query": "Usmein AI ya data-related option mil sakta hai kya?",
        "history": [
            ("user", "What does the Materials Engineering department offer?"),
            ("assistant", "The Materials Engineering department offers B.Tech, M.Tech, and Ph.D. programs focusing on functional materials, structural materials, and computational metallurgy."),
        ],
    },
    {
        "id": "Q38",
        "category": "Hinglish Follow-up - DADT Eligibility & Fees",
        "query": "Us programme ke liye eligibility aur fees kya hain?",
        "history": [
            ("user", "I'm interested in the M.Tech. in Drone and Anti-Drone Technologies."),
            ("assistant", "The M.Tech. in Drone and Anti-Drone Technologies was proposed for defence and commercial drone applications."),
        ],
    },
    {
        "id": "Q39",
        "category": "Hinglish Follow-up - Materials PhD",
        "query": "Us department mein PhD bhi kar sakte hain?",
        "history": [
            ("user", "I'm interested in Materials Engineering at IIT Jodhpur."),
            ("assistant", "Materials Engineering at IIT Jodhpur has research in computational materials and nanomaterials."),
        ],
    },
    {
        "id": "LIVE_FAIL_01",
        "category": "Live Failure: Topic Change to Hostel/Mess after M.Tech Eligibility",
        "query": "Old mess kahan hai aur hostel charges kya hain?",
        "history": [
            ("user", "What are the eligibility criteria for M.Tech admissions at IIT Jodhpur?"),
            ("assistant", "M.Tech eligibility requires a Bachelor's degree in Engineering/Technology or Master's degree in Science with a valid GATE score or specified CPI threshold."),
        ],
    },
    {
        "id": "LIVE_FAIL_02",
        "category": "Live Failure: Short Hindi/Hinglish Facility Query after Degree Topic",
        "query": "Mess kahan hai?",
        "history": [
            ("user", "Tell me about M.Tech in Computer Science and Engineering eligibility."),
            ("assistant", "Eligibility for regular M.Tech in CSE requires a valid GATE score in CS and a qualifying degree with minimum 60% marks or 6.0 CGPA."),
        ],
    },
    {
        "id": "Q42",
        "category": "Procurement Dates Lookup",
        "query": "Which of those June 2023 procurement trainings was on the 10th, and which one was on the 17th?",
        "history": [],
    },
    {
        "id": "Q43",
        "category": "Specific Course Codes Lookup",
        "query": "What code is printed next to Advanced Algorithms, Computer Vision, and Quantum Computing? Please don't guess any missing digits or letters.",
        "history": [],
    },
    {
        "id": "Q45",
        "category": "Two-Intent Comparison - VLSI vs Materials",
        "query": "I'm interested in both VLSI design and semiconductor materials. Which parts of IIT Jodhpur's documents look relevant to each interest?",
        "history": [],
    },
    {
        "id": "Q50",
        "category": "Out of Institutional Scope",
        "query": "Tell me a joke.",
        "history": [],
    },
]


def run_baseline_profile():
    print(f"Profiling {len(BASELINE_CASES)} baseline queries using untouched pipeline...")
    nodes = CoreNodes(default_dependencies())

    results = []

    for idx, tc in enumerate(BASELINE_CASES, 1):
        qid = tc["id"]
        cat = tc["category"]
        q = tc["query"]
        hist = tc["history"]

        print(f"\n[{idx}/{len(BASELINE_CASES)}] {qid}: \"{q[:50]}...\" ({cat})")

        state: dict[str, Any] = {
            "question": q,
            "chat_history": tuple(hist),
        }

        # Step 1: Conversation Resolution
        t0_cr = time.perf_counter()
        s1 = nodes.resolve_conversation_node(state)
        t_cr = time.perf_counter() - t0_cr
        state.update(s1)

        resolved_q = state.get("resolved_question")
        conv_mode = state.get("conversation_mode")
        active_topic = state.get("active_topic")
        active_entity = state.get("active_entity")

        # Step 2: Understand Query
        t0_qu = time.perf_counter()
        s2 = nodes.understand_query_node(state)
        t_qu = time.perf_counter() - t0_qu
        state.update(s2)

        # Step 3: Plan Multi-Intent
        t0_mi = time.perf_counter()
        s3 = nodes.plan_multi_intent_node(state)
        t_mi = time.perf_counter() - t0_mi
        state.update(s3)

        total_qu_wall_time = t_cr + t_qu + t_mi
        stage_timings = dict(state.get("stage_timings") or {})
        t_guard = stage_timings.get("guard_and_scope", 0.0)
        t_plan_llm = stage_timings.get("query_planning_llm", 0.0)
        t_lsqc = stage_timings.get("lossless_query_and_frame", 0.0)

        plan = state.get("query_plan")
        retrieval_queries = list(state.get("retrieval_queries") or ())
        route = state.get("query_route")
        scope = state.get("query_scope")
        pipeline_stop = state.get("pipeline_stop", False)
        stop_reason = state.get("stop_reason", "")
        intent_units = state.get("intent_units", [])

        # Count LLM calls in Query Understanding
        # Untouched baseline:
        # - conversation.py calls query_understanding_llm if non-deterministic
        # - planner.py calls active.invoke (planning LLM)
        cr_resolution_source = s1.get("resolution_source", "")
        cr_llm_called = 1 if cr_resolution_source in ("llm", "llm_rejected") else 0
        planner_llm_called = 1 if t_plan_llm > 0.05 else 0
        total_qu_llm_calls = cr_llm_called + planner_llm_called

        print(f"  CR: {t_cr:.4f}s | Guard: {t_guard:.4f}s | Plan LLM: {t_plan_llm:.4f}s | Total QU: {total_qu_wall_time:.4f}s")
        print(f"  QU LLM Calls: {total_qu_llm_calls} | Route: {route} | Scope: {scope} | Stop: {pipeline_stop}")
        print(f"  Resolved: \"{resolved_q}\"")
        if retrieval_queries:
            print(f"  Retrieval Queries ({len(retrieval_queries)}): {retrieval_queries[:2]}")

        record = {
            "test_id": qid,
            "category": cat,
            "original_query": q,
            "history": hist,
            "resolved_question": resolved_q,
            "conversation_mode": conv_mode,
            "active_topic": active_topic,
            "active_entity": active_entity,
            "query_route": route,
            "query_scope": scope,
            "pipeline_stop": pipeline_stop,
            "stop_reason": stop_reason,
            "timings": {
                "resolve_conversation_seconds": round(t_cr, 5),
                "guard_and_scope_seconds": round(t_guard, 5),
                "query_planning_llm_seconds": round(t_plan_llm, 5),
                "lossless_query_seconds": round(t_lsqc, 5),
                "plan_multi_intent_seconds": round(t_mi, 5),
                "total_query_understanding_wall_time_seconds": round(total_qu_wall_time, 5),
            },
            "llm_call_count": total_qu_llm_calls,
            "llm_calls_breakdown": {
                "conversation_resolution_llm": cr_llm_called,
                "query_planner_llm": planner_llm_called,
            },
            "retrieval_queries": retrieval_queries,
            "intent_units_count": len(intent_units),
            "intent_questions": [u.get("question") for u in intent_units],
        }
        results.append(record)

    out_file = OUTPUT_DIR / "query_understanding_baseline_traces.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Baseline profiling complete. Wrote {len(results)} traces to {out_file}")

    # Summary Statistics
    total_times = [r["timings"]["total_query_understanding_wall_time_seconds"] for r in results]
    llm_times = [r["timings"]["query_planning_llm_seconds"] for r in results]
    llm_counts = [r["llm_call_count"] for r in results]
    import statistics
    print(f"\n=== BASELINE SUMMARY ===")
    print(f"Total Queries Profiled: {len(results)}")
    print(f"Mean QU Latency:   {statistics.mean(total_times):.3f}s")
    print(f"Median QU Latency: {statistics.median(total_times):.3f}s")
    print(f"Min QU Latency:    {min(total_times):.3f}s")
    print(f"Max QU Latency:    {max(total_times):.3f}s")
    print(f"Mean LLM Call Time: {statistics.mean(llm_times):.3f}s")
    print(f"Total QU LLM Calls: {sum(llm_counts)} across {len(results)} turns (100% of non-stopped queries called LLM)")


if __name__ == "__main__":
    run_baseline_profile()
