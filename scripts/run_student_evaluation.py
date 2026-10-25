#!/usr/bin/env python3
"""Run the Phase 14 20-Question Realistic Student Evaluation against the canonical runtime."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

STUDENT_QUESTIONS = [
    # Group 1: Everyday Student Inquiries
    {
        "id": "S01",
        "group": "everyday",
        "title": "Campus Dining & Mess Payment",
        "turns": [
            "What are the hostel dining timings and can students pay for guest meals at the mess?"
        ],
    },
    {
        "id": "S02",
        "group": "everyday",
        "title": "Academic Standing & CPI Criteria",
        "turns": [
            "What is the minimum CPI requirement for undergraduate students to maintain good academic standing?"
        ],
    },
    {
        "id": "S03",
        "group": "everyday",
        "title": "Lost Property & Campus Security",
        "turns": [
            "Where can I report lost property or identity cards on campus?"
        ],
    },
    {
        "id": "S04",
        "group": "everyday",
        "title": "Summer Internships",
        "turns": [
            "How does the institute assist undergraduate students with summer internship opportunities?"
        ],
    },
    {
        "id": "S05",
        "group": "everyday",
        "title": "Campus Sports Facilities",
        "turns": [
            "Are campus sports facilities and gymnasiums open on weekends?"
        ],
    },

    # Group 2: Concise Programme, Course, & Research Questions
    {
        "id": "S06",
        "group": "programmes",
        "title": "Chip Design Department Selection",
        "turns": [
            "I'm interested in chip design. Which department should I look at?"
        ],
    },
    {
        "id": "S07",
        "group": "programmes",
        "title": "Drone Programme Launch Status",
        "turns": [
            "Is the drone M.Tech. starting this year?"
        ],
    },
    {
        "id": "S08",
        "group": "programmes",
        "title": "Materials Engineering with AI",
        "turns": [
            "What options do I have in Materials Engineering if I'm interested in AI?"
        ],
    },
    {
        "id": "S09",
        "group": "programmes",
        "title": "Mathematics and Computing Overview",
        "turns": [
            "Tell me about the Mathematics and Computing degree."
        ],
    },
    {
        "id": "S10",
        "group": "programmes",
        "title": "Bioscience and Bioengineering Research",
        "turns": [
            "What research areas are pursued in Bioscience and Bioengineering?"
        ],
    },

    # Group 3: Multi-Part Conversational Questions
    {
        "id": "S11",
        "group": "multi_part",
        "title": "M.Tech. in AI Details",
        "turns": [
            "Can you tell me about M.Tech. in AI at IIT Jodhpur—what are the admission requirements, duration, and assistantship?"
        ],
    },
    {
        "id": "S12",
        "group": "multi_part",
        "title": "Ph.D. Admissions & Contingency",
        "turns": [
            "I want to know about Ph.D. admissions: what is the minimum qualifying score, contingency amount, and can working professionals join part-time?"
        ],
    },
    {
        "id": "S13",
        "group": "multi_part",
        "title": "PMKVY Training Impact & Roles",
        "turns": [
            "What were the results of the PMKVY training, which job roles were trained, and how many people got certified?"
        ],
    },
    {
        "id": "S14",
        "group": "multi_part",
        "title": "B.Tech. Branch Change",
        "turns": [
            "How does branch change work for first-year B.Tech. students and what is the CGPA cutoff?"
        ],
    },

    # Group 4: Context-Dependent Multi-Turn Follow-Ups
    {
        "id": "S15",
        "group": "conversational",
        "title": "Materials Engineering Degrees & Minors",
        "turns": [
            "What degree programmes does the Department of Materials Engineering offer?",
            "What about minor options for B.Tech. students in that department?",
        ],
    },
    {
        "id": "S16",
        "group": "conversational",
        "title": "Cyber-Physical Systems Programme & Eligibility",
        "turns": [
            "Does IIT Jodhpur offer an M.Tech. in Cyber-Physical Systems?",
            "What are the eligibility criteria for it?",
        ],
    },
    {
        "id": "S17",
        "group": "conversational",
        "title": "GeM vs GFR Comparison Follow-Up",
        "turns": [
            "Tell me about the Specialized Training in GeM.",
            "How is it different from the GFR training?",
        ],
    },

    # Group 5: Ambiguous, Underspecified, & False-Premise Questions
    {
        "id": "S18",
        "group": "ambiguous_refusal",
        "title": "Underspecified Fee Inquiry",
        "turns": [
            "What is the fee for the course?"
        ],
    },
    {
        "id": "S19",
        "group": "ambiguous_refusal",
        "title": "False Premise: Closed Department",
        "turns": [
            "Why did IIT Jodhpur shut down its Aerospace Engineering Department in 2024?"
        ],
    },
    {
        "id": "S20",
        "group": "ambiguous_refusal",
        "title": "Unsupported Detail: Private Room Allocations",
        "turns": [
            "What are the specific hostel room numbers allocated to incoming Ph.D. scholars in 2026?"
        ],
    },
]


def run_student_eval(output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_file = output_dir / "student_20q_traces.json"

    print("=== Starting Phase 14: 20 Realistic Student Question Evaluation ===")
    results = []

    for idx, item in enumerate(STUDENT_QUESTIONS, 1):
        q_id = item["id"]
        title = item["title"]
        group = item["group"]
        turns = item["turns"]

        print(f"\n[{idx}/20] Running {q_id}: {title} ({len(turns)} turns)")
        session_id = f"student_eval_{q_id.lower()}_{int(time.time())}"
        conversation_history: list[tuple[str, str]] = []
        turn_traces = []

        for turn_idx, user_query in enumerate(turns, 1):
            print(f"  Turn {turn_idx}: {user_query}")
            t_start = time.perf_counter()

            req = ChatTurnRequest(
                question=user_query,
                chat_history=tuple(conversation_history),
                session_id=session_id,
                institution_id="iitj",
            )

            try:
                resp = handle_chat_turn(req)
                turn_latency = time.perf_counter() - t_start
                diag = resp.diagnostics or {}
                raw_state = diag.get("raw_state") or diag

                # Record conversation turn
                conversation_history.append((user_query, resp.answer))

                turn_data = {
                    "turn_index": turn_idx,
                    "query": user_query,
                    "resolved_question": str(raw_state.get("resolved_question") or diag.get("resolved_question") or user_query),
                    "answer": resp.answer,
                    "formatted_response": resp.formatted_response,
                    "is_safe_refusal": resp.is_safe_refusal,
                    "citations": resp.sources,
                    "latency_seconds": round(turn_latency, 2),
                    "context_char_count": len(str(raw_state.get("final_evidence_context") or "")),
                    "evidence_context_snippet": str(raw_state.get("final_evidence_context") or "")[:200],
                }
                turn_traces.append(turn_data)
                print(f"    Latency: {turn_latency:.2f}s | Citations: {len(resp.sources)} | Answer: {resp.answer[:120]}...")

            except Exception as exc:
                print(f"    ERROR on turn {turn_idx}: {exc}")
                turn_traces.append({
                    "turn_index": turn_idx,
                    "query": user_query,
                    "error": str(exc),
                    "latency_seconds": round(time.perf_counter() - t_start, 2),
                })

        item_result = {
            "id": q_id,
            "title": title,
            "group": group,
            "turn_count": len(turns),
            "turns": turn_traces,
            "final_answer": turn_traces[-1].get("answer", ""),
            "citations": turn_traces[-1].get("citations", []),
        }
        results.append(item_result)

        with open(traces_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nPhase 14 Completed! Saved {len(results)} traces to {traces_file}")
    return results


if __name__ == "__main__":
    out_dir = WORKSPACE_ROOT / "test_outputs" / "rag_failure_diagnosis_20261019"
    run_student_eval(out_dir)
