#!/usr/bin/env python3
"""Run the Phase 15 10-Question Isolated Held-Out Evaluation against the canonical runtime."""

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

HELD_OUT_QUESTIONS = [
    {
        "id": "H01",
        "title": "Civil and Infrastructure Research Thrusts",
        "turns": [
            "What are the major research thrusts in the Department of Civil and Infrastructure Engineering?"
        ],
        "category": "unseen_department",
    },
    {
        "id": "H02",
        "title": "Chemical Engineering Dual Degree",
        "turns": [
            "Does the Chemical Engineering department offer a double B.Tech. degree with Engineering Science?"
        ],
        "category": "unseen_curriculum",
    },
    {
        "id": "H03",
        "title": "IDRP IoT Facilities & Thrusts",
        "turns": [
            "What are the key facilities and research themes available at the IDRP on IoT and Applications?"
        ],
        "category": "research_platform",
    },
    {
        "id": "H04",
        "title": "Course Dropping Procedure",
        "turns": [
            "What is the procedure and deadline for dropping a course during the regular semester?"
        ],
        "category": "academic_progression",
    },
    {
        "id": "H05",
        "title": "M.Sc. in Physics Curriculum & Duration",
        "turns": [
            "What is the duration and core focus of the M.Sc. in Physics curriculum?"
        ],
        "category": "postgraduate_offering",
    },
    {
        "id": "H06",
        "title": "Quantum Technologies & Platform Follow-Up",
        "turns": [
            "Which departments offer programmes in Quantum Technologies?",
            "Is there a specific research platform dedicated to quantum information?",
        ],
        "category": "conversational_followup",
    },
    {
        "id": "H07",
        "title": "Registration Kept Alive (RKA) Rules & Limit",
        "turns": [
            "What are the rules for Registration Kept Alive (RKA) for postgraduate students?",
            "How many maximum semesters can RKA be maintained?",
        ],
        "category": "conversational_followup",
    },
    {
        "id": "H08",
        "title": "Undergraduate Refundable Deposit",
        "turns": [
            "According to the official fee table, what is the one-time refundable deposit for new undergraduate admissions?"
        ],
        "category": "table_numeric",
    },
    {
        "id": "H09",
        "title": "External Out-of-Scope Bus Timings",
        "turns": [
            "What are the exact bus timings from IIT Jodhpur to Jaipur railway station on Sundays?"
        ],
        "category": "out_of_scope_abstention",
    },
    {
        "id": "H10",
        "title": "Non-GATE M.Tech. Assistantship Rule",
        "turns": [
            "Can a student who has not qualified GATE receive the MHRD teaching assistantship in regular M.Tech.?"
        ],
        "category": "policy_constraint",
    },
]


def run_held_out_eval(output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_file = output_dir / "held_out_10q_traces.json"

    print("=== Starting Phase 15: 10 Isolated Held-Out Question Evaluation ===")
    results = []

    for idx, item in enumerate(HELD_OUT_QUESTIONS, 1):
        q_id = item["id"]
        title = item["title"]
        category = item["category"]
        turns = item["turns"]

        print(f"\n[{idx}/10] Running {q_id}: {title} ({len(turns)} turns)")
        session_id = f"held_out_eval_{q_id.lower()}_{int(time.time())}"
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
            "category": category,
            "turn_count": len(turns),
            "turns": turn_traces,
            "final_answer": turn_traces[-1].get("answer", ""),
            "citations": turn_traces[-1].get("citations", []),
        }
        results.append(item_result)

        with open(traces_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nPhase 15 Completed! Saved {len(results)} traces to {traces_file}")
    return results


if __name__ == "__main__":
    out_dir = WORKSPACE_ROOT / "test_outputs" / "rag_failure_diagnosis_20261019"
    run_held_out_eval(out_dir)
