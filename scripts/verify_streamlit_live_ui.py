#!/usr/bin/env python3
"""Verify Phase 16: Six representative conversations through Streamlit frontend integration.

Tests:
1. Source-specific factual question (Q1: Aerospace Vision)
2. Question involving numbers or table data (Q20: Institute UG credit structure)
3. Cross-department or programme-specific question (Q30: Electronics vs Materials)
4. Multi-part question (Q11: GeM training dates, duration, mode, audience, modules)
5. Conversational follow-up with history (S15: Materials degrees -> minor options)
6. Genuinely unsupported or ambiguous question (Q34: Ambiguous graduation credit requirement)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Ensure streamlit session state mock works cleanly outside browser if imported
import streamlit as st

class SessionStateMock(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
    def __setattr__(self, key, value):
        self[key] = value

from UI.chat import get_response, initialize_chat, create_new_chat

TEST_CONVERSATIONS = [
    {
        "id": "UI_01",
        "category": "Source-specific factual question",
        "description": "Aerospace Vision and aims (Q1)",
        "turns": [
            "What is the vision of IIT Jodhpur's Aerospace Engineering Department? Explain its three main aims in your own words."
        ],
    },
    {
        "id": "UI_02",
        "category": "Number / table data",
        "description": "Institute UG credit structure (Q20)",
        "turns": [
            "According to the Mathematics General Information document's Institute UG structure table, what is the total graded credit count, the total credit count for the degree, and the difference between them? Identify the non-graded components that account for that difference."
        ],
    },
    {
        "id": "UI_03",
        "category": "Cross-department / comparison question",
        "description": "Electronics vs Materials research themes (Q30)",
        "turns": [
            "Compare Electronics Engineering and Materials Engineering research themes. Identify at least one meaningful overlap and three areas that distinguish the departments, using only the uploaded documents."
        ],
    },
    {
        "id": "UI_04",
        "category": "Multi-part question",
        "description": "GeM training dates, mode, audience, modules (Q11)",
        "turns": [
            "Describe the Specialized Training in Government e-Marketplace (GeM). Include its dates, duration, mode, intended audience, and three hands-on modules."
        ],
    },
    {
        "id": "UI_05",
        "category": "Conversational follow-up with history",
        "description": "Materials Engineering degrees -> minor options (S15)",
        "turns": [
            "What degree programmes does the Department of Materials Engineering offer?",
            "What about minor options for B.Tech. students in that department?",
        ],
    },
    {
        "id": "UI_06",
        "category": "Genuinely unsupported / ambiguous question",
        "description": "Ambiguous graduation credit query (Q34)",
        "turns": [
            "How many credits do I need to graduate from IIT Jodhpur? Give the exact number."
        ],
    },
]


def run_ui_verification(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "streamlit_live_ui_acceptance.json"

    print("=== Starting Phase 16: Streamlit UI Integration Verification ===")
    results = []

    for conv_idx, conv in enumerate(TEST_CONVERSATIONS, 1):
        conv_id = conv["id"]
        cat = conv["category"]
        desc = conv["description"]
        turns = conv["turns"]

        print(f"\n[{conv_idx}/6] Running Conversation {conv_id}: {desc} ({cat})")

        # Setup fresh active chat session state
        chat_id = f"chat_{conv_id.lower()}_{int(time.time())}"
        session_id = f"session_{conv_id.lower()}_{int(time.time())}"
        mock_chat = {
            "title": f"Chat {conv_id}",
            "session_id": session_id,
            "messages": [],
        }
        mock_state = SessionStateMock({
            "active_chat": chat_id,
            "conversations": {chat_id: mock_chat},
        })

        conv_turns_data = []

        with patch.object(st, "session_state", mock_state):
            for turn_idx, question in enumerate(turns, 1):
                print(f"  Turn {turn_idx}: {question[:80]}...")
                t_start = time.perf_counter()

                try:
                    formatted_answer, latency = get_response(question)
                    elapsed = time.perf_counter() - t_start

                    # Append to messages as Streamlit UI does
                    mock_chat["messages"].append({
                        "role": "user",
                        "content": question,
                    })
                    mock_chat["messages"].append({
                        "role": "assistant",
                        "content": formatted_answer,
                        "response_time": latency,
                    })

                    turn_rec = {
                        "turn": turn_idx,
                        "question": question,
                        "formatted_answer": formatted_answer,
                        "latency_reported": round(latency, 2) if latency else None,
                        "wall_clock_seconds": round(elapsed, 2),
                        "has_sources": "Sources:" in formatted_answer or "Citations:" in formatted_answer,
                        "status": "PASS",
                    }
                    conv_turns_data.append(turn_rec)
                    print(f"    Latency: {elapsed:.2f}s | Answer: {formatted_answer[:120]}...")

                except Exception as exc:
                    print(f"    ERROR on turn {turn_idx}: {exc}")
                    conv_turns_data.append({
                        "turn": turn_idx,
                        "question": question,
                        "error": str(exc),
                        "status": "FAIL",
                    })

        results.append({
            "id": conv_id,
            "category": cat,
            "description": desc,
            "turns": conv_turns_data,
            "final_answer": conv_turns_data[-1].get("formatted_answer", ""),
            "overall_status": "PASS" if all(t.get("status") == "PASS" for t in conv_turns_data) else "FAIL",
        })

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nPhase 16 UI Acceptance Completed! Saved to {report_file}")
    return results


if __name__ == "__main__":
    out_dir = REPO_ROOT / "test_outputs" / "rag_failure_diagnosis_20261019"
    run_ui_verification(out_dir)
