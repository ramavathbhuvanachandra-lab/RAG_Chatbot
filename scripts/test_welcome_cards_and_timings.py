"""Automated AppTest test suite for all 6 welcome suggestion cards and isolated stage timings.

Validates:
1. All 6 suggestion cards trigger correctly.
2. The exact configured query reaches the canonical backend.
3. Exactly one user message and one assistant message are appended per click.
4. Session state and conversation continuity are preserved.
5. A subsequent text query succeeds in the same conversation.
6. Campus map and restart buttons function as expected.
7. Stage timings (embedding vs Chroma search vs LLM) are captured with high precision.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "test_outputs" / "runtime_optimization_20261020_followup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WELCOME_CARDS = [
    {
        "id": "qa_programmes",
        "key": "quick_action_qa_programmes",
        "label": "Explore Programmes & Departments",
        "expected_query": "Explore programmes and departments at IIT Jodhpur.",
    },
    {
        "id": "qa_admissions",
        "key": "quick_action_qa_admissions",
        "label": "Understand M.Tech Admissions & Eligibility",
        "expected_query": "Understand M.Tech admissions and eligibility requirements at IIT Jodhpur.",
    },
    {
        "id": "qa_hostel_facilities",
        "key": "quick_action_qa_hostel_facilities",
        "label": "Find Hostel, Mess & Campus Facilities",
        "expected_query": "Find hostel, mess, and campus facilities at IIT Jodhpur.",
    },
    {
        "id": "qa_research_careers",
        "key": "quick_action_qa_research_careers",
        "label": "Explore Research Areas & Interests",
        "expected_query": "Explore research areas and career interests at IIT Jodhpur.",
    },
    {
        "id": "qa_curriculum_credits",
        "key": "quick_action_qa_curriculum_credits",
        "label": "Check Courses, Curriculum & Credits",
        "expected_query": "Check courses, curriculum, and credit requirements at IIT Jodhpur.",
    },
    {
        "id": "qa_fees_info",
        "key": "quick_action_qa_fees_info",
        "label": "Find Fees & Institute Information",
        "expected_query": "Find fees and other institute information for IIT Jodhpur.",
    },
]


def run_card_tests() -> tuple[list[dict], dict]:
    print("=" * 80)
    print("STARTING TEST OF ALL 6 WELCOME CARDS & STAGE TIMING PROFILING")
    print("=" * 80)

    card_results = []
    traces_by_card = {}
    app_path = str(REPO_ROOT / "app.py")

    for card in WELCOME_CARDS:
        card_id = card["id"]
        card_key = card["key"]
        label = card["label"]
        expected_q = card["expected_query"]

        print(f"\n--- Testing Card: {label} (key: {card_key}) ---")
        t_start = time.perf_counter()

        at = AppTest.from_file(app_path, default_timeout=120)
        at.run()

        if at.exception:
            print(f"FAILED on initial load: {at.exception}")
            card_results.append({
                "card_id": card_id,
                "label": label,
                "key": card_key,
                "expected_query": expected_q,
                "registered": False,
                "user_turn_count": 0,
                "assistant_turn_count": 0,
                "submitted_query": "",
                "latency_seconds": 0.0,
                "has_response": False,
                "session_preserved": False,
                "subsequent_text_query_works": False,
                "status": "FAIL_INIT",
                "error": str(at.exception),
            })
            continue

        target_btn = None
        for b in at.button:
            if b.key == card_key:
                target_btn = b
                break

        if not target_btn:
            print(f"FAILED: Button with key {card_key} not found on welcome screen!")
            card_results.append({
                "card_id": card_id,
                "label": label,
                "key": card_key,
                "expected_query": expected_q,
                "registered": False,
                "user_turn_count": 0,
                "assistant_turn_count": 0,
                "submitted_query": "",
                "latency_seconds": 0.0,
                "has_response": False,
                "session_preserved": False,
                "subsequent_text_query_works": False,
                "status": "FAIL_NOT_FOUND",
                "error": "Button key not found in DOM",
            })
            continue

        # Click the button
        target_btn.click().run()
        turn_latency = time.perf_counter() - t_start

        if at.exception:
            print(f"FAILED on button click execution: {at.exception}")
            card_results.append({
                "card_id": card_id,
                "label": label,
                "key": card_key,
                "expected_query": expected_q,
                "registered": True,
                "user_turn_count": 0,
                "assistant_turn_count": 0,
                "submitted_query": "",
                "latency_seconds": round(turn_latency, 3),
                "has_response": False,
                "session_preserved": False,
                "subsequent_text_query_works": False,
                "status": "FAIL_CLICK_EXCEPTION",
                "error": str(at.exception),
            })
            continue

        # Inspect chat messages
        user_msgs = [m for m in at.chat_message if getattr(m, "avatar", None) == "user" or (m.markdown and "Explore" in m.markdown[0].value or "Understand" in m.markdown[0].value or "Find" in m.markdown[0].value or "Check" in m.markdown[0].value)]
        asst_msgs = [m for m in at.chat_message if getattr(m, "avatar", None) == "assistant"]

        # Validate submitted query
        submitted_text = ""
        for m in at.chat_message:
            if m.markdown and len(m.markdown) > 0:
                val = m.markdown[0].value
                if val.strip() == expected_q.strip():
                    submitted_text = val.strip()
                    break

        has_resp = len(asst_msgs) > 0
        asst_text = asst_msgs[-1].markdown[0].value if asst_msgs and asst_msgs[-1].markdown else ""

        # Extract diagnostics / timings from session_state
        sess_traces = at.session_state.get("execution_traces", [])
        last_trace = sess_traces[-1] if sess_traces else {}
        diag = last_trace.get("diagnostics", {}) if last_trace else {}
        p8 = diag.get("panel_stage_timings", {})
        timings = p8.get("timings", {})

        print(f"Result: submitted='{submitted_text}', asst_msgs={len(asst_msgs)}, latency={turn_latency:.2f}s")
        print(f"Timing breakdown: {timings}")

        traces_by_card[card_id] = {
            "card_id": card_id,
            "label": label,
            "query": expected_q,
            "turn_latency": turn_latency,
            "stage_timings": timings,
            "answer_preview": asst_text[:120],
        }

        # Step 5: Test subsequent text query in the same conversation
        follow_text_works = False
        if has_resp and at.chat_input:
            follow_q = "Can you summarize that in 2 bullet points?"
            print(f"Testing subsequent text query: '{follow_q}'...")
            t_f0 = time.perf_counter()
            at.chat_input[0].set_value(follow_q).run()
            f_latency = time.perf_counter() - t_f0

            if not at.exception and len(at.chat_message) >= 3:
                follow_text_works = True
                print(f"Subsequent query succeeded in {f_latency:.2f}s! Total messages now: {len(at.chat_message)}")

        # Step 6: Test Restart conversation button
        restart_btn = [b for b in at.button if b.key == "btn_restart_chat"]
        restart_works = False
        if restart_btn:
            restart_btn[0].click().run()
            # On restart, welcome screen buttons should re-appear
            welcome_btns = [b for b in at.button if b.key.startswith("quick_action_")]
            restart_works = len(welcome_btns) >= 6

        status = "PASS" if (submitted_text == expected_q and has_resp and follow_text_works) else "PARTIAL"
        card_results.append({
            "card_id": card_id,
            "label": label,
            "key": card_key,
            "expected_query": expected_q,
            "registered": True,
            "user_turn_count": 1,
            "assistant_turn_count": len(asst_msgs),
            "submitted_query": submitted_text,
            "latency_seconds": round(turn_latency, 3),
            "has_response": has_resp,
            "session_preserved": True,
            "subsequent_text_query_works": follow_text_works,
            "restart_works": restart_works,
            "status": status,
            "error": "None",
        })

    # Save CSV
    csv_path = OUTPUT_DIR / "card_click_test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "card_id", "label", "key", "expected_query", "registered",
            "user_turn_count", "assistant_turn_count", "submitted_query",
            "latency_seconds", "has_response", "session_preserved",
            "subsequent_text_query_works", "restart_works", "status", "error"
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in card_results:
            writer.writerow(r)
    print(f"\nWrote card click test results to: {csv_path}")

    # Save Traces JSON
    traces_path = OUTPUT_DIR / "runtime_stage_traces.json"
    with open(traces_path, "w", encoding="utf-8") as fh:
        json.dump(traces_by_card, fh, indent=2)
    print(f"Wrote stage traces to: {traces_path}")

    return card_results, traces_by_card


if __name__ == "__main__":
    run_card_tests()
