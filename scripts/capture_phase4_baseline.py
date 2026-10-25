"""Script to capture Phase 4 baseline before any code changes.

Executes all 11 scenarios (12 conversational turns) against the active
canonical runtime (assistant_service -> LangGraph -> Ollama qwen2.5:3b),
saving comprehensive structured traces and human-readable markdown summary.
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

OUT_DIR = WORKSPACE_ROOT / "test_outputs" / "rag_stabilization_baseline_20261018"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_candidate_summary(cand_list):
    res = []
    for c in cand_list or []:
        source = str(getattr(c, "source", None) or getattr(getattr(c, "document", None), "source", "") or "")
        heading = str(getattr(c, "heading", "") or getattr(getattr(c, "document", None), "heading", "") or "")
        score = float(getattr(c, "score", 0.0) or 0.0)
        res.append({
            "source": source.replace("\\", "/").split("/")[-1],
            "full_source": source.replace("\\", "/"),
            "heading": heading,
            "score": round(score, 4),
        })
    return res


def run_turn(scenario_id: str, turn_desc: str, question: str, history: list[dict] | None = None) -> dict:
    print(f"\n[{scenario_id}] {turn_desc}...")
    print(f"  Question: '{question}'")
    if history:
        print(f"  History turns: {len(history)}")

    t0 = time.perf_counter()
    req = ChatTurnRequest(
        question=question,
        chat_history=tuple(history or []),
        session_id=scenario_id,
        institution_id="iitj",
    )
    resp = handle_chat_turn(req)
    total_latency = time.perf_counter() - t0

    diag = resp.diagnostics or {}
    resolved_q = diag.get("resolved_question") or question
    qp = diag.get("query_plan")
    route = getattr(qp, "route", "unknown") if qp else "unknown"
    is_multi = bool(diag.get("is_multi_intent"))

    # Evidence and ranking
    dense_hits = []
    bm25_hits = []
    ret_results = diag.get("retrieval_results") or ()
    if len(ret_results) > 0 and isinstance(ret_results[0], (list, tuple)):
        dense_hits = extract_candidate_summary(ret_results[0][:5])
    if len(ret_results) > 1 and isinstance(ret_results[1], (list, tuple)):
        bm25_hits = extract_candidate_summary(ret_results[1][:5])

    fused = extract_candidate_summary(diag.get("fused_candidates", [])[:5])
    verified = extract_candidate_summary(diag.get("verified_candidates", [])[:5])
    ranked = extract_candidate_summary(diag.get("ranked_candidates", [])[:5])
    context = str(diag.get("final_evidence_context") or "")

    print(f"  Resolved: '{resolved_q}' (Route: {route}, Multi: {is_multi})")
    print(f"  Latency: {total_latency:.2f}s | Citations: {len(resp.sources)}")
    print(f"  Answer preview: {resp.formatted_response[:120]}...")

    return {
        "scenario_id": scenario_id,
        "turn_description": turn_desc,
        "question": question,
        "history": history or [],
        "resolved_question": resolved_q,
        "route": route,
        "is_multi_intent": is_multi,
        "dense_top5": dense_hits,
        "bm25_top5": bm25_hits,
        "fused_top5": fused,
        "verified_top5": verified,
        "ranked_top5": ranked,
        "context_char_count": len(context),
        "context": context,
        "answer": resp.answer,
        "formatted_response": resp.formatted_response,
        "sources": resp.sources,
        "is_safe_refusal": resp.is_safe_refusal,
        "evidence_status": resp.evidence_status,
        "latency_seconds": round(total_latency, 2),
    }


def main():
    print("=== CAPTURING PHASE 4 BASELINE (11 SCENARIOS / 12 TURNS) ===")
    traces = []

    # Scenario 1: Graduation and academic requirements
    t1 = run_turn(
        "scenario_1",
        "Graduation and academic requirements",
        "What are the graduation requirements, minimum CPI criteria, and course registration guidelines at IIT Jodhpur?",
        history=None,
    )
    traces.append(t1)

    # Scenario 2: M.Tech fee
    t2 = run_turn(
        "scenario_2",
        "M.Tech tuition fee per semester",
        "What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. programme at IIT Jodhpur?",
        history=None,
    )
    traces.append(t2)

    # Scenario 3: Sponsored-student follow-up (same conversation as Scenario 2)
    s2_history = [
        {"role": "user", "content": t2["question"]},
        {"role": "assistant", "content": t2["answer"]},
    ]
    t3 = run_turn(
        "scenario_3",
        "Sponsored-student fee follow-up",
        "What about sponsored students?",
        history=s2_history,
    )
    traces.append(t3)

    # Scenario 4: Hostel and mess follow-up (continuing the same conversation)
    s3_history = list(s2_history) + [
        {"role": "user", "content": t3["question"]},
        {"role": "assistant", "content": t3["answer"]},
    ]
    t4 = run_turn(
        "scenario_4",
        "Hostel and mess fee inclusion follow-up",
        "Does that include hostel and mess charges?",
        history=s3_history,
    )
    traces.append(t4)

    # Scenario 5: Electrical Engineering research
    t5 = run_turn(
        "scenario_5",
        "Electrical Engineering research areas",
        "What research areas are available in Electrical Engineering?",
        history=None,
    )
    traces.append(t5)

    # Scenario 6: Topic transition and ambiguous reference (after Scenario 5)
    s5_history = [
        {"role": "user", "content": t5["question"]},
        {"role": "assistant", "content": t5["answer"]},
    ]
    t6 = run_turn(
        "scenario_6",
        "Topic transition: ambiguous eligibility for this programme",
        "What are the eligibility criteria for this programme?",
        history=s5_history,
    )
    traces.append(t6)

    # Scenario 7: Campus dining (Turn 1: Where can I find food?)
    t7a = run_turn(
        "scenario_7a",
        "Campus dining location inquiry",
        "Where can I find food on campus?",
        history=None,
    )
    traces.append(t7a)

    # Scenario 7: Campus dining (Turn 2: Tell me about the mess)
    s7_history = [
        {"role": "user", "content": t7a["question"]},
        {"role": "assistant", "content": t7a["answer"]},
    ]
    t7b = run_turn(
        "scenario_7b",
        "Campus dining: tell me about the mess follow-up",
        "Tell me about the mess.",
        history=s7_history,
    )
    traces.append(t7b)

    # Scenario 8: Admissions and reporting
    t8 = run_turn(
        "scenario_8",
        "Admissions, reporting, verification, orientation",
        "Provide information about admissions, physical reporting, document verification, orientation, and joining formalities at IIT Jodhpur.",
        history=None,
    )
    traces.append(t8)

    # Scenario 9: Fresh-session ambiguity
    t9 = run_turn(
        "scenario_9",
        "Fresh-session unanchored fee inquiry",
        "What is the fee for this programme?",
        history=None,
    )
    traces.append(t9)

    # Scenario 10: Unsupported claim and citations
    t10 = run_turn(
        "scenario_10",
        "Unsupported Supreme Court degree cancellation claim",
        "Why did the Supreme Court order IIT Jodhpur to cancel all degrees awarded in 2024?",
        history=None,
    )
    traces.append(t10)

    # Scenario 11: Structured navigation
    t11 = run_turn(
        "scenario_11",
        "Structured navigation: Lecture Hall Complex",
        "Where is the Lecture Hall Complex?",
        history=None,
    )
    traces.append(t11)

    # Save structured JSON
    json_path = OUT_DIR / "baseline_traces.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    print(f"\n[DONE] Saved JSON traces to: {json_path}")

    # Generate Markdown Summary
    md_lines = [
        "# RAG Stabilization Acceptance Suite — Before-Change Baseline Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Runtime Configuration:** Local Ollama (`qwen2.5:3b`), nomic-embed-text, Chroma collection `iitj_v1`",
        "**Target Suite:** Phase 4 Acceptance Scenarios (11 Scenarios / 12 Conversational Turns)",
        "",
        "---",
        "",
        "## 1. Scenario Summary Matrix",
        "",
        "| Scenario | Description | Resolved Query | Latency | Refusal? | Citations | Answer Snapshot |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for t in traces:
        ans_snap = t["formatted_response"].replace("\n", " ")[:90] + "..."
        md_lines.append(
            f"| `{t['scenario_id']}` | {t['turn_description']} | {t['resolved_question'][:45]}... | "
            f"{t['latency_seconds']}s | {t['is_safe_refusal']} | {len(t['sources'])} | {ans_snap} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Detailed Per-Scenario Baseline Logs",
        "",
    ])

    for t in traces:
        md_lines.extend([
            f"### {t['scenario_id']} — {t['turn_description']}",
            f"- **User Question:** *\"{t['question']}\"*",
            f"- **Resolved Query:** `{t['resolved_question']}`",
            f"- **Route:** `{t['route']}` | **Multi-Intent:** `{t['is_multi_intent']}`",
            f"- **Latency:** `{t['latency_seconds']}s` | **Refusal:** `{t['is_safe_refusal']}`",
            f"- **Sources Emitted ({len(t['sources'])}):**",
        ])
        if t["sources"]:
            for s in t["sources"]:
                md_lines.append(f"  * `{s.get('source', '')}`: {s.get('heading', '')}")
        else:
            md_lines.append("  * *(None emitted)*")

        md_lines.extend([
            f"- **Generated Response:**",
            "> " + t["formatted_response"].replace("\n", "\n> "),
            "",
            f"- **Context Length:** `{t['context_char_count']} chars`",
            "",
            "---",
            "",
        ])

    md_path = OUT_DIR / "baseline_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[DONE] Saved Markdown summary to: {md_path}")


if __name__ == "__main__":
    main()
