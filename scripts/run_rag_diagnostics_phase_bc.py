"""Comprehensive RAG Diagnostic & Tracing Script for Phases B and C.

Executes the 5 critical failure sequences through the canonical assistant service,
captures every diagnostic stage from graph state, and writes machine-readable traces
and human-readable reports to test_outputs/rag_diagnostics_20261018/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_platform.service.assistant_service import (
    ChatTurnRequest,
    ChatTurnResponse,
    handle_chat_turn,
)

OUTPUT_DIR = PROJECT_ROOT / "test_outputs" / "rag_diagnostics_20261018"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_candidate(c: Any) -> dict[str, Any]:
    doc = getattr(c, "document", c)
    src = getattr(doc, "metadata", {}).get("source", "") if hasattr(doc, "metadata") else getattr(c, "source", "")
    text = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else getattr(c, "text", str(c))
    score = getattr(c, "score", 0.0)
    doc_id = getattr(c, "document_id", getattr(doc, "metadata", {}).get("document_id", ""))
    return {
        "document_id": str(doc_id)[:40],
        "source": str(src),
        "score": round(float(score or 0.0), 4),
        "excerpt": str(text)[:180].replace("\n", " ").strip(),
    }


def _extract_turn_diagnostics(
    turn_num: int,
    session_id: str,
    query: str,
    history: Sequence[dict[str, str]],
    resp: ChatTurnResponse,
    start_time: float,
) -> dict[str, Any]:
    elapsed = resp.latency_seconds or (time.perf_counter() - start_time)
    diag = resp.diagnostics or {}

    # Dense & Keyword counts
    raw_retrievals = diag.get("retrieval_results", ())
    dense_count = len(raw_retrievals[0]) if len(raw_retrievals) > 0 else 0
    bm25_count = len(raw_retrievals[1]) if len(raw_retrievals) > 1 else 0

    # Fused candidates
    fused = diag.get("fused_candidates", ())
    fused_serialized = [_serialize_candidate(c) for c in fused[:10]]

    # Verified & Rejected
    verified = diag.get("verified_candidates", ())
    verified_serialized = [_serialize_candidate(c) for c in verified]
    rejected = diag.get("rejected_candidates", ())
    rejected_serialized = [_serialize_candidate(c) for c in rejected[:10]]

    # Packed evidence units
    units = diag.get("evidence_units", ())
    units_serialized = [
        {
            "evidence_id": str(getattr(u, "evidence_id", u.get("evidence_id") if isinstance(u, dict) else "")),
            "source": str(getattr(u, "source", u.get("source") if isinstance(u, dict) else "")),
            "heading": str(getattr(u, "heading", u.get("heading") if isinstance(u, dict) else "")),
            "text_excerpt": str(getattr(u, "text", u.get("text") if isinstance(u, dict) else ""))[:180].replace("\n", " ").strip(),
        }
        for u in units
    ]

    # Model generation & finish reason
    ans_gen = diag.get("answer_generation")
    finish_reason = getattr(ans_gen, "finish_reason", None) if ans_gen else None

    # Rendered sources in response
    rendered_sources = [
        {
            "evidence_id": s.get("evidence_id", ""),
            "source": s.get("source", ""),
            "heading": s.get("heading", ""),
            "excerpt": s.get("text", "")[:120].replace("\n", " ").strip(),
        }
        for s in resp.sources
    ]

    return {
        "turn": turn_num,
        "session_id": session_id,
        "original_query": query,
        "history_supplied": list(history),
        "resolved_question": diag.get("resolved_question", query),
        "active_entity": diag.get("active_entity"),
        "active_topic": diag.get("active_topic"),
        "is_multi_intent": bool(diag.get("is_multi_intent")),
        "retrieval_queries": list(diag.get("retrieval_queries", ())),
        "dense_candidates_count": dense_count,
        "bm25_candidates_count": bm25_count,
        "fused_top10": fused_serialized,
        "verified_count": len(verified),
        "verified_candidates": verified_serialized,
        "rejected_count": len(rejected),
        "rejected_candidates": rejected_serialized,
        "packed_units_count": len(units_serialized),
        "packed_evidence_units": units_serialized,
        "sources_rendered_count": len(rendered_sources),
        "sources_rendered": rendered_sources,
        "is_safe_refusal": resp.is_safe_refusal,
        "evidence_status": resp.evidence_status,
        "finish_reason": finish_reason,
        "latency_seconds": round(elapsed, 2),
        "answer": resp.answer,
        "formatted_response": resp.formatted_response,
        "suggested_questions": resp.suggested_questions,
    }


def run_sequence(
    seq_id: str,
    title: str,
    turns: list[str],
    session_id: str,
) -> dict[str, Any]:
    print(f"\n==================================================")
    print(f"Running {seq_id}: {title} (Session: {session_id})")
    print(f"==================================================")

    history: list[dict[str, str]] = []
    turn_records: list[dict[str, Any]] = []

    for idx, q in enumerate(turns, start=1):
        print(f"\n>>> Turn {idx}: {q}")
        t0 = time.perf_counter()
        req = ChatTurnRequest(
            question=q,
            chat_history=tuple(history),
            session_id=session_id,
            institution_id="iitj",
        )
        resp = handle_chat_turn(req)
        t_elapsed = time.perf_counter() - t0

        record = _extract_turn_diagnostics(idx, session_id, q, history, resp, t0)
        turn_records.append(record)

        print(f"Latency: {round(t_elapsed, 2)}s | Refusal: {resp.is_safe_refusal} | Sources: {len(resp.sources)}")
        print(f"Resolved Query: {record['resolved_question']}")
        print(f"Answer:\n{resp.answer[:300]}...")

        # Update history
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": resp.answer})

    return {
        "sequence_id": seq_id,
        "title": title,
        "session_id": session_id,
        "turns": turn_records,
    }


def main():
    start_total = time.perf_counter()
    results = {}

    # Sequence 1: Programme-specific fees
    results["sequence_1"] = run_sequence(
        "sequence_1",
        "Programme-specific fees (M.Tech)",
        [
            "What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. programme at IIT Jodhpur?",
            "What about sponsored students?",
            "Does that include hostel and mess charges?",
        ],
        session_id="session_seq_1_fees",
    )

    # Sequence 2: Campus dining
    results["sequence_2"] = run_sequence(
        "sequence_2",
        "Campus dining & mess facilities",
        [
            "Where can I find food on campus?",
            "Tell me about the mess.",
        ],
        session_id="session_seq_2_dining",
    )

    # Sequence 3: Electrical Engineering research
    results["sequence_3"] = run_sequence(
        "sequence_3",
        "Electrical Engineering research areas",
        [
            "What research areas are available in Electrical Engineering?",
        ],
        session_id="session_seq_3_ee",
    )

    # Sequence 4: Unsupported premise
    results["sequence_4"] = run_sequence(
        "sequence_4",
        "Unsupported premise (Supreme Court degree cancellation)",
        [
            "Why did the Supreme Court order IIT Jodhpur to cancel all degrees awarded in 2024?",
        ],
        session_id="session_seq_4_sc_premise",
    )

    # Sequence 5: Context isolation
    results["sequence_5"] = run_sequence(
        "sequence_5",
        "Context isolation (Fresh session vague question)",
        [
            "What is the fee for this programme?",
        ],
        session_id="session_seq_5_fresh_isolation",
    )

    total_time = round(time.perf_counter() - start_total, 2)
    results["metadata"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_latency_seconds": total_time,
        "model": "qwen2.5:3b",
    }

    # Save JSON trace
    json_path = OUTPUT_DIR / "traces.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Wrote machine-readable trace to: {json_path}")

    # Generate Markdown report
    md_path = OUTPUT_DIR / "diagnostic_report.md"
    generate_markdown_report(results, md_path)
    print(f"[OK] Wrote human-readable report to: {md_path}")


def generate_markdown_report(results: dict[str, Any], out_path: Path):
    lines = [
        "# RAG Diagnostic Forensic Report — Phase B & C Trace",
        "",
        f"- **Model**: `{results.get('metadata', {}).get('model', 'qwen2.5:3b')}`",
        f"- **Timestamp**: `{results.get('metadata', {}).get('timestamp')}`",
        f"- **Total Latency**: `{results.get('metadata', {}).get('total_latency_seconds')}s`",
        "",
        "---",
        "",
        "## Executive Summary of Observed Failures",
        "",
        "| Sequence | Turn | User Query | Key Defect Observed | Latency |",
        "|---|---|---|---|---|",
    ]

    for seq_key in ["sequence_1", "sequence_2", "sequence_3", "sequence_4", "sequence_5"]:
        seq = results.get(seq_key, {})
        for t in seq.get("turns", []):
            ans_snip = t["answer"][:100].replace("\n", " ").strip()
            lines.append(
                f"| `{seq.get('sequence_id')}` | T{t['turn']} | {t['original_query']} | {ans_snip}... | {t['latency_seconds']}s |"
            )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Stage-by-Stage Traces")
    lines.append("")

    for seq_key in ["sequence_1", "sequence_2", "sequence_3", "sequence_4", "sequence_5"]:
        seq = results.get(seq_key, {})
        lines.append(f"### {seq.get('sequence_id')}: {seq.get('title')}")
        lines.append(f"- **Session ID**: `{seq.get('session_id')}`")
        lines.append("")

        for t in seq.get("turns", []):
            lines.append(f"#### Turn {t['turn']}: *\"{t['original_query']}\"*")
            lines.append(f"- **Resolved Question**: `{t['resolved_question']}`")
            lines.append(f"- **Active Entity**: `{t['active_entity']}` | **Active Topic**: `{t['active_topic']}`")
            lines.append(f"- **Dense Candidates**: {t['dense_candidates_count']} | **BM25 Candidates**: {t['bm25_candidates_count']}")
            lines.append(f"- **Verified Candidates**: {t['verified_count']} | **Rejected Candidates**: {t['rejected_count']}")
            lines.append(f"- **Packed Units**: {t['packed_units_count']} | **Rendered Sources**: {t['sources_rendered_count']}")
            lines.append(f"- **Safe Refusal**: `{t['is_safe_refusal']}` | **Status**: `{t['evidence_status']}` | **Latency**: `{t['latency_seconds']}s`")
            lines.append("")
            lines.append("**Top Verified Candidates Packed into Context:**")
            lines.append("")
            for u in t.get("packed_evidence_units", [])[:5]:
                lines.append(f"- `[{u['source']} | {u['heading']}]`: {u['text_excerpt']}")
            lines.append("")
            lines.append("**Final Answer Generated:**")
            lines.append("")
            lines.append(f"> {t['answer']}")
            lines.append("")
            lines.append(f"- **Sources Displayed in UI ({t['sources_rendered_count']})**:")
            for s in t.get("sources_rendered", [])[:5]:
                lines.append(f"  * `{s['source']}` - {s['heading']}")
            if t['sources_rendered_count'] > 5:
                lines.append(f"  * *(and {t['sources_rendered_count'] - 5} more...)*")
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
