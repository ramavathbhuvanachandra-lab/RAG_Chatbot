"""Baseline Retrieval Pipeline Audit Script.

Executes 15 representative evaluation queries across all required categories:
 1. Short factual lookup
 2. Natural-language standalone query
 3. Long paragraph containing multiple related requests
 4. Multi-intent comparison
 5. Follow-up referencing the correct prior turn
 6. Topic change after unrelated conversation history
 7. English query
 8. Hindi query
 9. Hinglish query
10. Exact course-code lookup
11. Numerical or tabular lookup
12. A query for evidence missing from current corpus
13. Similar terminology in two different academic programmes
14. Broad institute-wide programme discovery
15. A question with an explicit qualifier, date, number, or instruction not to guess

Outputs deliverables to test_outputs/retrieval_audit_20261020_master/:
- retrieval_baseline_matrix.csv
- query_understanding_handoff_traces.json
- hybrid_retrieval_traces.json
- ranking_verification_decisions.json
- final_llm_context_traces.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.clients.assistant_client import send_chat_turn

OUTPUT_DIR = REPO_ROOT / "test_outputs" / "retrieval_audit_20261020_master"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "id": "T01",
        "category": "1. Short factual lookup",
        "query": "What is the M.Tech tuition fee for regular Indian students per semester?",
        "history": [],
        "requested_intents": "M.Tech regular semester tuition fee amount for Indian nationals",
        "expected_evidence": "Fee structure table or tuition fee circular mentioning M.Tech regular semester fee (e.g. 15,000 INR).",
        "key_terms": ["tuition fee", "m.tech", "semester", "regular"],
        "expect_missing": False,
    },
    {
        "id": "T02",
        "category": "2. Natural-language standalone query",
        "query": "What research areas and laboratories are active in the Department of Electrical Engineering?",
        "history": [],
        "requested_intents": "Research thrust areas, specializations, and labs in Electrical Engineering",
        "expected_evidence": "Electrical Engineering department overview or research page mentioning VLSI, Microelectronics, Communications, Power Systems, etc.",
        "key_terms": ["electrical engineering", "research", "laborator"],
        "expect_missing": False,
    },
    {
        "id": "T03",
        "category": "3. Long paragraph containing multiple related requests",
        "query": "I am interested in applying for the M.Tech programme at IIT Jodhpur. Could you explain the minimum eligibility requirements, whether a valid GATE score is required for regular candidates, and what the tuition fee per semester is for regular Indian students?",
        "history": [],
        "requested_intents": "M.Tech eligibility requirements; GATE score requirement; Regular semester tuition fee",
        "expected_evidence": "M.Tech admission brochure/ordinance specifying qualifying degree/marks, GATE qualification, and fee schedule.",
        "key_terms": ["m.tech", "eligibility", "gate", "tuition fee"],
        "expect_missing": False,
    },
    {
        "id": "T04",
        "category": "4. Multi-intent comparison",
        "query": "Compare Electronics Engineering and Materials Engineering for chip design and semiconductor research.",
        "history": [],
        "requested_intents": "Electronics Engineering chip/VLSI focus vs Materials Engineering semiconductor materials/synthesis focus",
        "expected_evidence": "Curriculum/research evidence from both Electrical/Electronics (VLSI/circuits) and Materials Engineering (semiconductors/functional materials).",
        "key_terms": ["electronics", "materials", "chip design", "semiconductor"],
        "expect_missing": False,
    },
    {
        "id": "T05",
        "category": "5. Follow-up referencing the correct prior turn",
        "query": "Are SC and ST students exempt from paying it?",
        "history": [
            {"role": "user", "content": "What is the tuition fee for the regular M.Tech programme?"},
            {"role": "assistant", "content": "The tuition fee for regular Indian M.Tech students is 15,000 INR per semester."},
        ],
        "requested_intents": "Tuition fee waiver or exemption policy specifically for SC/ST students in M.Tech",
        "expected_evidence": "Tuition fee exemption policy for SC/ST students in postgraduate/M.Tech programmes.",
        "key_terms": ["sc/st", "sc and st", "exemption", "tuition fee", "waiver"],
        "expect_missing": False,
    },
    {
        "id": "T06",
        "category": "6. Topic change after unrelated conversation history",
        "query": "What is the Drone and Anti-Drone Technologies programme at IIT Jodhpur?",
        "history": [
            {"role": "user", "content": "Where is the Old Mess located on campus?"},
            {"role": "assistant", "content": "The Old Mess is located near the student residential hostel blocks."},
        ],
        "requested_intents": "Drone and Anti-Drone Technologies programme description and proposal",
        "expected_evidence": "Proposal or curriculum document for Drone and Anti-Drone Technologies programme (must not be contaminated by mess/dining).",
        "key_terms": ["drone", "anti-drone"],
        "expect_missing": False,
    },
    {
        "id": "T07",
        "category": "7. English query",
        "query": "In which year was IIT Jodhpur established and where was its temporary transit campus initially located?",
        "history": [],
        "requested_intents": "Year of establishment (2008) and transit campus location (MBM Engineering College)",
        "expected_evidence": "Institute overview or history document mentioning establishment in 2008 and MBM Engineering College campus.",
        "key_terms": ["2008", "established", "mbm", "transit campus"],
        "expect_missing": False,
    },
    {
        "id": "T08",
        "category": "8. Hindi query",
        "query": "क्या अनुसूचित जाति (SC) और अनुसूचित जनजाति (ST) के छात्रों को ट्यूशन फीस में छूट मिलती है?",
        "history": [],
        "requested_intents": "SC/ST tuition fee exemption inquiry in Hindi",
        "expected_evidence": "Fee structure document stating 100% tuition fee waiver for SC/ST candidates.",
        "key_terms": ["sc", "st", "tuition fee", "waiver", "exemption"],
        "expect_missing": False,
    },
    {
        "id": "T09",
        "category": "9. Hinglish query",
        "query": "IIT Jodhpur me MTech admission ke liye eligibility criteria kya hai aur GATE score zaroori hai kya?",
        "history": [],
        "requested_intents": "M.Tech admission eligibility and GATE requirement inquiry in Hinglish",
        "expected_evidence": "M.Tech admissions eligibility and GATE requirement criteria.",
        "key_terms": ["m.tech", "eligibility", "gate"],
        "expect_missing": False,
    },
    {
        "id": "T10",
        "category": "10. Exact course-code lookup",
        "query": "What exact course code is printed next to Quantum Computing in the Mathematics curriculum?",
        "history": [],
        "requested_intents": "Exact course alphanumeric code assigned to Quantum Computing course in Mathematics & Computing curriculum",
        "expected_evidence": "Mathematics curriculum course list table row containing 'Quantum Computing' with its specific code.",
        "key_terms": ["quantum computing", "mathematics"],
        "expect_missing": False,
    },
    {
        "id": "T11",
        "category": "11. Numerical or tabular lookup",
        "query": "What is the total credit requirement for the B.S. in Mathematics and Computing, and what are the listed credit components?",
        "history": [],
        "requested_intents": "Total credit count and category breakdown for B.S. Mathematics and Computing",
        "expected_evidence": "Curriculum structure table showing total credits and component breakdown (Science, Engineering, Core, Electives).",
        "key_terms": ["credits", "mathematics and computing", "b.s."],
        "expect_missing": False,
    },
    {
        "id": "T12",
        "category": "12. A query for evidence missing from current corpus",
        "query": "What were the exact terms of the Supreme Court degree cancellation order for IIT Jodhpur in 2024?",
        "history": [],
        "requested_intents": "Fictitious/missing event: Supreme Court degree cancellation order",
        "expected_evidence": "None (unsupported/hallucination bait; safe refusal or ungrounded abstention required).",
        "key_terms": ["supreme court", "degree cancellation"],
        "expect_missing": True,
    },
    {
        "id": "T13",
        "category": "13. Similar terminology in two different academic programmes",
        "query": "What are the academic progression and course overload rules for B.Tech students compared to M.Tech students?",
        "history": [],
        "requested_intents": "Progression/overload rules for B.Tech compared with M.Tech regulations",
        "expected_evidence": "Both B.Tech ordinances (overload credits/SGPA) and M.Tech ordinances distinguished cleanly.",
        "key_terms": ["progression", "overload", "b.tech", "m.tech"],
        "expect_missing": False,
    },
    {
        "id": "T14",
        "category": "14. Broad institute-wide programme discovery",
        "query": "What undergraduate and postgraduate degree programmes and academic departments are offered at IIT Jodhpur?",
        "history": [],
        "requested_intents": "Broad discovery of UG (B.Tech, B.S.) and PG (M.Tech, M.Sc., Ph.D.) degrees and major departments",
        "expected_evidence": "Academic programmes overview or directory of departments and schools.",
        "key_terms": ["b.tech", "m.tech", "departments", "programmes"],
        "expect_missing": False,
    },
    {
        "id": "T15",
        "category": "15. Explicit qualifier, date, number, or instruction not to guess",
        "query": "According to the official fee notification, what is the exact semester tuition fee for the ITEP programme for the academic year 2026-2027? Do not guess if it is not in the text.",
        "history": [],
        "requested_intents": "ITEP tuition fee for specific future AY 2026-2027 with negative constraint (do not guess)",
        "expected_evidence": "Fee notification mentioning AY 2026-2027 ITEP, or explicit safe abstention if future year is not in corpus.",
        "key_terms": ["itep", "2026-2027", "tuition fee"],
        "expect_missing": True,
    },
]


def run_baseline_evaluation():
    print("=" * 80)
    print("STARTING BASELINE RETRIEVAL PIPELINE EVALUATION (15 QUERIES)")
    print("=" * 80)

    handoff_traces = []
    hybrid_traces = []
    ranking_traces = []
    context_traces = []
    matrix_rows = []

    for idx, tc in enumerate(TEST_CASES, start=1):
        q_id = tc["id"]
        category = tc["category"]
        query = tc["query"]
        history = tc["history"]
        intents = tc["requested_intents"]
        expected = tc["expected_evidence"]
        key_terms = tc["key_terms"]
        expect_missing = tc["expect_missing"]

        print(f"\n[{idx}/15] {q_id} ({category}): '{query}'")
        t0 = time.perf_counter()

        resp = send_chat_turn(
            question=query,
            history=history,
            session_id=f"audit_session_{q_id}",
            institution_id="iitj",
        )
        latency = time.perf_counter() - t0
        diag = resp.diagnostics or {}

        # Extract Panels
        p1 = diag.get("panel_query_understanding", {})
        p2 = diag.get("panel_dense_retrieval", {})
        p3 = diag.get("panel_bm25_retrieval", {})
        p4 = diag.get("panel_hybrid_fusion", {})
        p5 = diag.get("panel_ranking_verification", {})
        p6 = diag.get("panel_packaged_context", {})
        p7 = diag.get("panel_answer_generation", {})

        dense_cands = p2.get("candidates", [])
        bm25_cands = p3.get("candidates", [])
        fused_cands = p4.get("top_15_candidates", [])
        decisions = p5.get("decisions", [])
        packaged_units = p6.get("packaged_units", [])

        # Evaluate evidence presence
        def has_evidence(text_list: list[str]) -> bool:
            if not text_list:
                return False
            combined = " ".join(text_list).casefold()
            # Must contain at least half of key terms
            matches = sum(1 for term in key_terms if term.casefold() in combined)
            return matches >= max(1, len(key_terms) // 2)

        dense_texts = [d.get("text", "") for d in dense_cands]
        bm25_texts = [b.get("text", "") for b in bm25_cands]
        fused_texts = [f.get("text_snippet", "") for f in fused_cands]
        context_texts = [u.get("text", "") for u in packaged_units]

        dense_found = has_evidence(dense_texts)
        bm25_found = has_evidence(bm25_texts)
        fused_found = has_evidence(fused_texts)
        context_found = has_evidence(context_texts)

        # Verified present check
        verified_found = any(
            d.get("verification_status") == "verified" and has_evidence([d.get("source", "")])
            for d in decisions
        ) or (fused_found and p5.get("verified_count", 0) > 0)

        # Outcome determination
        if expect_missing:
            if resp.is_safe_refusal or "not" in resp.answer.casefold() or "do not have" in resp.answer.casefold() or not packaged_units:
                outcome = "Safe Abstention (Expected)"
                failure_cls = "None (Correct Corpus Boundary)"
            else:
                outcome = "Hallucination Risk"
                failure_cls = "Overgeneration on Missing Corpus Evidence"
        else:
            if context_found and len(packaged_units) > 0:
                outcome = "Success (Packaged)"
                failure_cls = "None (Evidence Retained)"
            elif fused_found and not context_found:
                outcome = "Filtered/Dropped Downstream"
                failure_cls = "Ranking or Verification Suppression"
            elif (dense_found or bm25_found) and not fused_found:
                outcome = "Fusion Dropout"
                failure_cls = "RRF Fusion Cutoff"
            elif not dense_found and not bm25_found:
                outcome = "Retrieval Miss"
                failure_cls = "Dual-Branch Retrieval Miss or Corpus Missing"
            else:
                outcome = "Partial"
                failure_cls = "Partial Retrieval"

        print(f"    Outcome: {outcome} | Dense: {dense_found} | BM25: {bm25_found} | Fused: {fused_found} | Context: {context_found} | Latency: {latency:.2f}s")

        # Save Matrix Row
        matrix_rows.append({
            "test_id": q_id,
            "category": category,
            "query": query,
            "requested_intents": intents,
            "expected_evidence": expected,
            "retrieval_outcome": outcome,
            "dense_recall": "Present" if dense_found else "Absent",
            "bm25_recall": "Present" if bm25_found else "Absent",
            "fusion_top15_present": "Present" if fused_found else "Absent",
            "verified_present": "Present" if verified_found else "Absent",
            "final_context_present": "Present" if context_found else "Absent",
            "failure_classification": failure_cls,
            "latency_seconds": round(latency, 3),
            "is_safe_refusal": resp.is_safe_refusal,
            "packaged_units_count": len(packaged_units),
            "answer_preview": resp.answer[:120].replace("\n", " "),
        })

        # Save Structured Traces
        handoff_traces.append({
            "test_id": q_id,
            "query": query,
            "panel_1_query_understanding": p1,
        })

        hybrid_traces.append({
            "test_id": q_id,
            "query": query,
            "dense_candidate_count": len(dense_cands),
            "dense_candidates": dense_cands[:10],
            "bm25_candidate_count": len(bm25_cands),
            "bm25_candidates": bm25_cands[:10],
            "fused_candidate_count": p4.get("total_fused_count", 0),
            "fused_top_15": fused_cands,
        })

        ranking_traces.append({
            "test_id": q_id,
            "query": query,
            "verified_count": p5.get("verified_count", 0),
            "uncertain_count": p5.get("uncertain_count", 0),
            "rejected_count": p5.get("rejected_count", 0),
            "ranked_count": p5.get("ranked_count", 0),
            "decisions": decisions,
        })

        context_traces.append({
            "test_id": q_id,
            "query": query,
            "packaged_units_count": len(packaged_units),
            "estimated_tokens": p6.get("estimated_tokens", 0),
            "token_budget": p6.get("token_budget", 4000),
            "token_budget_percentage": p6.get("token_budget_percentage", 0.0),
            "packaged_units": packaged_units,
            "prompt_messages": p6.get("prompt_messages", []),
            "final_answer": resp.answer,
        })

    # Write retrieval_baseline_matrix.csv
    csv_path = OUTPUT_DIR / "retrieval_baseline_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)
    print(f"\n[OK] Wrote baseline matrix to: {csv_path}")

    # Write JSON traces
    t1_path = OUTPUT_DIR / "query_understanding_handoff_traces.json"
    with open(t1_path, "w", encoding="utf-8") as fp:
        json.dump(handoff_traces, fp, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote query understanding traces to: {t1_path}")

    t2_path = OUTPUT_DIR / "hybrid_retrieval_traces.json"
    with open(t2_path, "w", encoding="utf-8") as fp:
        json.dump(hybrid_traces, fp, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote hybrid retrieval traces to: {t2_path}")

    t3_path = OUTPUT_DIR / "ranking_verification_decisions.json"
    with open(t3_path, "w", encoding="utf-8") as fp:
        json.dump(ranking_traces, fp, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote ranking & verification traces to: {t3_path}")

    t4_path = OUTPUT_DIR / "final_llm_context_traces.json"
    with open(t4_path, "w", encoding="utf-8") as fp:
        json.dump(context_traces, fp, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote final context traces to: {t4_path}")

    print("\nBaseline evaluation completed successfully!")


if __name__ == "__main__":
    run_baseline_evaluation()
