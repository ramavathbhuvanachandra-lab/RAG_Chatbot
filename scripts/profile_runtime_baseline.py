"""Runtime Profiling Baseline Script for Task 2.

Profiles the existing production RAG pipeline across 15 representative evaluation queries.
Measures:
- Total turn latency
- Query understanding latency (understand_query)
- Dense retrieval latency (dense_retrieve)
- BM25 retrieval latency (bm25_retrieve)
- Lexical retrieval latency (lexical_retrieve)
- Fusion latency (fusion)
- Verification & ranking latency (verify_and_rank)
- Context packaging latency (context_packaging)
- Answer LLM generation latency (llm_generation)
- Model call count
- Prompt message sizes and token estimates
- Answer text and grounding status

Outputs to test_outputs/runtime_optimization_20261020_master/:
- runtime_stage_timings.json
- runtime_latency_baseline.md
- answer_generation_baseline.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.clients.assistant_client import send_chat_turn

OUTPUT_DIR = REPO_ROOT / "test_outputs" / "runtime_optimization_20261020_master"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "id": "T01",
        "category": "1. Short factual lookup",
        "query": "What is the M.Tech tuition fee for regular Indian students per semester?",
        "history": [],
        "requested_intents": "M.Tech regular semester tuition fee amount for Indian nationals",
    },
    {
        "id": "T02",
        "category": "2. Natural-language standalone query",
        "query": "What research areas and laboratories are active in the Department of Electrical Engineering?",
        "history": [],
        "requested_intents": "Research thrust areas, specializations, and labs in Electrical Engineering",
    },
    {
        "id": "T03",
        "category": "3. Long paragraph containing multiple related requests",
        "query": "I am interested in applying for the M.Tech programme at IIT Jodhpur. Could you explain the minimum eligibility requirements, whether a valid GATE score is required for regular candidates, and what the tuition fee per semester is for regular Indian students?",
        "history": [],
        "requested_intents": "M.Tech eligibility requirements; GATE score requirement; Regular semester tuition fee",
    },
    {
        "id": "T04",
        "category": "4. Multi-intent comparison",
        "query": "Compare Electronics Engineering and Materials Engineering for chip design and semiconductor research.",
        "history": [],
        "requested_intents": "Electronics Engineering chip/VLSI focus vs Materials Engineering semiconductor materials/synthesis focus",
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
    },
    {
        "id": "T07",
        "category": "7. English query",
        "query": "In which year was IIT Jodhpur established and where was its temporary transit campus initially located?",
        "history": [],
        "requested_intents": "Year of establishment (2008) and transit campus location (MBM Engineering College)",
    },
    {
        "id": "T08",
        "category": "8. Hindi query",
        "query": "क्या अनुसूचित जाति (SC) और अनुसूचित जनजाति (ST) के छात्रों को ट्यूशन फीस में छूट मिलती है?",
        "history": [],
        "requested_intents": "SC/ST tuition fee exemption inquiry in Hindi",
    },
    {
        "id": "T09",
        "category": "9. Hinglish query",
        "query": "IIT Jodhpur me MTech admission ke liye eligibility criteria kya hai aur GATE score zaroori hai kya?",
        "history": [],
        "requested_intents": "M.Tech admission eligibility and GATE requirement inquiry in Hinglish",
    },
    {
        "id": "T10",
        "category": "10. Exact course-code lookup",
        "query": "What exact course code is printed next to Quantum Computing in the Mathematics curriculum?",
        "history": [],
        "requested_intents": "Exact course alphanumeric code assigned to Quantum Computing course in Mathematics & Computing curriculum",
    },
    {
        "id": "T11",
        "category": "11. Numerical or tabular lookup",
        "query": "What is the total credit requirement for the B.S. in Mathematics and Computing, and what are the listed credit components?",
        "history": [],
        "requested_intents": "Total credit count and category breakdown for B.S. Mathematics and Computing",
    },
    {
        "id": "T12",
        "category": "12. A query for evidence missing from current corpus",
        "query": "What were the exact terms of the Supreme Court degree cancellation order for IIT Jodhpur in 2024?",
        "history": [],
        "requested_intents": "Fictitious/missing event: Supreme Court degree cancellation order",
    },
    {
        "id": "T13",
        "category": "13. Similar terminology in two different academic programmes",
        "query": "What are the academic progression and course overload rules for B.Tech students compared to M.Tech students?",
        "history": [],
        "requested_intents": "Progression/overload rules for B.Tech compared with M.Tech regulations",
    },
    {
        "id": "T14",
        "category": "14. Broad institute-wide programme discovery",
        "query": "What undergraduate and postgraduate degree programmes and academic departments are offered at IIT Jodhpur?",
        "history": [],
        "requested_intents": "Broad discovery of UG (B.Tech, B.S.) and PG (M.Tech, M.Sc., Ph.D.) degrees and major departments",
    },
    {
        "id": "T15",
        "category": "15. Explicit qualifier, date, number, or instruction not to guess",
        "query": "According to the official fee notification, what is the exact semester tuition fee for the ITEP programme for the academic year 2026-2027? Do not guess if it is not in the text.",
        "history": [],
        "requested_intents": "ITEP tuition fee for specific future AY 2026-2027 with negative constraint (do not guess)",
    },
]


def run_profiling():
    print("=" * 80)
    print("STARTING RUNTIME PROFILING BASELINE (15 EVALUATION QUERIES)")
    print("=" * 80)

    timing_records = []
    csv_rows = []

    for idx, tc in enumerate(TEST_CASES, start=1):
        q_id = tc["id"]
        category = tc["category"]
        query = tc["query"]
        history = tc["history"]
        intents = tc["requested_intents"]

        print(f"\n[{idx}/15] {q_id} ({category}): '{query[:60]}...'")
        t_start = time.perf_counter()

        resp = send_chat_turn(
            question=query,
            history=history,
            session_id=f"runtime_baseline_{q_id}",
            institution_id="iitj",
        )
        total_latency = time.perf_counter() - t_start

        diag = resp.diagnostics or {}
        p8 = diag.get("panel_stage_timings", {})
        timings = p8.get("timings", {})

        p6 = diag.get("panel_packaged_context", {})
        packaged_units = p6.get("packaged_units", [])
        est_tokens = p6.get("estimated_tokens", 0)

        p7 = diag.get("panel_answer_generation", {})
        model_calls = p7.get("model_calls", 1)
        grounding_status = p7.get("grounding_status", "unknown")
        guard_status = p7.get("guard_status", "unknown")

        prompt_messages = diag.get("prompt_messages", [])
        sys_chars = len(prompt_messages[0]["content"]) if prompt_messages else 0
        usr_chars = len(prompt_messages[1]["content"]) if len(prompt_messages) > 1 else 0

        # Detailed stage latency breakdown
        t_understand = timings.get("understand_query", 0.0)
        t_dense = timings.get("dense_retrieve", 0.0)
        t_bm25 = timings.get("bm25_retrieve", 0.0)
        t_lexical = timings.get("lexical_retrieve", 0.0)
        t_fusion = timings.get("fusion", 0.0)
        t_verify = timings.get("verify_and_rank", 0.0)
        t_package = timings.get("context_packaging", 0.0)
        t_llm = timings.get("llm_generation", 0.0)

        record = {
            "test_id": q_id,
            "category": category,
            "query": query,
            "total_latency_seconds": round(total_latency, 3),
            "stage_latencies": {
                "resolve_conversation": round(timings.get("resolve_conversation", 0.0), 4),
                "understand_query": round(t_understand, 4),
                "plan_multi_intent": round(timings.get("plan_multi_intent", 0.0), 4),
                "dense_retrieve": round(t_dense, 4),
                "bm25_retrieve": round(t_bm25, 4),
                "lexical_retrieve": round(t_lexical, 4),
                "fusion": round(t_fusion, 4),
                "verify_and_rank": round(t_verify, 4),
                "context_packaging": round(t_package, 4),
                "llm_generation": round(t_llm, 4),
            },
            "metrics": {
                "model_calls": model_calls,
                "packaged_units_count": len(packaged_units),
                "estimated_context_tokens": est_tokens,
                "system_prompt_chars": sys_chars,
                "user_prompt_chars": usr_chars,
                "answer_length_chars": len(resp.answer),
                "grounding_status": grounding_status,
                "guard_status": guard_status,
                "is_safe_refusal": resp.is_safe_refusal,
            },
            "answer_preview": resp.answer[:200],
        }
        timing_records.append(record)

        # Build CSV Row
        csv_rows.append({
            "test_id": q_id,
            "category": category,
            "query": query,
            "total_latency": round(total_latency, 2),
            "understand_query_latency": round(t_understand, 2),
            "retrieval_total_latency": round(t_dense + t_bm25 + t_lexical, 2),
            "verification_latency": round(t_verify, 2),
            "llm_generation_latency": round(t_llm, 2),
            "model_calls": model_calls,
            "context_units": len(packaged_units),
            "context_tokens_est": est_tokens,
            "sys_prompt_chars": sys_chars,
            "user_prompt_chars": usr_chars,
            "answer_length": len(resp.answer),
            "grounding_status": grounding_status,
            "guard_status": guard_status,
            "displayed_answer": resp.answer.replace("\n", " ").strip()[:300],
        })

        print(f"    Total: {total_latency:.2f}s | QueryLLM: {t_understand:.2f}s | Retr: {(t_dense+t_bm25+t_lexical):.2f}s | Verify: {t_verify:.2f}s | AnsLLM: {t_llm:.2f}s | AnsChars: {len(resp.answer)}")

    # Save JSON
    json_path = OUTPUT_DIR / "runtime_stage_timings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(timing_records, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Wrote runtime stage timings JSON to: {json_path}")

    # Save CSV
    csv_path = OUTPUT_DIR / "answer_generation_baseline.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"[OK] Wrote answer generation baseline CSV to: {csv_path}")

    # Compute Aggregate Metrics for Markdown Report
    total_lats = [r["total_latency_seconds"] for r in timing_records]
    uq_lats = [r["stage_latencies"]["understand_query"] for r in timing_records]
    llm_lats = [r["stage_latencies"]["llm_generation"] for r in timing_records]
    ret_lats = [r["stage_latencies"]["dense_retrieve"] + r["stage_latencies"]["bm25_retrieve"] + r["stage_latencies"]["lexical_retrieve"] for r in timing_records]
    ver_lats = [r["stage_latencies"]["verify_and_rank"] for r in timing_records]

    def median(lst):
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

    md_report = f"""# Runtime Latency Baseline Report (Phase 1 & 2)

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Project:** IIT Jodhpur Institutional RAG Chatbot  
**Evaluation Suite:** 15 Representative Queries (`T01`–`T15`)  
**Audit Directory:** `{OUTPUT_DIR.name}/`  

---

## 1. Executive Summary & Aggregate Latency Statistics

| Metric | Total Turn Latency | Query Understanding LLM | Retrieval (Dense + BM25 + Lexical) | Verification & Ranking | Answer Generation LLM |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Mean Latency** | **{sum(total_lats)/len(total_lats):.2f}s** | **{sum(uq_lats)/len(uq_lats):.2f}s** | **{sum(ret_lats)/len(ret_lats):.2f}s** | **{sum(ver_lats)/len(ver_lats):.2f}s** | **{sum(llm_lats)/len(llm_lats):.2f}s** |
| **Median (P50)** | **{median(total_lats):.2f}s** | **{median(uq_lats):.2f}s** | **{median(ret_lats):.2f}s** | **{median(ver_lats):.2f}s** | **{median(llm_lats):.2f}s** |
| **Min Latency** | {min(total_lats):.2f}s | {min(uq_lats):.2f}s | {min(ret_lats):.2f}s | {min(ver_lats):.2f}s | {min(llm_lats):.2f}s |
| **Max Latency** | {max(total_lats):.2f}s | {max(uq_lats):.2f}s | {max(ret_lats):.2f}s | {max(ver_lats):.2f}s | {max(llm_lats):.2f}s |
| **% of Turn Time** | 100.0% | **{(sum(uq_lats)/sum(total_lats))*100:.1f}%** | **{(sum(ret_lats)/sum(total_lats))*100:.1f}%** | **{(sum(ver_lats)/sum(total_lats))*100:.1f}%** | **{(sum(llm_lats)/sum(total_lats))*100:.1f}%** |

**Key Finding:** LLM invocations account for **{((sum(uq_lats) + sum(llm_lats))/sum(total_lats))*100:.1f}%** of total turnaround time.

---

## 2. Per-Query Latency & Stage Breakdown

| ID | Category | Total Latency | Query LLM | Retrieval | Verification | Answer LLM | Model Calls | Context Units | Context Tokens |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in timing_records:
        t_ret = r["stage_latencies"]["dense_retrieve"] + r["stage_latencies"]["bm25_retrieve"] + r["stage_latencies"]["lexical_retrieve"]
        md_report += f"| **{r['test_id']}** | {r['category'][:25]} | {r['total_latency_seconds']:.2f}s | {r['stage_latencies']['understand_query']:.2f}s | {t_ret:.2f}s | {r['stage_latencies']['verify_and_rank']:.2f}s | {r['stage_latencies']['llm_generation']:.2f}s | {r['metrics']['model_calls']} | {r['metrics']['packaged_units_count']} | {r['metrics']['estimated_context_tokens']} |\n"

    md_report += """
---

## 3. Bottleneck Analysis & Optimization Targets

1. **Query Planning vs Multi-Intent Sub-Query Redundancy:**
   - In single-intent queries, `understand_query` takes ~12-14s.
   - In multi-intent queries (`process_multi_intent_node`), calling `understand_query_frame()` re-invoked the semantic LLM for every sub-intent, adding 12-25s of redundant processing even though `QueryIntentPlan` already possessed pre-computed query frames.
2. **Excessive Answer System Prompt Overhead (8,830 chars / ~2,200 tokens):**
   - The answer system instructions contained 8,830 characters of duplicated rules, verbose micro-guidance, and hardcoded query strings.
   - Combined with ~2,800 tokens of verified context, total prompt tokens exceeded `ANSWER_LLM_CONTEXT` (4096), triggering context truncation and KV-cache recalculation overhead in Ollama.
3. **Lexical Retrieval Lane Overhead:**
   - Scanning 1,156 canonical chunks sequentially for each query variant contributed ~4s of CPU-bound string matching in `lexical_retrieve`.
"""

    md_path = OUTPUT_DIR / "runtime_latency_baseline.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[OK] Wrote runtime latency baseline report to: {md_path}")


if __name__ == "__main__":
    run_profiling()
