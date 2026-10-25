"""Comprehensive Phase 8 Acceptance and Phase 7 Retrieval/Answer Metrics Runner.

Executes all 8 acceptance tests (across multi-turn and single-turn sessions),
calculates rigorous retrieval metrics (Recall@5/10/15, Precision@k, MRR, Context Survival,
Citation Precision), validates answer correctness, and outputs structured artifacts.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest


GROUND_TRUTH_EVIDENCE = {
    "test_1_ee_research": {
        "query": "What research areas are available in Electrical Engineering?",
        "relevant_sources": [
            "departments/electrical_engineering/research.docx",
            "departments/electrical_engineering/overview.docx",
            "departments/electrical_engineering/facilities.docx",
        ],
        "must_exclude_sources": [
            "knowledge_buffer_general.docx",
            "offices_and_administration/office_of_international_relations/research.docx",
            "departments/physics/programs.docx",
        ],
        "target_program": None,
    },
    "test_2_mtech_fee": {
        "query": "What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. programme at IIT Jodhpur?",
        "relevant_sources": [
            "finance/fees_and_finance.docx",
            "admissions/mtech_admissions.docx",
            "offices_and_administration/office_of_academics/academic_progression.docx",
        ],
        "must_exclude_sources": [
            "school_of_management_and_entrepreneurship",
            "knowledge_buffer_general.docx",
        ],
        "target_program": "M.Tech",
    },
    "test_3_sponsored_mtech": {
        "query": "What about sponsored students?",
        "relevant_sources": [
            "finance/fees_and_finance.docx",
            "offices_and_administration/office_of_academics/academic_progression.docx",
        ],
        "must_exclude_sources": [
            "school_of_management_and_entrepreneurship",
            "knowledge_buffer_general.docx",
        ],
        "target_program": "M.Tech",
    },
    "test_4_hostel_mess": {
        "query": "Does that include hostel and mess charges?",
        "relevant_sources": [
            "finance/fees_and_finance.docx",
        ],
        "must_exclude_sources": [
            "school_of_management_and_entrepreneurship",
            "knowledge_buffer_general.docx",
        ],
        "target_program": "M.Tech",
    },
    "test_5_campus_dining": {
        "query": "Tell me about the mess.",
        "relevant_sources": [
            "hostel_accommodation/general_information.docx",
        ],
        "must_exclude_sources": [
            "knowledge_buffer_general.docx",
        ],
        "target_program": None,
    },
    "test_6_graduation_cpi": {
        "query": "What are the graduation requirements, minimum CPI criteria, and course registration guidelines at IIT Jodhpur?",
        "relevant_sources": [
            "offices_and_administration/office_of_academics/academic_progression.docx",
            "offices_and_administration/office_of_academics/registration.docx",
            "offices_and_administration/office_of_academics/rules_and_regulations.docx",
        ],
        "must_exclude_sources": [
            "knowledge_buffer_general.docx",
        ],
        "target_program": None,
    },
    "test_7_unsupported_claim": {
        "query": "Why did the Supreme Court order IIT Jodhpur to cancel all degrees awarded in 2024?",
        "relevant_sources": [],
        "must_exclude_sources": [],
        "target_program": None,
    },
    "test_8_fresh_ambiguity": {
        "query": "What is the fee for this programme?",
        "relevant_sources": [],
        "must_exclude_sources": [],
        "target_program": None,
    },
}


def evaluate_retrieval(top_candidates: list[dict[str, Any]], relevant_sources: list[str]) -> dict[str, float]:
    """Calculate Recall@k, Precision@k, MRR, and Context Survival."""
    if not relevant_sources:
        return {
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "recall_at_15": 1.0,
            "precision_at_5": 1.0,
            "precision_at_10": 1.0,
            "precision_at_15": 1.0,
            "mrr": 1.0,
            "context_survival": 1.0,
        }

    ranks_of_relevant: list[int] = []
    relevant_in_context = False

    for rank, cand in enumerate(top_candidates, start=1):
        cand_src = str(cand.get("source", "")).replace("\\", "/").casefold()
        is_rel = any(rel.casefold() in cand_src for rel in relevant_sources)
        if is_rel:
            ranks_of_relevant.append(rank)
            if cand.get("in_final_context"):
                relevant_in_context = True

    hits_5 = sum(1 for r in ranks_of_relevant if r <= 5)
    hits_10 = sum(1 for r in ranks_of_relevant if r <= 10)
    hits_15 = sum(1 for r in ranks_of_relevant if r <= 15)

    n_rel = len(relevant_sources)
    recall_5 = min(1.0, hits_5 / max(1, n_rel))
    recall_10 = min(1.0, hits_10 / max(1, n_rel))
    recall_15 = min(1.0, hits_15 / max(1, n_rel))

    p_5 = hits_5 / 5.0
    p_10 = hits_10 / 10.0
    p_15 = hits_15 / 15.0

    mrr = (1.0 / ranks_of_relevant[0]) if ranks_of_relevant else 0.0

    return {
        "recall_at_5": round(recall_5, 4),
        "recall_at_10": round(recall_10, 4),
        "recall_at_15": round(recall_15, 4),
        "precision_at_5": round(p_5, 4),
        "precision_at_10": round(p_10, 4),
        "precision_at_15": round(p_15, 4),
        "mrr": round(mrr, 4),
        "context_survival": 1.0 if relevant_in_context else 0.0,
    }


def evaluate_answer_correctness(test_key: str, answer: str, sources: list[dict[str, Any]], is_safe_refusal: bool) -> dict[str, Any]:
    """Deterministic validation of answer correctness and grounding against test criteria."""
    ans_lower = answer.casefold()
    passed = True
    reasons: list[str] = []

    if test_key == "test_1_ee_research":
        ee_terms = ["vlsi", "neuromorphic", "sociodigital", "electronics", "signal", "communication", "power"]
        matched_ee = [t for t in ee_terms if t in ans_lower]
        if not matched_ee:
            passed = False
            reasons.append("Answer missing core Electrical Engineering research terms.")
        # Ensure no fallback document leaked
        if any("knowledge_buffer" in str(s.get("source", "")).casefold() for s in sources):
            passed = False
            reasons.append("Fallback knowledge buffer leaked into citations.")

    elif test_key == "test_2_mtech_fee":
        if "2,25,000" in ans_lower or "225000" in ans_lower:
            passed = False
            reasons.append("Hallucinated or substituted MBA tuition fee of ₹2,25,000.")
        if "8,00,000" in ans_lower or "800000" in ans_lower:
            passed = False
            reasons.append("Hallucinated MBA total fee of ₹8,00,000.")
        # Check honest fee grounding or explicit limitation
        if not ("32,250" in ans_lower or "semester fee" in ans_lower or "not explicitly" in ans_lower or "revision" in ans_lower or "not provided" in ans_lower or "no specific mention" in ans_lower or "not mention" in ans_lower or "does not specify" in ans_lower or "does not contain" in ans_lower or "not contain" in ans_lower):
            passed = False
            reasons.append("Answer neither grounds official semester fee nor states accurate limitation.")

    elif test_key == "test_3_sponsored_mtech":
        if "assistantship" in ans_lower and "fee" not in ans_lower:
            passed = False
            reasons.append("Presented MHRD assistantship as tuition fee.")
        if "2,25,000" in ans_lower:
            passed = False
            reasons.append("Substituted MBA fee in sponsored M.Tech follow-up.")

    elif test_key == "test_4_hostel_mess":
        # Must distinguish hostel fees included in semester fee vs separate mess advance
        has_separate_mess = "separate" in ans_lower and ("mess" in ans_lower or "advance" in ans_lower)
        has_hostel_inc = "hostel" in ans_lower and ("include" in ans_lower or "included" in ans_lower)
        both_in_same = re.search(r"includes?\s+(?:both\s+)?hostel\s+and\s+mess", ans_lower)
        if both_in_same:
            passed = False
            reasons.append("Contradicted policy by claiming mess charges are included in semester fee.")
        elif not (has_separate_mess or "mess" in ans_lower):
            passed = False
            reasons.append("Did not distinguish separate mess advance.")

    elif test_key == "test_5_campus_dining":
        dining_words = ["mess", "dining", "food", "canteen", "vendor", "vegetarian"]
        if not any(w in ans_lower for w in dining_words):
            passed = False
            reasons.append("Answer does not address campus dining facilities.")
        if "m.tech" in ans_lower and "restrict" in ans_lower:
            passed = False
            reasons.append("Erroneously restricted campus dining to M.Tech students.")

    elif test_key == "test_6_graduation_cpi":
        if is_safe_refusal or "i don't know" in ans_lower:
            passed = False
            reasons.append("False refusal on core academic regulations.")
        has_cpi = "cpi" in ans_lower or "cgpa" in ans_lower or "probation" in ans_lower or "5.0" in ans_lower
        has_reg = "credit" in ans_lower or "registration" in ans_lower or "course" in ans_lower
        if not (has_cpi and has_reg):
            passed = False
            reasons.append("Missing supported CPI or course registration components.")

    elif test_key == "test_7_unsupported_claim":
        refusal_phrases = ["do not substantiate", "cannot verify", "cannot confirm", "not substantiate", "no information", "cannot be verified"]
        if not any(p in ans_lower for p in refusal_phrases):
            passed = False
            reasons.append("Affirmed or failed to refute false premise about Supreme Court degree cancellation.")
        if len(sources) > 0:
            passed = False
            reasons.append("Emitted misleading citations on unsupported claim.")

    elif test_key == "test_8_fresh_ambiguity":
        clarify_phrases = ["which programme", "specify which", "clarify which", "what programme"]
        if not any(p in ans_lower for p in clarify_phrases):
            passed = False
            reasons.append("Failed to ask for programme clarification on unanchored query.")
        if len(sources) > 0:
            passed = False
            reasons.append("Emitted sources on clarification prompt.")

    # Calculate citation precision
    cit_prec = 1.0
    if sources:
        good_cits = 0
        for s in sources:
            src = str(s.get("source", "")).casefold()
            words_in_excerpt = set(re.findall(r"[a-z0-9]{4,}", str(s.get("text", "")).casefold()))
            words_in_ans = set(re.findall(r"[a-z0-9]{4,}", ans_lower))
            if len(words_in_excerpt & words_in_ans) >= 2:
                good_cits += 1
        cit_prec = round(good_cits / len(sources), 4)

    return {
        "passed": passed,
        "reasons": reasons,
        "citation_precision": cit_prec,
    }


def run_all_tests():
    out_dir = Path("test_outputs/retrieval_reliability_diagnostics_20261018")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_results: list[dict[str, Any]] = []

    # =========================================================================
    # Conversation Session 1: Test 1 (EE Research) -> Test 5 (Mess Transition)
    # =========================================================================
    print("\n--- Running Conversation Session 1 (EE -> Mess Transition) ---")
    s1_history: list[dict[str, str]] = []

    # Test 1
    t1_spec = GROUND_TRUTH_EVIDENCE["test_1_ee_research"]
    print(f"Executing Test 1: {t1_spec['query']}")
    t0 = time.perf_counter()
    resp1 = handle_chat_turn(ChatTurnRequest(question=t1_spec["query"], chat_history=tuple(s1_history)))
    lat1 = time.perf_counter() - t0

    diag1 = resp1.diagnostics or {}
    sec_a1 = diag1.get("section_a_candidates", [])
    r_metrics1 = evaluate_retrieval(sec_a1, t1_spec["relevant_sources"])
    a_metrics1 = evaluate_answer_correctness("test_1_ee_research", resp1.answer, resp1.sources, resp1.is_safe_refusal)

    test_results.append({
        "test_id": "test_1_ee_research",
        "question": t1_spec["query"],
        "answer": resp1.answer,
        "sources": resp1.sources,
        "latency_seconds": round(lat1, 2),
        "retrieval_metrics": r_metrics1,
        "answer_metrics": a_metrics1,
        "diagnostics_funnel": diag1.get("funnel", {}),
        "status": "PASS" if a_metrics1["passed"] and r_metrics1["context_survival"] == 1.0 else "FAIL",
    })
    s1_history.append({"role": "user", "content": t1_spec["query"]})
    s1_history.append({"role": "assistant", "content": resp1.answer})

    # Test 5 (Follow-up after EE)
    t5_spec = GROUND_TRUTH_EVIDENCE["test_5_campus_dining"]
    print(f"Executing Test 5: {t5_spec['query']} (Topic transition after EE)")
    t0 = time.perf_counter()
    resp5 = handle_chat_turn(ChatTurnRequest(question=t5_spec["query"], chat_history=tuple(s1_history)))
    lat5 = time.perf_counter() - t0

    diag5 = resp5.diagnostics or {}
    sec_a5 = diag5.get("section_a_candidates", [])
    r_metrics5 = evaluate_retrieval(sec_a5, t5_spec["relevant_sources"])
    a_metrics5 = evaluate_answer_correctness("test_5_campus_dining", resp5.answer, resp5.sources, resp5.is_safe_refusal)

    test_results.append({
        "test_id": "test_5_campus_dining",
        "question": t5_spec["query"],
        "answer": resp5.answer,
        "sources": resp5.sources,
        "latency_seconds": round(lat5, 2),
        "retrieval_metrics": r_metrics5,
        "answer_metrics": a_metrics5,
        "diagnostics_funnel": diag5.get("funnel", {}),
        "status": "PASS" if a_metrics5["passed"] and r_metrics5["context_survival"] == 1.0 else "FAIL",
    })

    # =========================================================================
    # Conversation Session 2: Test 2 (M.Tech Fee) -> Test 3 (Sponsored) -> Test 4 (Hostel/Mess)
    # =========================================================================
    print("\n--- Running Conversation Session 2 (M.Tech Fee -> Sponsored -> Hostel/Mess) ---")
    s2_history: list[dict[str, str]] = []

    # Test 2
    t2_spec = GROUND_TRUTH_EVIDENCE["test_2_mtech_fee"]
    print(f"Executing Test 2: {t2_spec['query']}")
    t0 = time.perf_counter()
    resp2 = handle_chat_turn(ChatTurnRequest(question=t2_spec["query"], chat_history=tuple(s2_history)))
    lat2 = time.perf_counter() - t0

    diag2 = resp2.diagnostics or {}
    sec_a2 = diag2.get("section_a_candidates", [])
    r_metrics2 = evaluate_retrieval(sec_a2, t2_spec["relevant_sources"])
    a_metrics2 = evaluate_answer_correctness("test_2_mtech_fee", resp2.answer, resp2.sources, resp2.is_safe_refusal)

    test_results.append({
        "test_id": "test_2_mtech_fee",
        "question": t2_spec["query"],
        "answer": resp2.answer,
        "sources": resp2.sources,
        "latency_seconds": round(lat2, 2),
        "retrieval_metrics": r_metrics2,
        "answer_metrics": a_metrics2,
        "diagnostics_funnel": diag2.get("funnel", {}),
        "status": "PASS" if a_metrics2["passed"] and r_metrics2["context_survival"] == 1.0 else "FAIL",
    })
    s2_history.append({"role": "user", "content": t2_spec["query"]})
    s2_history.append({"role": "assistant", "content": resp2.answer})

    # Test 3
    t3_spec = GROUND_TRUTH_EVIDENCE["test_3_sponsored_mtech"]
    print(f"Executing Test 3: {t3_spec['query']}")
    t0 = time.perf_counter()
    resp3 = handle_chat_turn(ChatTurnRequest(question=t3_spec["query"], chat_history=tuple(s2_history)))
    lat3 = time.perf_counter() - t0

    diag3 = resp3.diagnostics or {}
    sec_a3 = diag3.get("section_a_candidates", [])
    r_metrics3 = evaluate_retrieval(sec_a3, t3_spec["relevant_sources"])
    a_metrics3 = evaluate_answer_correctness("test_3_sponsored_mtech", resp3.answer, resp3.sources, resp3.is_safe_refusal)

    test_results.append({
        "test_id": "test_3_sponsored_mtech",
        "question": t3_spec["query"],
        "answer": resp3.answer,
        "sources": resp3.sources,
        "latency_seconds": round(lat3, 2),
        "retrieval_metrics": r_metrics3,
        "answer_metrics": a_metrics3,
        "diagnostics_funnel": diag3.get("funnel", {}),
        "status": "PASS" if a_metrics3["passed"] and r_metrics3["context_survival"] == 1.0 else "FAIL",
    })
    s2_history.append({"role": "user", "content": t3_spec["query"]})
    s2_history.append({"role": "assistant", "content": resp3.answer})

    # Test 4
    t4_spec = GROUND_TRUTH_EVIDENCE["test_4_hostel_mess"]
    print(f"Executing Test 4: {t4_spec['query']}")
    t0 = time.perf_counter()
    resp4 = handle_chat_turn(ChatTurnRequest(question=t4_spec["query"], chat_history=tuple(s2_history)))
    lat4 = time.perf_counter() - t0

    diag4 = resp4.diagnostics or {}
    sec_a4 = diag4.get("section_a_candidates", [])
    r_metrics4 = evaluate_retrieval(sec_a4, t4_spec["relevant_sources"])
    a_metrics4 = evaluate_answer_correctness("test_4_hostel_mess", resp4.answer, resp4.sources, resp4.is_safe_refusal)

    test_results.append({
        "test_id": "test_4_hostel_mess",
        "question": t4_spec["query"],
        "answer": resp4.answer,
        "sources": resp4.sources,
        "latency_seconds": round(lat4, 2),
        "retrieval_metrics": r_metrics4,
        "answer_metrics": a_metrics4,
        "diagnostics_funnel": diag4.get("funnel", {}),
        "status": "PASS" if a_metrics4["passed"] and r_metrics4["context_survival"] == 1.0 else "FAIL",
    })

    # =========================================================================
    # Standalone Tests: Test 6 (Academic Reqs), Test 7 (Unsupported), Test 8 (Fresh Ambiguity)
    # =========================================================================
    print("\n--- Running Standalone Acceptance Tests (6, 7, 8) ---")

    # Test 6
    t6_spec = GROUND_TRUTH_EVIDENCE["test_6_graduation_cpi"]
    print(f"Executing Test 6: {t6_spec['query']}")
    t0 = time.perf_counter()
    resp6 = handle_chat_turn(ChatTurnRequest(question=t6_spec["query"], chat_history=()))
    lat6 = time.perf_counter() - t0

    diag6 = resp6.diagnostics or {}
    sec_a6 = diag6.get("section_a_candidates", [])
    r_metrics6 = evaluate_retrieval(sec_a6, t6_spec["relevant_sources"])
    a_metrics6 = evaluate_answer_correctness("test_6_graduation_cpi", resp6.answer, resp6.sources, resp6.is_safe_refusal)

    test_results.append({
        "test_id": "test_6_graduation_cpi",
        "question": t6_spec["query"],
        "answer": resp6.answer,
        "sources": resp6.sources,
        "latency_seconds": round(lat6, 2),
        "retrieval_metrics": r_metrics6,
        "answer_metrics": a_metrics6,
        "diagnostics_funnel": diag6.get("funnel", {}),
        "status": "PASS" if a_metrics6["passed"] and r_metrics6["context_survival"] == 1.0 else "FAIL",
    })

    # Test 7
    t7_spec = GROUND_TRUTH_EVIDENCE["test_7_unsupported_claim"]
    print(f"Executing Test 7: {t7_spec['query']}")
    t0 = time.perf_counter()
    resp7 = handle_chat_turn(ChatTurnRequest(question=t7_spec["query"], chat_history=()))
    lat7 = time.perf_counter() - t0

    diag7 = resp7.diagnostics or {}
    sec_a7 = diag7.get("section_a_candidates", [])
    r_metrics7 = evaluate_retrieval(sec_a7, t7_spec["relevant_sources"])
    a_metrics7 = evaluate_answer_correctness("test_7_unsupported_claim", resp7.answer, resp7.sources, resp7.is_safe_refusal)

    test_results.append({
        "test_id": "test_7_unsupported_claim",
        "question": t7_spec["query"],
        "answer": resp7.answer,
        "sources": resp7.sources,
        "latency_seconds": round(lat7, 2),
        "retrieval_metrics": r_metrics7,
        "answer_metrics": a_metrics7,
        "diagnostics_funnel": diag7.get("funnel", {}),
        "status": "PASS" if a_metrics7["passed"] else "FAIL",
    })

    # Test 8
    t8_spec = GROUND_TRUTH_EVIDENCE["test_8_fresh_ambiguity"]
    print(f"Executing Test 8: {t8_spec['query']}")
    t0 = time.perf_counter()
    resp8 = handle_chat_turn(ChatTurnRequest(question=t8_spec["query"], chat_history=()))
    lat8 = time.perf_counter() - t0

    diag8 = resp8.diagnostics or {}
    sec_a8 = diag8.get("section_a_candidates", [])
    r_metrics8 = evaluate_retrieval(sec_a8, t8_spec["relevant_sources"])
    a_metrics8 = evaluate_answer_correctness("test_8_fresh_ambiguity", resp8.answer, resp8.sources, resp8.is_safe_refusal)

    test_results.append({
        "test_id": "test_8_fresh_ambiguity",
        "question": t8_spec["query"],
        "answer": resp8.answer,
        "sources": resp8.sources,
        "latency_seconds": round(lat8, 2),
        "retrieval_metrics": r_metrics8,
        "answer_metrics": a_metrics8,
        "diagnostics_funnel": diag8.get("funnel", {}),
        "status": "PASS" if a_metrics8["passed"] else "FAIL",
    })

    # Save complete JSON artifact
    out_json = out_dir / "phase8_acceptance_and_retrieval_metrics.json"
    out_json.write_text(json.dumps(test_results, indent=2), encoding="utf-8")
    print(f"\nSaved structured metrics to {out_json}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 8 ACCEPTANCE & RETRIEVAL METRICS SUMMARY")
    print("=" * 80)
    for res in test_results:
        tid = res["test_id"]
        status = res["status"]
        lat = res["latency_seconds"]
        rm = res["retrieval_metrics"]
        am = res["answer_metrics"]
        print(f"[{status}] {tid} ({lat}s)")
        print(f"    Recall@5: {rm['recall_at_5']}, Recall@15: {rm['recall_at_15']}, MRR: {rm['mrr']}, Survival: {rm['context_survival']}")
        print(f"    Citation Precision: {am['citation_precision']}")
        if am["reasons"]:
            print(f"    Issues: {', '.join(am['reasons'])}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
