"""Semantic Accuracy & Reliability Evaluation Runner for Held-Out Benchmarks.

Measures:
1. Retrieval Recall@k (evidence candidate retrieval & verification)
2. Semantic Answer Correctness (factual accuracy, constraint preservation)
3. Material Inversion / Contradiction Rate
4. Correct Abstention & False Abstention Rates
5. End-to-End Latency per Question
Outputs results to an explicit, isolated directory:
`test_outputs/rag_quality/semantic_eval_<timestamp>/`
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any

from ai_platform.runtime.engine import ask_with_diagnostics


ROOT = Path(__file__).resolve().parents[2]
HELD_OUT_FILE = ROOT / "tests_reusable" / "rag_quality" / "held_out_questions.json"
DEFAULT_OUTPUT_BASE = ROOT / "test_outputs" / "rag_quality"


def resolve_output_dir(custom_path: str | None = None) -> Path:
    if custom_path:
        out_dir = Path(custom_path).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (DEFAULT_OUTPUT_BASE / f"semantic_eval_{timestamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def evaluate_response(
    item: dict[str, Any],
    result: dict[str, Any],
    latency_sec: float,
) -> dict[str, Any]:
    question_id = item["id"]
    q_type = item.get("type", "general")
    expected_behavior = item.get("expected_behavior", "answer")
    expected_facts = item.get("expected_key_facts", [])
    forbidden_inversions = item.get("forbidden_inversions", [])

    answer = str(result.get("answer") or "").strip()
    guard_status = str(result.get("answer_guard_status") or "").lower()
    guard_reason = str(result.get("answer_guard_reason") or "")
    evidence_candidates = result.get("evidence_candidates") or ()
    verified_candidates = result.get("verified_candidates") or ()
    is_multi_intent = bool(result.get("is_multi_intent", False))
    grounding = result.get("answer_grounding")
    grounding_status = getattr(grounding, "status", str(grounding or "")).lower()

    # Abstention check
    _answer_lower = answer.lower()
    _ABSTENTION_PHRASES = (
        "i do not have enough information",
        "i don't know",
        "insufficient evidence",
        "does not contain information",
        "cannot find",
        "no information available",
        "not contain information",
        "not mentioned in",
        "not explicitly mentioned",
        "not in the provided",
        "the provided evidence does not",
    )
    abstained = bool(
        "insufficient" in guard_status
        or "review" in guard_status
        or "blocked" in guard_status
        or "outside" in guard_reason.lower()
        or not answer
        or any(phrase in _answer_lower for phrase in _ABSTENTION_PHRASES)
    )

    # Retrieval Recall check: did we retrieve and verify at least one candidate for answerable queries?
    has_retrieved_evidence = len(evidence_candidates) > 0 or len(verified_candidates) > 0
    retrieval_success = (has_retrieved_evidence if expected_behavior == "answer" else True)

    # Inversion / Contradiction check
    inversions_found: list[str] = []
    for inv in forbidden_inversions:
        if inv and re.search(re.escape(inv), answer, re.IGNORECASE):
            inversions_found.append(inv)

    has_inversion = len(inversions_found) > 0

    # Key Facts preservation check
    # Normalize hyphens so "first-class" and "first class" are treated as equivalent.
    def _normalize(s: str) -> str:
        return s.replace("-", " ").replace("_", " ")

    _answer_normalized = _normalize(answer)

    facts_present: list[str] = []
    facts_missing: list[str] = []
    for fact in expected_facts:
        fact_norm = _normalize(fact)
        if re.search(re.escape(fact), answer, re.IGNORECASE) or re.search(
            re.escape(fact_norm), _answer_normalized, re.IGNORECASE
        ):
            facts_present.append(fact)
        else:
            facts_missing.append(fact)

    facts_preserved = (len(facts_missing) == 0) if expected_facts else True

    # Semantic correctness determination
    if expected_behavior == "abstain":
        semantic_correct = abstained and not has_inversion
        correct_abstention = abstained
        false_abstention = False
    else:  # expected_behavior == "answer"
        semantic_correct = (not abstained) and (not has_inversion) and facts_preserved
        correct_abstention = False
        false_abstention = abstained

    return {
        "id": question_id,
        "type": q_type,
        "question": item["question"],
        "expected_behavior": expected_behavior,
        "semantic_correct": semantic_correct,
        "retrieval_success": retrieval_success,
        "abstained": abstained,
        "correct_abstention": correct_abstention,
        "false_abstention": false_abstention,
        "inversions_found": inversions_found,
        "facts_present": facts_present,
        "facts_missing": facts_missing,
        "answer_length": len(answer),
        "guard_status": guard_status,
        "grounding_status": grounding_status,
        "is_multi_intent": is_multi_intent,
        "candidates_count": len(evidence_candidates),
        "verified_count": len(verified_candidates),
        "latency_sec": round(latency_sec, 2),
        "answer_preview": answer[:250] + ("..." if len(answer) > 250 else ""),
    }


def run_benchmark(
    questions_file: Path = HELD_OUT_FILE,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    with open(questions_file, "r", encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)

    out_dir = resolve_output_dir(str(output_dir) if output_dir else None)
    print(f"Running Held-Out Semantic Evaluation on {len(questions)} questions...")
    print(f"Output directory: {out_dir}")

    evaluations: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for idx, q_item in enumerate(questions, 1):
        q_text = q_item["question"]
        print(f"[{idx}/{len(questions)}] {q_item['id']} ... ", end="", flush=True)

        q_start = time.perf_counter()
        try:
            diag_result = ask_with_diagnostics(q_text)
            latency = time.perf_counter() - q_start
            eval_res = evaluate_response(q_item, diag_result, latency)
        except Exception as exc:
            latency = time.perf_counter() - q_start
            eval_res = {
                "id": q_item["id"],
                "type": q_item.get("type", "general"),
                "question": q_text,
                "expected_behavior": q_item.get("expected_behavior", "answer"),
                "semantic_correct": False,
                "retrieval_success": False,
                "abstained": False,
                "correct_abstention": False,
                "false_abstention": False,
                "inversions_found": [],
                "facts_present": [],
                "facts_missing": q_item.get("expected_key_facts", []),
                "answer_length": 0,
                "guard_status": "error",
                "grounding_status": "error",
                "is_multi_intent": False,
                "candidates_count": 0,
                "verified_count": 0,
                "latency_sec": round(latency, 2),
                "error": str(exc),
                "answer_preview": f"ERROR: {exc}",
            }

        status_str = "CORRECT" if eval_res["semantic_correct"] else "INCORRECT"
        print(f"{status_str} ({eval_res['latency_sec']}s)")
        evaluations.append(eval_res)

        # Flush intermediate results after each question for crash/quota resilience
        interim_json = out_dir / "semantic_eval_results.json"
        try:
            with open(interim_json, "w", encoding="utf-8") as f_int:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "total_questions": len(questions),
                    "completed_questions": len(evaluations),
                    "semantic_correct_total": sum(1 for e in evaluations if e["semantic_correct"]),
                    "evaluations": evaluations,
                }, f_int, indent=2)
        except Exception:
            pass

    total_time = time.perf_counter() - total_start

    # Summary metrics
    n_total = len(evaluations)
    n_correct = sum(1 for e in evaluations if e["semantic_correct"])
    n_answerable = sum(1 for e in evaluations if e["expected_behavior"] == "answer")
    n_unanswerable = sum(1 for e in evaluations if e["expected_behavior"] == "abstain")
    
    correct_answers = sum(
        1 for e in evaluations if e["expected_behavior"] == "answer" and e["semantic_correct"]
    )
    correct_abstentions = sum(
        1 for e in evaluations if e["expected_behavior"] == "abstain" and e["correct_abstention"]
    )
    false_abstentions = sum(1 for e in evaluations if e["false_abstention"])
    inversions_count = sum(1 for e in evaluations if len(e.get("inversions_found", [])) > 0)
    retrieval_recalls = sum(1 for e in evaluations if e["retrieval_success"])

    semantic_accuracy_total = round((n_correct / n_total) * 100.0, 2) if n_total else 0.0
    answerable_accuracy = (
        round((correct_answers / n_answerable) * 100.0, 2) if n_answerable else 0.0
    )
    abstention_accuracy = (
        round((correct_abstentions / n_unanswerable) * 100.0, 2) if n_unanswerable else 0.0
    )
    retrieval_recall_rate = (
        round((retrieval_recalls / n_total) * 100.0, 2) if n_total else 0.0
    )
    avg_latency = round(total_time / n_total, 2) if n_total else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": n_total,
        "answerable_questions": n_answerable,
        "unanswerable_or_oos_questions": n_unanswerable,
        "semantic_correct_total": n_correct,
        "semantic_accuracy_total_pct": semantic_accuracy_total,
        "answerable_accuracy_pct": answerable_accuracy,
        "correct_abstention_pct": abstention_accuracy,
        "false_abstentions_count": false_abstentions,
        "inversions_contradictions_count": inversions_count,
        "retrieval_recall_rate_pct": retrieval_recall_rate,
        "total_duration_sec": round(total_time, 2),
        "avg_latency_sec": avg_latency,
        "evaluations": evaluations,
    }

    # Write output files
    json_path = out_dir / "semantic_eval_results.json"
    txt_path = out_dir / "semantic_eval_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("HELD-OUT SEMANTIC ACCURACY & RELIABILITY EVALUATION REPORT\n")
        f.write("======================================================================\n")
        f.write(f"Timestamp:                    {summary['timestamp']}\n")
        f.write(f"Total Questions Evaluated:    {n_total}\n")
        f.write(f"Total Semantic Correctness:   {n_correct}/{n_total} ({semantic_accuracy_total}%)\n")
        f.write(f"Answerable Questions Acc:     {correct_answers}/{n_answerable} ({answerable_accuracy}%)\n")
        f.write(f"Unanswerable Abstention Acc:  {correct_abstentions}/{n_unanswerable} ({abstention_accuracy}%)\n")
        f.write(f"False Abstentions:            {false_abstentions}\n")
        f.write(f"Inversions / Contradictions:  {inversions_count}\n")
        f.write(f"Retrieval Recall@k Rate:      {retrieval_recall_rate}%\n")
        f.write(f"Average Latency:              {avg_latency}s / question\n")
        f.write("======================================================================\n\n")
        f.write("ITEM-BY-ITEM DETAILS:\n")
        for e in evaluations:
            f.write(f"ID: {e['id']} | Type: {e['type']} | Status: {'PASS' if e['semantic_correct'] else 'FAIL'}\n")
            f.write(f"Question: {e['question']}\n")
            f.write(f"Answer: {e['answer_preview']}\n")
            f.write(f"Latency: {e['latency_sec']}s | Guard: {e['guard_status']} | Grounding: {e['grounding_status']}\n")
            if e.get("inversions_found"):
                f.write(f"INVERSIONS: {e['inversions_found']}\n")
            if e.get("facts_missing"):
                f.write(f"MISSING FACTS: {e['facts_missing']}\n")
            f.write("-" * 70 + "\n")

    print("\n" + "=" * 60)
    print("EVALUATION FINISHED")
    print(f"Total Semantic Correctness: {semantic_accuracy_total}% ({n_correct}/{n_total})")
    print(f"Answerable Accuracy: {answerable_accuracy}% ({correct_answers}/{n_answerable})")
    print(f"Abstention Accuracy: {abstention_accuracy}% ({correct_abstentions}/{n_unanswerable})")
    print(f"Retrieval Recall Rate: {retrieval_recall_rate}%")
    print(f"Report written to: {txt_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-Out Semantic Evaluation Runner")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--questions-file", type=str, default=None)
    args = parser.parse_args()
    q_file = Path(args.questions_file) if args.questions_file else HELD_OUT_FILE
    run_benchmark(questions_file=q_file, output_dir=Path(args.output_dir) if args.output_dir else None)
